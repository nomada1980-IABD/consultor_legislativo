#!/usr/bin/env bash
# Arranca el modelo de IA local (Qwen2.5-7B) en la GPU AMD vía llama.cpp.
# Expone una API OpenAI-compatible en :8080.
#
# Uso:   ./start-local-llm.sh
# Logs:  ~/llamacpp/server.log
#
# El script se autodetecta en vez de cablear rutas e índices, porque el
# proyecto vive en un disco portátil y se mueve entre equipos: la ruta del
# build de llama.cpp y el orden de enumeración de las GPUs cambian de máquina
# a máquina. Todo lo autodetectado se puede forzar por variable de entorno.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${LOCAL_LLM_PORT:-8080}"
HOST="${LOCAL_LLM_HOST:-127.0.0.1}"
MODEL="${LOCAL_LLM_HF_MODEL:-bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M}"
CTX="${LOCAL_LLM_CTX:-8192}"

# Caché de modelos GGUF: junto al proyecto, no en ~/.cache/llama.cpp. Cada
# modelo ocupa varios GB y la partición /home va justa; además así el modelo
# viaja en el mismo disco que el proyecto. Está en .gitignore (models/).
export LLAMA_CACHE="${LLAMA_CACHE:-$PROJECT_DIR/models}"
mkdir -p "$LLAMA_CACHE"

# --- 1. Localizar el build de llama.cpp -------------------------------------
# Antes estaba cableado a ~/llamacpp/llama-b9581, que se rompe al actualizar.
# Se coge el build más reciente por número de versión (llama-b<N>).
if [ -n "${LLAMA_DIR:-}" ]; then
  : # respetar el valor del entorno
else
  LLAMA_DIR=$(find "$HOME/llamacpp" -maxdepth 1 -type d -name 'llama-b*' 2>/dev/null \
              | sort -t b -k2 -n | tail -1)
fi

if [ -z "${LLAMA_DIR:-}" ] || [ ! -x "$LLAMA_DIR/llama-server" ]; then
  echo "❌ No se encontró llama-server."
  echo "   Buscado en: ~/llamacpp/llama-b*/llama-server"
  echo "   Descarga un build de llama.cpp o exporta LLAMA_DIR=/ruta/al/build"
  exit 1
fi

# El build enlaza sus .so por ruta relativa; ROCm aporta las suyas si hay
# backend HIP. Se antepone para no depender del LD_LIBRARY_PATH heredado.
export LD_LIBRARY_PATH="$LLAMA_DIR:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

# --- 2. Comprobar que el puerto está libre ----------------------------------
PORT_BUSY=$(ss -tln 2>/dev/null | grep -E ":$PORT\s" || true)
if [ -n "$PORT_BUSY" ]; then
  EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | head -1 || true)
  PID_MSG=""
  if [ -n "$EXISTING_PID" ]; then
    PID_MSG=" (PID $EXISTING_PID)"
  fi
  echo "⚠️  El puerto $PORT ya está en uso${PID_MSG}."
  if [ -n "$EXISTING_PID" ]; then
    echo "    Para detenerlo de forma segura:  kill $EXISTING_PID"
  fi
  exit 1
fi


# --- 3. Elegir la GPU dedicada ----------------------------------------------
# Antes se cableaba --device Vulkan1 asumiendo que Vulkan0 era la iGPU. Ese
# orden no está garantizado. Se elige por nombre: se descartan las iGPU de
# Ryzen (RAPHAEL/MENDOCINO/mismas siglas) y se queda la primera dedicada.
# Funciona igual con backend Vulkan (VulkanN) que con ROCm/HIP (ROCmN).
DEVICES=$("$LLAMA_DIR/llama-server" --list-devices 2>/dev/null | grep -E '^\s+(Vulkan|ROCm|CUDA)[0-9]+:' || true)

if [ -z "$DEVICES" ]; then
  echo "❌ llama-server no detecta ninguna GPU."
  echo "   Comprueba los drivers y que el usuario esté en los grupos 'video' y 'render':"
  echo "     id -nG    # debe incluir video y render"
  exit 1
fi

if [ -n "${LLAMA_DEVICE:-}" ]; then
  DEVICE="$LLAMA_DEVICE"
else
  DEVICE=$(echo "$DEVICES" | grep -viE 'RAPHAEL|MENDOCINO|GRANITERIDGE|integrated' \
           | head -1 | grep -oE '^\s+[A-Za-z]+[0-9]+' | tr -d ' :' || true)
fi

if [ -z "${DEVICE:-}" ]; then
  echo "❌ Solo se ven GPUs integradas. Dispositivos detectados:"
  echo "$DEVICES"
  echo "   Fuerza uno con: LLAMA_DEVICE=Vulkan1 ./start-local-llm.sh"
  exit 1
fi

echo "🎮 GPU seleccionada: $(echo "$DEVICES" | grep -E "^\s+$DEVICE:" | sed 's/^ *//')"
echo "🔧 Build:            $LLAMA_DIR"

# --- 4. Avisar de la descarga inicial del modelo ----------------------------
# -hf descarga el GGUF a ~/.cache/llama.cpp la primera vez (~4,7 GB). En un
# equipo nuevo esto parece un cuelgue si no se avisa.
if ! find "$LLAMA_CACHE" -name '*.gguf' -size +1G 2>/dev/null | grep -q .; then
  echo "⬇️  Primera ejecución en este equipo: se descargará $MODEL (~4,7 GB)."
  echo "    Destino: $LLAMA_CACHE — puede tardar varios minutos."
fi

# --- 5. Arrancar ------------------------------------------------------------
# -ngl 99 -> todas las capas a la GPU.  -c $CTX -> ventana de contexto.
#
# --skip-chat-parsing: clave para la estabilidad. Por defecto llama-server usa la
# plantilla Jinja de Qwen2.5 con su parser de tool-calls "peg-native", que
# devuelve HTTP 500 ("Failed to parse tool call arguments as JSON") cuando el
# modelo emite JSON o respuestas largas cortadas por max_tokens. Con este flag se
# fuerza un parser de contenido puro: la respuesta llega tal cual en
# message.content (no necesitamos tool-calls), sin ese fallo intermitente.
exec "$LLAMA_DIR/llama-server" \
  -hf "$MODEL" \
  --device "$DEVICE" \
  --host "$HOST" --port "$PORT" \
  -ngl 99 -c "$CTX" --jinja --skip-chat-parsing
