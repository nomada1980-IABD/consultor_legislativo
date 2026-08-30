#!/usr/bin/env bash
# Inicia el Consultor Legislativo España con IA local en GPU.
set -euo pipefail

cd "$(dirname "$0")"

# --- Configuración local ----------------------------------------------------
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# --- Corpus -----------------------------------------------------------------
# La aplicación ya no vive dentro del repositorio de datos, así que hay que
# comprobar que el corpus está donde dice CORPUS_PATH. Avisar aquí es mucho
# más claro que arrancar y encontrarse un índice vacío en la interfaz.
CORPUS_DIR="${CORPUS_PATH:-$PWD/legalize-es}"
if [ ! -d "$CORPUS_DIR" ]; then
  echo "❌ No se encuentra el corpus legalize-es en $CORPUS_DIR"
  echo "   Ejecuta ./setup-corpus.sh (descarga ~1 GB) o define CORPUS_PATH"
  echo "   en .env apuntando a una copia que ya tengas."
  exit 1
fi
export CORPUS_PATH="$CORPUS_DIR"

VENV=".venv"
PY="$VENV/bin/python3"
LLM_URL="${LOCAL_LLM_URL:-http://127.0.0.1:8080/v1}"
LLAMA_LOG="$HOME/llamacpp/server.log"

# --- Entorno virtual --------------------------------------------------------
# Se usa el venv del proyecto, no el python3 del sistema: instalar con pip en
# el sistema falla en distros con PEP 668 y ensucia el equipo. Además el venv
# se rompe si el proyecto se mueve a una máquina con otra versión de Python,
# así que se valida importando de verdad, no solo comprobando que existe.
if [ ! -x "$PY" ] || ! "$PY" -c "import fastapi, uvicorn, yaml" 2>/dev/null; then
  echo "⚙️  Preparando entorno virtual con $(python3 -V)..."
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  echo "   Entorno listo."
fi

# --- Modelo local -----------------------------------------------------------
mkdir -p "$(dirname "$LLAMA_LOG")"
echo "Iniciando/verificando modelo local..."
./start-local-llm.sh > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!

# Al salir, parar también el modelo: antes quedaba huérfano ocupando la GPU.
cleanup() { kill "$LLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Esperar a que responda en vez de un 'sleep 2' a ciegas. La primera vez en un
# equipo nuevo hay que descargar ~4,7 GB de GGUF, así que el margen es amplio.
echo -n "Esperando al modelo (puede tardar si es la primera descarga)"
AI_INFO="sin IA local — la app usará HuggingFace/Anthropic si están configurados"
for _ in $(seq 1 300); do
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo ""
    echo "⚠️  El modelo local no arrancó. Últimas líneas de $LLAMA_LOG:"
    tail -5 "$LLAMA_LOG" 2>/dev/null | sed 's/^/     /'
    echo "    Se continúa sin IA local."
    break
  fi
  if curl -sf -o /dev/null "$LLM_URL/models" 2>/dev/null; then
    echo " ✅"
    AI_INFO="IA LOCAL en GPU ($LLM_URL)"
    break
  fi
  echo -n "."
  sleep 2
done

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   Consultor Legislativo España             ║"
echo "║   http://localhost:8000                    ║"
echo "║                                            ║"
echo "║   • El índice se construye en segundo      ║"
echo "║     plano (primera vez ~60-90 s)           ║"
echo "║   • Ctrl+C para detener                    ║"
echo "╚════════════════════════════════════════════╝"
echo "   IA: $AI_INFO"
echo ""

# Sin 'exec': hay que conservar este shell vivo para que el trap de arriba
# pare el modelo al salir. Con exec el proceso se reemplaza y llama-server
# quedaría huérfano ocupando la VRAM.
"$PY" app.py
