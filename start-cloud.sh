#!/usr/bin/env bash
# Arranca el Consultor Legislativo España usando HuggingFace en la nube,
# sin necesidad de GPU. Para el modo local en GPU usa ./start.sh
#
# El modelo por defecto es Qwen2.5-7B-Instruct porque el default histórico de
# llm.py, mistralai/Mistral-7B-Instruct-v0.2, ya no está servido por el
# marketplace de Inference Providers de HuggingFace.
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

export HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# --- Token de HuggingFace ---------------------------------------------------
# Orden: entorno -> .env del proyecto -> ~/.bashrc. Antes solo se miraba en
# ~/.bashrc, que no viaja con el repo al cambiar de equipo.
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
  eval "$(grep '^export HF_TOKEN=' "$HOME/.bashrc" || true)"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "❌ HF_TOKEN no está definido."
  echo "   Consíguelo gratis en https://huggingface.co/settings/tokens y luego:"
  echo "     export HF_TOKEN=hf_xxxx         # solo esta sesión"
  echo "     echo 'HF_TOKEN=hf_xxxx' > .env  # persistente para el proyecto"
  exit 1
fi
export HF_TOKEN

# --- Entorno virtual --------------------------------------------------------
# Se valida importando: un venv traído de otra máquina con distinta versión de
# Python existe pero no resuelve sus paquetes.
if [ ! -x "$PY" ] || ! "$PY" -c "import fastapi, uvicorn, yaml" 2>/dev/null; then
  echo "⚙️  Preparando entorno virtual con $(python3 -V)..."
  rm -rf "$VENV"
  python3 -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi

nohup "$PY" app.py > server.log 2>&1 &
PID=$!
disown
echo "✅ Arrancado con PID $PID — modelo HF_MODEL=$HF_MODEL"
echo "   http://localhost:8000   (logs: server.log   parar: kill $PID)"
