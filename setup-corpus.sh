#!/usr/bin/env bash
# Descarga (o actualiza) el corpus legalize-es que indexa la aplicación.
#
# El corpus vive en un repositorio aparte, mantenido por el proyecto Legalize:
# esta aplicación solo lo consume. Se clona en profundidad 1 porque no usamos
# el histórico: son ~12.000 normas y cada reforma es un commit, así que el
# histórico completo multiplica varias veces el tamaño de descarga.
set -euo pipefail

cd "$(dirname "$0")"

CORPUS_URL="${CORPUS_URL:-https://github.com/legalize-dev/legalize-es.git}"
CORPUS_DIR="${CORPUS_PATH:-$PWD/legalize-es}"

if [ -d "$CORPUS_DIR/.git" ]; then
  echo "Actualizando corpus en $CORPUS_DIR..."
  git -C "$CORPUS_DIR" pull --ff-only --depth 1 origin HEAD
else
  if [ -e "$CORPUS_DIR" ] && [ -n "$(ls -A "$CORPUS_DIR" 2>/dev/null)" ]; then
    echo "❌ $CORPUS_DIR existe y no está vacío, pero no es un clon de git."
    echo "   Si ya tienes ahí una copia del corpus, no hace falta este script:"
    echo "   arranca con CORPUS_PATH=\"$CORPUS_DIR\" ./start.sh"
    exit 1
  fi
  echo "Clonando corpus (~1 GB) en $CORPUS_DIR..."
  git clone --depth 1 "$CORPUS_URL" "$CORPUS_DIR"
fi

# Verificación mínima: que estén los directorios de jurisdicción que indexa la app.
n=$(find "$CORPUS_DIR" -maxdepth 1 -type d -name 'es*' | wc -l)
if [ "$n" -eq 0 ]; then
  echo "❌ El clon no contiene directorios de jurisdicción (es/, es-pv/, es-ct/…)."
  exit 1
fi
normas=$(find "$CORPUS_DIR" -maxdepth 2 -name '*.md' | wc -l)
echo "✅ Corpus listo: $n jurisdicciones, $normas normas."
echo "   La primera indexación tarda ~1 minuto; después se lee de .cache/."
