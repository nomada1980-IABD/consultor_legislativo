# Consultor Legislativo España

Aplicación web que indexa las **~12.000 normas españolas** del dataset [legalize-es](https://github.com/legalize-dev/legalize-es) y permite buscarlas, preguntar sobre ellas en lenguaje natural y generar borradores de escritos administrativos en PDF con la normativa aplicable ya citada.

Funciona **sin ninguna clave de API** (búsqueda por palabra clave), y con IA **local en GPU**, **HuggingFace** o **Anthropic** si los configuras.

> Este repositorio contiene **solo la aplicación**. Los datos legislativos son del proyecto Legalize y se descargan aparte con `./setup-corpus.sh`.

---

## Qué hace

- **Búsqueda** por palabras clave o acrónimos (IRPF, LAU, RGPD, ET, ERTE, DANA…), con filtros por comunidad autónoma, rango normativo y estado de vigencia.
- **Ranking BM25 sobre el articulado completo**, no solo sobre los títulos: "estafa" encuentra el Código Penal aunque esa palabra no esté en su título.
- **Búsqueda semántica** opcional, que reordena resultados con embeddings multilingües.
- **Asistente jurídico** que responde citando identificadores BOE y los pasajes concretos de los artículos que sustentan la respuesta.
- **Documentos adjuntos**: sube un PDF, un DOCX o una imagen (multa, resolución, notificación) y la aplicación extrae su texto, localiza la normativa aplicable y lo usa como contexto.
- **Generación de escritos** (solicitud, hoja de queja, recurso de alzada) en LaTeX → PDF, fechados automáticamente para poder firmarlos electrónicamente sin firma manuscrita.

---

## Instalación

```bash
git clone https://github.com/nomada1980-IABD/consultor_legislativo
cd consultor_legislativo

./setup-corpus.sh              # clona legalize-es (~1 GB, profundidad 1)
pip install -r requirements.txt
python3 app.py                 # → http://localhost:8000
```

La primera indexación tarda **~60-90 s** y se cachea en `.cache/`; los arranques siguientes son inmediatos. Mientras tanto la interfaz muestra el progreso.

### Dónde vive el corpus

La aplicación no necesita estar dentro del repositorio de datos. Busca el corpus, por este orden:

1. La variable de entorno `CORPUS_PATH`.
2. `./legalize-es` — lo que crea `setup-corpus.sh`.

Si ya tienes una copia de `legalize-es` en el disco, no la descargues otra vez. O la apuntas:

```bash
echo 'CORPUS_PATH=/ruta/a/legalize-es' >> .env
```

…o copias sus directorios de jurisdicción a `./legalize-es`. `setup-corpus.sh` reconoce esa copia y no la toca, pero al no ser un clon de git tampoco puede actualizarla: para eso, bórrala y ejecuta el script.

`GET /api/status` devuelve `corpus_path` e `index_error`, que dicen exactamente qué corpus se cargó o por qué no se pudo.

---

## Modos de IA — independiente de proveedor

La aplicación detecta y usa el mejor proveedor disponible:

| Prioridad | Proveedor | Variable | Nota |
|-----------|-----------|----------|------|
| 1 | **IA local** (llama.cpp / GPU) | `LOCAL_LLM_URL` | Gratis, privado, sin latencia de red |
| 2 | **HuggingFace** (nube) | `HF_TOKEN` | Gratuito con cuenta HF; no necesita GPU |
| 3 | **Anthropic Claude** (nube) | `ANTHROPIC_API_KEY` | Máxima precisión normativa |
| — | Sin IA | *(ninguna)* | La búsqueda por palabra clave funciona igual |

### Modo precisión y respaldo automático

Para consultas donde la exactitud importa —citar el artículo correcto, redactar los fundamentos de derecho de un recurso— añade `"precision": true` a la petición. Eso **invierte la cadena**:

| Modo | Orden de proveedores |
|------|----------------------|
| Normal | local → HuggingFace → Anthropic |
| Precisión | **Anthropic** → local → HuggingFace |

Si un proveedor falla por **cuota agotada, límite de peticiones, saldo insuficiente o caída de red**, se pasa al siguiente automáticamente. Si se agota el crédito de Anthropic a mitad de sesión, la app sigue respondiendo con el modelo local en vez de devolver un error.

La respuesta indica siempre **quién contestó de verdad**:

```jsonc
{
  "ai_provider": "local",          // respondió el local, no Anthropic
  "ai_fallback": [{ "provider": "anthropic", "reason": "..." }]
}
```

> ⚠️ **El modelo local puede inventarse citas.** Un 7B cuantizado redacta buen español y encuentra las normas pertinentes, pero llega a fabricar números de artículo e identificadores BOE con total aplomo. Para cualquier escrito que vaya a presentarse ante una Administración, usa el modo precisión y **verifica siempre las citas** contra el texto consolidado del BOE.

---

## Arranque

### Solo HuggingFace (sin GPU)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx     # token gratuito en huggingface.co/settings/tokens
./start-cloud.sh
```

### Con IA local en GPU

```bash
./start.sh
```

Prepara el entorno virtual, levanta el modelo local (**Qwen2.5-7B-Instruct** vía llama.cpp en `:8080`), espera a que responda, arranca la web en `:8000` y detiene el modelo al salir.

La primera ejecución descarga el GGUF (~4,4 GB) a `models/` dentro del propio proyecto —no a `~/.cache`— para no llenar la partición del sistema. `start-local-llm.sh` busca el build más reciente en `~/llamacpp/llama-b*` y elige la GPU **dedicada** por nombre, descartando la iGPU.

Arranque manual, en dos terminales:

```bash
./start-local-llm.sh              # modelo local, API OpenAI-compatible en :8080
.venv/bin/python3 app.py          # web → http://localhost:8000
```

---

## Variables de entorno

Copia `.env.example` a `.env` y rellena solo lo que uses.

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `CORPUS_PATH` | `./legalize-es` | Raíz del corpus legalize-es |
| `CACHE_DIR` | `./.cache` | Dónde se guardan los índices derivados |
| `LOCAL_LLM_URL` | `http://127.0.0.1:8080/v1` | Endpoint OpenAI-compatible del modelo local |
| `LOCAL_LLM_MODEL` | `local` | Nombre de modelo enviado al servidor local |
| `HF_TOKEN` | *(vacío)* | Token HuggingFace — activa generación y embeddings |
| `HF_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Modelo de generación HF |
| `HF_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embeddings para búsqueda semántica |
| `ANTHROPIC_API_KEY` | *(vacío)* | Clave Anthropic — habilita el modo precisión |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Modelo de Claude |
| `ANTHROPIC_EFFORT` | `high` | Profundidad de razonamiento: `low` … `max` |
| `LLAMA_DEVICE` | *(autodetección)* | Dispositivo concreto, p. ej. `Vulkan1` o `ROCm0` |
| `LOCAL_LLM_CTX` | `8192` | Ventana de contexto del modelo local |
| `LLAMA_CACHE` | `./models` | Carpeta de descarga de los GGUF |

---

## Requisitos

- **Python 3.10+** y `requirements.txt`.
- *(Solo IA local)* `llama-server` de llama.cpp con soporte Vulkan o ROCm.
- *(Opcional)* **XeLaTeX** o **pdflatex** para compilar a PDF. Sin ellos, la app entrega el `.tex` descargable.
- *(Opcional)* **tesseract** y **poppler** para OCR de adjuntos escaneados.

---

## API

```bash
# Buscar (los acrónimos se expanden solos)
curl "http://localhost:8000/api/search?q=IRPF%20deduccion%20vivienda&limit=5"

# Búsqueda semántica
curl "http://localhost:8000/api/search?q=permiso+maternidad&semantic=true"

# Preguntar al asistente
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question":"¿Cuántos días de permiso por fallecimiento de un familiar tengo?","semantic":true}'

# Generar un recurso de alzada
curl -X POST http://localhost:8000/api/generar -H "Content-Type: application/json" \
  -d '{"doc_type":"recurso_alzada","use_ai":true,"precision":true,
       "datos":{"nombre":"Ana García","dni":"12345678Z",
                "organismo":"Ayuntamiento de Madrid",
                "hechos":"Me han denegado una licencia de obra sin motivar la resolución",
                "peticion":"Que se anule la resolución y se conceda la licencia",
                "acto_recurrido":"Resolución 2024/123"}}'
# Descarga: GET /api/generar/<doc_id>.tex  |  .pdf
```

Endpoints: `/api/status`, `/api/search`, `/api/ask`, `/api/adjunto`, `/api/recent`, `/api/doc/{id}`, `/api/doc-types`, `/api/generar`, `/api/generar/{id}.{fmt}`.

---

## Personalización

| Qué cambiar | Dónde |
|---|---|
| Tipos de documento y sus campos | `DOC_TYPES` en `documents.py` |
| Acrónimos jurídicos reconocidos | `ABBREVIATIONS` en `app.py` |
| Terminología jurídica de la búsqueda | `LEGAL_TERMS` en `app.py` |
| Comunidades autónomas / rangos | `REGIONS`, `RANK_LABELS` en `app.py` |
| Parámetros de ranking (BM25, RRF) | constantes al inicio de `app.py` |
| Aspecto de la interfaz | `static/index.html` |

---

## Hoja de ruta

El corpus se consume hoy clonando el repositorio de datos. Legalize ofrece además dos vías que esta aplicación podrá usar más adelante:

- **API REST** (`legalize.dev`) — elimina la indexación local y mantiene las normas siempre al día, a cambio de depender de la red.
- **Conector MCP** — expone la búsqueda y lectura de normas directamente al modelo, que decide por sí mismo qué consultar.

---

## Créditos y licencia

Los datos legislativos son del proyecto **[Legalize](https://github.com/legalize-dev/legalize-es)**, obtenidos de la API de datos abiertos del BOE y sujetos a las condiciones de reutilización del BOE (cita obligatoria de la fuente). Esta aplicación solo los consume; no los redistribuye.

El código de esta aplicación se publica bajo licencia **MIT** (ver `LICENSE`).

> **Aviso.** Reproducción automatizada de fuentes oficiales, generada además con modelos de lenguaje. No es un texto legal oficial ni verificado, ni constituye asesoramiento jurídico. Consulta siempre la fuente oficial del BOE antes de presentar cualquier escrito.
