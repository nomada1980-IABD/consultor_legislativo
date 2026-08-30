"""
Capa de abstracción del proveedor de IA, con respaldo automático.

Hay dos cadenas de prioridad según lo que pida quien llama:

  * Modo normal (`precision=False`) — prioriza gratis y privado:
        LOCAL → HUGGINGFACE → ANTHROPIC
  * Modo precisión (`precision=True`) — prioriza exactitud normativa:
        ANTHROPIC → LOCAL → HUGGINGFACE

En ambos casos, si un proveedor falla por cuota agotada, límite de peticiones,
saldo insuficiente, credenciales inválidas o caída de red, **se pasa solo al
siguiente de la cadena**. Así, cuando se agota el límite de tokens de Anthropic
la aplicación sigue respondiendo con el modelo local en vez de devolver un
error. Solo si fallan todos se informa del problema.

Nota: el parámetro `fallbacks` del servidor de Anthropic no sirve aquí — solo
actúa ante rechazos de contenido, nunca ante límites de cuota, que es
justamente el caso que hay que cubrir. Por eso la cadena se hace en cliente.

Variables de entorno:
    LOCAL_LLM_URL     (por defecto http://127.0.0.1:8080/v1)
    LOCAL_LLM_MODEL   (por defecto "local"; llama.cpp ignora el nombre)
    HF_TOKEN          → activa HuggingFace en la cadena
    HF_MODEL          → modelo HF de generación (por defecto Qwen2.5-7B-Instruct)
    HF_EMBED_MODEL    → modelo de embeddings para búsqueda semántica
    ANTHROPIC_API_KEY → activa Anthropic en la cadena
    ANTHROPIC_MODEL   → modelo de Claude (por defecto claude-opus-5)
    ANTHROPIC_EFFORT  → profundidad de razonamiento: low|medium|high|xhigh|max
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

LOCAL_LLM_URL  = os.environ.get("LOCAL_LLM_URL",    "http://127.0.0.1:8080/v1").rstrip("/")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL",  "local")
HF_TOKEN        = os.environ.get("HF_TOKEN",          "")
HF_GEN_MODEL    = os.environ.get("HF_MODEL",         "Qwen/Qwen2.5-7B-Instruct")
HF_EMBED_MODEL  = os.environ.get("HF_EMBED_MODEL",
                                  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL",   "claude-opus-5")
ANTHROPIC_EFFORT = os.environ.get("ANTHROPIC_EFFORT", "high")


class _Fallback(Exception):
    """Señal interna: este proveedor no puede atender, prueba el siguiente.

    Se usa para condiciones recuperables (cuota agotada, límite de peticiones,
    servicio saturado, red caída), no para errores de programación.
    """


@dataclass
class Result:
    """Resultado de una generación, con trazabilidad de a quién se acabó pidiendo."""
    text:      Optional[str]
    provider:  str = "none"          # el que respondió de verdad
    model:     str = ""
    attempts:  list = field(default_factory=list)  # [(proveedor, motivo_del_fallo)]

    @property
    def fell_back(self) -> bool:
        return bool(self.attempts)


# ---------------------------------------------------------------------------
# Detección de proveedores
# ---------------------------------------------------------------------------

_local_cache: tuple[float, bool] = (0.0, False)
_local_cache_lock = threading.Lock()


def _local_reachable(ttl: float = 5.0) -> bool:
    """Prueba si el servidor llama.cpp responde (sin enviar una petición real).

    El resultado se cachea unos segundos: durante una misma petición HTTP esto
    se consulta varias veces y cada sondeo fallido cuesta hasta 2 s de espera.
    """
    global _local_cache
    now = time.monotonic()
    with _local_cache_lock:
        stamp, value = _local_cache
        if now - stamp < ttl:
            return value
        _local_cache = (now, value)
    try:
        req = urllib.request.Request(
            f"{LOCAL_LLM_URL}/models",
            headers={"Accept": "application/json"},
            method="GET",
        )
        urllib.request.urlopen(req, timeout=2)
        value = True
    except Exception:
        value = False
    with _local_cache_lock:
        _local_cache = (time.monotonic(), value)
    return value


def _enabled(name: str) -> bool:
    """¿Está configurado y accesible este proveedor?"""
    if name == "local":
        return _local_reachable()
    if name == "huggingface":
        return bool(HF_TOKEN)
    if name == "anthropic":
        return bool(ANTHROPIC_KEY)
    return False


def chain(precision: bool = False) -> list[str]:
    """Orden de proveedores a intentar, ya filtrado por disponibilidad."""
    order = (["anthropic", "local", "huggingface"] if precision
             else ["local", "huggingface", "anthropic"])
    return [p for p in order if _enabled(p)]


def model_of(name: str) -> str:
    """Nombre del modelo que usaría un proveedor dado."""
    return {
        "local":       LOCAL_LLM_MODEL,
        "huggingface": HF_GEN_MODEL,
        "anthropic":   ANTHROPIC_MODEL,
    }.get(name, "")


def provider(precision: bool = False) -> str:
    """Proveedor que atendería la próxima llamada: 'local', 'huggingface',
    'anthropic' o 'none'."""
    c = chain(precision)
    return c[0] if c else "none"


def available() -> bool:
    """¿Hay algún backend de IA disponible?"""
    return bool(chain())


def active_model(precision: bool = False) -> str:
    """Nombre del modelo que se usará en la próxima llamada."""
    return model_of(provider(precision))


# ---------------------------------------------------------------------------
# Backend local (llama.cpp OpenAI-compatible)
# ---------------------------------------------------------------------------

def _complete_local(system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({
        "model": LOCAL_LLM_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(3):
        req = urllib.request.Request(
            f"{LOCAL_LLM_URL}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code >= 500 and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise _Fallback(f"HTTP {exc.code} del modelo local") from exc
        except urllib.error.URLError as exc:
            raise _Fallback(f"modelo local inaccesible en {LOCAL_LLM_URL}") from exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise _Fallback(f"respuesta inesperada del modelo local: {exc}") from exc

    raise _Fallback(f"el modelo local falló tras varios intentos: {last_exc}")


# ---------------------------------------------------------------------------
# Backend HuggingFace Inference API
# ---------------------------------------------------------------------------

def _complete_hf(system: str, user: str, max_tokens: int) -> str:
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        resp = client.chat_completion(
            model    = HF_GEN_MODEL,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens  = max_tokens,
            temperature = 0.3,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:
        # Aquí caen cuota mensual agotada, modelo no servido por Inference
        # Providers y errores de red: todos recuperables pasando al siguiente.
        raise _Fallback(f"HuggingFace ({HF_GEN_MODEL}): {exc}") from exc

    if not text.strip():
        raise _Fallback("HuggingFace devolvió una respuesta vacía")
    return text


# ---------------------------------------------------------------------------
# Backend Anthropic Claude
# ---------------------------------------------------------------------------

def _complete_anthropic(system: str, user: str, max_tokens: int) -> str:
    """Genera con Claude, con razonamiento extendido para precisión normativa.

    Se activa el pensamiento adaptativo: en cuestiones jurídicas la diferencia
    entre citar bien o mal un artículo depende de razonar sobre el texto de la
    norma, no de recitarlo. `ANTHROPIC_EFFORT` gradúa esa profundidad.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise _Fallback("el paquete 'anthropic' no está instalado") from exc

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # max_tokens acota razonamiento + respuesta juntos, así que hay que dar
    # margen o la respuesta se corta a media frase. Solo se factura lo generado.
    budget = max(max_tokens, 1500) + 8000

    try:
        kwargs = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": budget,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        # Enable thinking if model supports it (Claude 3.7+)
        if "3-7" in ANTHROPIC_MODEL or "3.7" in ANTHROPIC_MODEL:
            effort_map = {
                "low": 1024,
                "medium": 2048,
                "high": 4096,
                "xhigh": 8192,
                "max": 16384,
            }
            thinking_budget = effort_map.get(str(ANTHROPIC_EFFORT).lower(), 2048)
            budget = max(max_tokens, 1500) + thinking_budget + 1000
            kwargs["max_tokens"] = budget
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        msg = client.messages.create(**kwargs)
    except anthropic.RateLimitError as exc:
        # El caso que pidió el usuario: límite de tokens agotado -> al siguiente.
        raise _Fallback("límite de peticiones/tokens de Anthropic agotado") from exc
    except anthropic.AuthenticationError as exc:
        raise _Fallback("clave de Anthropic inválida o revocada") from exc
    except anthropic.PermissionDeniedError as exc:
        raise _Fallback("la clave de Anthropic no tiene acceso a ese modelo") from exc
    except anthropic.BadRequestError as exc:
        # Incluye saldo insuficiente, que la API devuelve como 400.
        raise _Fallback(f"petición rechazada por Anthropic: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise _Fallback("sin conexión con la API de Anthropic") from exc
    except anthropic.APIStatusError as exc:
        raise _Fallback(f"Anthropic respondió {exc.status_code}") from exc

    # Un rechazo por políticas de seguridad llega como HTTP 200 con
    # stop_reason 'refusal' y contenido vacío: hay que mirarlo antes de leer.
    if msg.stop_reason == "refusal":
        raise _Fallback("Anthropic declinó responder a esta consulta")

    # Con razonamiento activo el primer bloque es de tipo 'thinking', no texto:
    # se recorren los bloques en vez de asumir content[0].
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        raise _Fallback("Anthropic devolvió una respuesta vacía")
    return text


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

_BACKENDS = {
    "local":       _complete_local,
    "huggingface": _complete_hf,
    "anthropic":   _complete_anthropic,
}


def complete_ex(system: str, user: str, max_tokens: int = 1500,
                precision: bool = False) -> Result:
    """Genera recorriendo la cadena de proveedores hasta que uno responda.

    `precision=True` antepone Anthropic (máxima exactitud normativa). Si su
    cuota está agotada se continúa con el modelo local automáticamente, sin
    que la aplicación se quede sin respuesta.
    """
    candidates = chain(precision)
    if not candidates:
        return Result(text=None)

    attempts: list = []
    for name in candidates:
        try:
            text = _BACKENDS[name](system, user, max_tokens)
        except _Fallback as exc:
            attempts.append((name, str(exc)))
            continue
        except Exception as exc:                      # imprevisto: no tumbar la app
            attempts.append((name, f"error inesperado: {exc}"))
            continue
        return Result(text=text, provider=name, model=model_of(name),
                      attempts=attempts)

    # Nadie pudo responder: se informa de todo lo intentado, no de un genérico.
    detalle = "; ".join(f"{p}: {motivo}" for p, motivo in attempts)
    return Result(text=f"⚠️ Ningún proveedor de IA pudo responder ({detalle}).",
                  attempts=attempts)


def complete(system: str, user: str, max_tokens: int = 1500,
             precision: bool = False) -> Optional[str]:
    """Versión simple de `complete_ex`, para quien solo necesita el texto."""
    return complete_ex(system, user, max_tokens, precision).text


# ---------------------------------------------------------------------------
# Embeddings para búsqueda semántica (siempre via HuggingFace)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """Genera embeddings para una lista de textos usando HF Inference API.

    Devuelve None si HF_TOKEN no está configurado o la llamada falla.
    """
    if not HF_TOKEN:
        return None
    try:
        import numpy as np
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)

        all_embs: list[list[float]] = []
        for i in range(0, len(texts), 24):
            batch  = texts[i : i + 24]
            result = client.feature_extraction(batch, model=HF_EMBED_MODEL)
            arr    = np.array(result, dtype=float)
            if arr.ndim == 3:
                arr = arr.mean(axis=1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            all_embs.extend(arr.tolist())
        return all_embs

    except Exception as exc:
        print(f"[llm.embed_texts] error: {exc}")
        return None
