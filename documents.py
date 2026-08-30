"""
Generación de documentos administrativos (LaTeX → PDF).

Define un catálogo de tipos de documento (hoja de queja, solicitud genérica,
recurso de alzada…), rellena una plantilla LaTeX controlada con los datos del
interesado y, opcionalmente, con una redacción jurídica generada por Claude que
cita la normativa BOE relevante. El .tex resultante se compila con `pdflatex`.

El diseño separa "plantilla controlada" de "contenido generado": la estructura
LaTeX la fijamos nosotros (para que compile siempre) y la IA sólo aporta texto
que insertamos escapado. Así una respuesta inesperada del modelo no rompe la
compilación.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path
from typing import List, Optional

# Carpeta donde se guardan los documentos generados (.tex y .pdf).
DOCS_DIR = Path(tempfile.gettempdir()) / "legalize_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _clean_old_docs(max_age_seconds: int = 86400) -> None:
    """Elimina carpetas de documentos generados más antiguas que max_age_seconds (24h por defecto)."""
    now = time.time()
    if not DOCS_DIR.exists():
        return
    for item in DOCS_DIR.iterdir():
        if item.is_dir():
            try:
                if now - item.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(item, ignore_errors=True)
            except OSError:
                pass


# XeLaTeX es el compilador recomendado: maneja UTF-8 y fuentes del sistema de
# forma nativa y funciona con polyglossia. Si no estuviera, caemos a pdflatex.
XELATEX = shutil.which("xelatex")
PDFLATEX = shutil.which("pdflatex")
LATEX_ENGINE = XELATEX or PDFLATEX

# ---------------------------------------------------------------------------
# Catálogo de tipos de documento
# ---------------------------------------------------------------------------
# Cada tipo define las etiquetas de sus secciones y los campos del formulario.
# `fields` se usa tanto para construir el formulario en el frontend como para
# validar lo que llega al backend.

COMMON_FIELDS = [
    {"name": "nombre", "label": "Nombre y apellidos", "required": True},
    {"name": "dni", "label": "DNI / NIE", "required": True},
    {"name": "domicilio", "label": "Domicilio a efectos de notificación", "required": False},
    {"name": "email", "label": "Correo electrónico", "required": False},
    {"name": "telefono", "label": "Teléfono", "required": False},
    {"name": "organismo", "label": "Órgano / organismo destinatario", "required": True},
    {"name": "lugar", "label": "Lugar (ciudad)", "required": False},
    {"name": "fecha", "label": "Fecha (por defecto, la de hoy)", "required": False},
    {"name": "asunto", "label": "Asunto", "required": False},
    {"name": "hechos", "label": "Hechos / motivo (descríbelo con tus palabras)", "required": True,
     "multiline": True},
    {"name": "peticion", "label": "Qué solicitas", "required": True, "multiline": True},
]

DOC_TYPES: dict[str, dict] = {
    "solicitud": {
        "label": "Solicitud / Instancia genérica",
        "description": "Escrito dirigido a una Administración para pedir algo "
                       "(autorización, prestación, certificado, etc.).",
        "titulo": "SOLICITUD",
        "verbo_expone": "EXPONE",
        "verbo_solicita": "SOLICITA",
        "fields": COMMON_FIELDS,
    },
    "hoja_queja": {
        "label": "Hoja de queja / reclamación",
        "description": "Reclamación formal por un servicio, actuación o trato "
                       "recibido de una Administración o entidad.",
        "titulo": "HOJA DE QUEJA / RECLAMACIÓN",
        "verbo_expone": "EXPONE LOS SIGUIENTES HECHOS",
        "verbo_solicita": "RECLAMA",
        "fields": COMMON_FIELDS,
    },
    "recurso_alzada": {
        "label": "Recurso de alzada",
        "description": "Recurso administrativo contra una resolución, ante el "
                       "órgano superior jerárquico (art. 121-122 Ley 39/2015).",
        "titulo": "RECURSO DE ALZADA",
        "verbo_expone": "ALEGA",
        "verbo_solicita": "SUPLICA",
        "fields": COMMON_FIELDS + [
            {"name": "acto_recurrido", "label": "Resolución / acto que se recurre",
             "required": True},
            {"name": "fecha_acto", "label": "Fecha de la resolución recurrida",
             "required": False},
            {"name": "organo_autor", "label": "Órgano que dictó la resolución",
             "required": False},
        ],
    },
    "recurso_reposicion": {
        "label": "Recurso potestativo de reposición",
        "description": "Recurso administrativo ante el mismo órgano que dictó el acto que pone fin a la vía administrativa (art. 123-124 Ley 39/2015).",
        "titulo": "RECURSO POTESTATIVO DE REPOSICIÓN",
        "verbo_expone": "ALEGA LOS SIGUIENTES HECHOS",
        "verbo_solicita": "SOLICITA",
        "fields": COMMON_FIELDS + [
            {"name": "acto_recurrido", "label": "Resolución / acto que se recurre", "required": True},
            {"name": "numero_expediente", "label": "N.º de expediente / referencia", "required": False},
            {"name": "fecha_acto", "label": "Fecha de la resolución recurrida", "required": False},
        ],
    },
    "alegaciones": {
        "label": "Escrito de alegaciones administrativas",
        "description": "Alegaciones en trámite de audiencia o procedimiento sancionador / administrativo (art. 53 y 82 Ley 39/2015).",
        "titulo": "ESCRITO DE ALEGACIONES",
        "verbo_expone": "ALEGA",
        "verbo_solicita": "SOLICITA",
        "fields": COMMON_FIELDS + [
            {"name": "numero_expediente", "label": "N.º de expediente sancionador / administrativo", "required": True},
            {"name": "tramite_notificado", "label": "Trámite o acuerdo notificado", "required": False},
        ],
    },
    "denuncia_itss": {
        "label": "Denuncia ante la Inspección de Trabajo (ITSS)",
        "description": "Denuncia ante la Inspección de Trabajo y Seguridad Social por infracciones laborales, prevención de riesgos o contratación (Ley 23/2015).",
        "titulo": "DENUNCIA ANTE LA INSPECCIÓN DE TRABAJO Y SEGURIDAD SOCIAL",
        "verbo_expone": "DENUNCIA LOS SIGUIENTES HECHOS",
        "verbo_solicita": "SOLICITA",
        "fields": COMMON_FIELDS + [
            {"name": "empresa_denunciada", "label": "Nombre / Razón Social de la empresa denunciada", "required": True},
            {"name": "cif_empresa", "label": "CIF / NIF de la empresa", "required": False},
            {"name": "centro_trabajo", "label": "Dirección del centro de trabajo", "required": False},
        ],
    },
    "reclamacion_previa_social": {
        "label": "Reclamación previa a la vía judicial social (Seguridad Social / Despidos)",
        "description": "Reclamación previa obligatoria ante la entidad gestora (INSS, TGSS, SEPE) antes de interponer demanda laboral (art. 71 Ley 36/2011 LRJS).",
        "titulo": "RECLAMACIÓN PREVIA A LA VÍA JUDICIAL SOCIAL",
        "verbo_expone": "EXPONE",
        "verbo_solicita": "SOLICITA",
        "fields": COMMON_FIELDS + [
            {"name": "expediente_inss", "label": "N.º de expediente / resolución de la entidad gestora", "required": True},
            {"name": "entidad_gestora", "label": "Entidad gestora (INSS, TGSS, SEPE, Mutua)", "required": True},
        ],
    },
}


def doc_types_catalog() -> List[dict]:
    """Catálogo serializable para el frontend."""
    return [
        {
            "id": key,
            "label": cfg["label"],
            "description": cfg["description"],
            "fields": cfg["fields"],
        }
        for key, cfg in DOC_TYPES.items()
    ]


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}


def latex_escape(text: str) -> str:
    """Escapa los caracteres especiales de LaTeX en texto de usuario."""
    if not text:
        return ""
    out = []
    for ch in str(text):
        out.append(_LATEX_REPLACEMENTS.get(ch, ch))
    return "".join(out)


def format_text_for_latex(text: str) -> str:
    """Convierte marcas y listas Markdown a comandos de formato XeLaTeX nativos y escapa contenido."""
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").strip()

    # 1. Convertir formato Markdown a marcadores de posición alfanuméricos limpios
    text = re.sub(r"\*\*(.*?)\*\*", lambda m: "LXPHBOLDOPEN" + m.group(1) + "LXPHBOLDCLOSE", text)
    text = re.sub(r"__(.*?)__", lambda m: "LXPHBOLDOPEN" + m.group(1) + "LXPHBOLDCLOSE", text)
    text = re.sub(r"\*(.*?)\*", lambda m: "LXPHITALICOPEN" + m.group(1) + "LXPHITALICCLOSE", text)
    text = re.sub(r"^#{1,3}\s+(.*)$", lambda m: "LXPHH3OPEN" + m.group(1) + "LXPHH3CLOSE", text, flags=re.MULTILINE)

    # 2. Agrupar párrafos o líneas numeradas (1., 2., 1., etc.) en un único entorno \begin{enumerate}
    def replace_enumerate(match):
        block = match.group(0).strip()
        lines = [l.strip() for l in re.split(r"\n+", block) if l.strip()]
        items = []
        for l in lines:
            cleaned_line = re.sub(r"^\s*\d+[\.\)]\s*", "", l)
            items.append("LXPHITEM" + cleaned_line)
        return "LXPHBENUM\n" + "\n".join(items) + "\nLXPHEENUM"

    text = re.sub(r"(?:^\s*\d+[\.\)]\s+.*(?:\n\s*\n?|\n|$))+", lambda m: replace_enumerate(m), text, flags=re.MULTILINE)

    # 3. Agrupar listas con viñetas (- o *) en un único entorno \begin{itemize}
    def replace_itemize(match):
        block = match.group(0).strip()
        lines = [l.strip() for l in re.split(r"\n+", block) if l.strip()]
        items = []
        for l in lines:
            cleaned_line = re.sub(r"^\s*[\-\*]\s*", "", l)
            items.append("LXPHITEM" + cleaned_line)
        return "LXPHBITEM\n" + "\n".join(items) + "\nLXPHEITEM"

    text = re.sub(r"(?:^\s*[\-\*]\s+.*(?:\n\s*\n?|\n|$))+", lambda m: replace_itemize(m), text, flags=re.MULTILINE)

    # 4. Escapar caracteres especiales de LaTeX
    text = latex_escape(text)

    # 5. Sustituir los marcadores por comandos XeLaTeX nativos
    text = text.replace("LXPHBOLDOPEN", r"\textbf{").replace("LXPHBOLDCLOSE", r"}")
    text = text.replace("LXPHITALICOPEN", r"\textit{").replace("LXPHITALICCLOSE", r"}")
    text = text.replace("LXPHH3OPEN", r"\subsection*{").replace("LXPHH3CLOSE", r"}")
    text = text.replace("LXPHBITEM", r"\begin{itemize}").replace("LXPHEITEM", r"\end{itemize}")
    text = text.replace("LXPHBENUM", r"\begin{enumerate}").replace("LXPHEENUM", r"\end{enumerate}")
    text = text.replace("LXPHITEM", r"\item ")

    return text


_MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre")


def _fecha_larga(dia: Optional[date] = None) -> str:
    """Fecha en el formato de los escritos administrativos: "1 de agosto de 2026"."""
    d = dia or date.today()
    return f"{d.day} de {_MESES[d.month - 1]} de {d.year}"


def _paragraphs(text: str) -> str:
    """Convierte texto en bloques de párrafos o entornos de listas XeLaTeX."""
    if not text:
        return ""
    return format_text_for_latex(text.strip())


def _filter_cited_normativa(normativa: Optional[List[dict]], ai_sections: Optional[dict]) -> List[dict]:
    """Conserva únicamente las normas cuyo BOE ID o número oficial figura explícitamente en la redacción de la IA."""
    if not ai_sections or not ai_sections.get("fundamentos"):
        return normativa[:6] if normativa else []

    ai_text = f"{ai_sections.get('fundamentos', '')} {ai_sections.get('exposicion', '')}".lower()
    if not ai_text.strip():
        return []

    cited = []
    seen_ids = set()

    if normativa:
        for doc in normativa:
            ident = (doc.get("identifier") or "").lower().strip()
            official_num = (doc.get("official_number") or "").lower().strip()

            # Coincidencia exacta por identificador BOE (ej. boe-a-2018-16673)
            if ident and ident in ai_text:
                if ident not in seen_ids:
                    seen_ids.add(ident)
                    cited.append(doc)
                continue

            # Coincidencia exacta por número oficial de norma (ej. 3/2018 o 7/2021 o 39/2015)
            if official_num and len(official_num) >= 3 and official_num in ai_text:
                key = official_num
                if key not in seen_ids:
                    seen_ids.add(key)
                    cited.append(doc)
                continue

    # Extraer citas de BOE que figuren en la redacción de la IA si faltaban en la lista de candidatos
    boe_matches = re.findall(r"(BOE-[A-Z]-\d{4}-\d+)", f"{ai_sections.get('fundamentos', '')} {ai_sections.get('exposicion', '')}", re.IGNORECASE)
    for boe_id in boe_matches:
        boe_upper = boe_id.upper()
        if boe_upper.lower() not in seen_ids:
            seen_ids.add(boe_upper.lower())
            cited.append({"identifier": boe_upper, "title": f"Normativa citada en los fundamentos ({boe_upper})"})

    return cited


def _normativa_block(normativa: Optional[List[dict]]) -> str:
    """Lista de normativa real encontrada en el índice, citada con su BOE.

    Se incluye siempre que haya resultados, garantizando que el documento
    referencie legislación existente en el BOE relacionada con el caso.
    """
    if not normativa:
        return ""
    items: List[str] = []
    for doc in normativa[:6]:
        ident = (doc.get("identifier") or "").strip()
        if not ident:
            continue
        title = (doc.get("title") or "Sin título").strip()
        date = (doc.get("publication_date") or "").strip()
        meta = ", ".join(filter(None, [ident, date]))
        ref = f"{title} ({meta})" if meta else title
        items.append("\\item " + latex_escape(ref))
    if not items:
        return ""
    return (
        "\\section*{NORMATIVA APLICABLE}\n"
        "\\noindent A los efectos del presente escrito se considera de aplicación, "
        "entre otra, la siguiente normativa publicada en el BOE:\n"
        "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}\n"
    )


def build_latex(
    doc_type: str,
    datos: dict,
    ai_sections: Optional[dict] = None,
    normativa: Optional[List[dict]] = None,
) -> str:
    """Construye el documento LaTeX completo.

    `ai_sections` (opcional) puede traer claves `exposicion`, `fundamentos` y
    `solicitud` redactadas por la IA. Si no, se usa el texto literal del usuario.
    `normativa` (opcional) es la lista de normas encontradas en el índice, que se
    citan en una sección "Normativa aplicable".
    """
    cfg = DOC_TYPES[doc_type]
    e = latex_escape  # alias corto

    nombre = e(datos.get("nombre", ""))
    dni = e(datos.get("dni", ""))
    domicilio = e(datos.get("domicilio", ""))
    email = e(datos.get("email", ""))
    telefono = e(datos.get("telefono", ""))
    organismo = e(datos.get("organismo", ""))
    lugar = e(datos.get("lugar", "")) or r"\rule{4cm}{0.4pt}"
    fecha = e(datos.get("fecha", "")) or _fecha_larga()
    asunto = e(datos.get("asunto", ""))

    ai = ai_sections or {}
    exposicion = _paragraphs(ai.get("exposicion") or datos.get("hechos", ""))
    solicitud = _paragraphs(ai.get("solicitud") or datos.get("peticion", ""))
    fundamentos = _paragraphs(ai.get("fundamentos") or "")

    # Datos identificativos del interesado
    ident_lines = [f"\\textbf{{{nombre}}}, con DNI/NIE \\textbf{{{dni}}}"]
    if domicilio:
        ident_lines.append(f"y domicilio a efectos de notificaciones en {domicilio}")
    contacto = ", ".join(filter(None, [
        f"correo electrónico {email}" if email else "",
        f"teléfono {telefono}" if telefono else "",
    ]))
    ident = " ".join(ident_lines) + ("" if not contacto else f" ({contacto})")

    # Bloque específico del tipo de trámite
    extra = ""
    if doc_type in ("recurso_alzada", "recurso_reposicion"):
        acto = e(datos.get("acto_recurrido", ""))
        fecha_acto = e(datos.get("fecha_acto", ""))
        organo_autor = e(datos.get("organo_autor", ""))
        num_exp = e(datos.get("numero_expediente", ""))
        det = acto
        if num_exp:
            det += f" (Expediente N.º {num_exp})"
        if fecha_acto:
            det += f", de fecha {fecha_acto}"
        if organo_autor:
            det += f", dictada por {organo_autor}"
        nombre_recurso = "RECURSO DE ALZADA" if doc_type == "recurso_alzada" else "RECURSO POTESTATIVO DE REPOSICIÓN"
        extra = (
            "\\medskip\n\\noindent Que, mediante el presente escrito, interpongo "
            f"\\textbf{{{nombre_recurso}}} contra la resolución siguiente: "
            f"{det}.\n"
        )
    elif doc_type == "alegaciones":
        num_exp = e(datos.get("numero_expediente", ""))
        tramite = e(datos.get("tramite_notificado", ""))
        det = f"en el expediente N.º {num_exp}" if num_exp else ""
        if tramite:
            det += f" relativo al trámite de {tramite}"
        extra = (
            "\\medskip\n\\noindent Que, habiendo sido notificada la apertura del trámite, formulamos las presentes "
            f"\\textbf{{ALEGACIONES}} {det}.\n"
        )
    elif doc_type == "denuncia_itss":
        empresa = e(datos.get("empresa_denunciada", ""))
        cif = e(datos.get("cif_empresa", ""))
        centro = e(datos.get("centro_trabajo", ""))
        det_empresa = empresa
        if cif:
            det_empresa += f" (CIF: {cif})"
        if centro:
            det_empresa += f", con centro de trabajo en {centro}"
        extra = (
            "\\medskip\n\\noindent Que formulo la presente \\textbf{{DENUNCIA ANTE LA INSPECCIÓN DE TRABAJO}} frente a la empresa "
            f"{det_empresa} por presunta infracción en materia laboral y de Seguridad Social.\n"
        )
    elif doc_type == "reclamacion_previa_social":
        exp = e(datos.get("expediente_inss", ""))
        entidad = e(datos.get("entidad_gestora", ""))
        extra = (
            "\\medskip\n\\noindent Que formulo \\textbf{{RECLAMACIÓN PREVIA A LA VÍA JUDICIAL SOCIAL}} "
            f"frente a {entidad} en relación con la resolución del expediente N.º {exp}.\n"
        )

    fundamentos_block = ""
    if fundamentos:
        fundamentos_block = (
            "\\section*{FUNDAMENTOS DE DERECHO}\n" + fundamentos + "\n"
        )

    normativa_block = _normativa_block(_filter_cited_normativa(normativa, ai_sections))

    asunto_block = f"\\noindent\\textbf{{Asunto:}} {asunto}\\par\\medskip\n" if asunto else ""

    # XeLaTeX (recomendado) usa fontspec + polyglossia; UTF-8 y fuentes nativas.
    # Fallback a pdflatex con babel para instalaciones sin XeLaTeX.
    if XELATEX:
        lang_preamble = (
            "\\usepackage{fontspec}\n"
            "\\usepackage{polyglossia}\n"
            "\\setdefaultlanguage{spanish}\n"
        )
    else:
        lang_preamble = (
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage[spanish,provide=*]{babel}\n"
            "\\usepackage{lmodern}\n"
        )

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
{lang_preamble}\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{parskip}}
\setlength{{\parindent}}{{0pt}}

\begin{{document}}

\begin{{center}}
{{\large\bfseries {e(cfg["titulo"])}}}
\end{{center}}

\bigskip

{asunto_block}\noindent\textbf{{DESTINATARIO:}} {organismo}\par
\medskip

\noindent\textbf{{DATOS DEL INTERESADO/A:}}\par
\noindent {ident}.\par
\bigskip

{extra}
\section*{{{e(cfg["verbo_expone"])}}}
{exposicion}

{fundamentos_block}{normativa_block}
\section*{{{e(cfg["verbo_solicita"])}}}
{solicitud}

\bigskip
\noindent En {lugar}, a {fecha}.

\bigskip\bigskip
\noindent Fdo.: {nombre}

\vfill
\noindent\footnotesize\textit{{Documento generado automáticamente por el Consultor
Legislativo. No constituye asesoramiento jurídico profesional; revise los datos y
la normativa citada antes de presentarlo.}}

\end{{document}}
"""
    return tex


def compile_pdf(tex: str, workdir: Path) -> Optional[Path]:
    """Compila el .tex a PDF con XeLaTeX (o pdflatex). Devuelve la ruta o None."""
    engines = [e for e in (XELATEX, PDFLATEX) if e is not None]
    if not engines:
        (workdir / "compile_error.txt").write_text("No hay motor LaTeX (xelatex o pdflatex) instalado en el sistema.", encoding="utf-8")
        return None

    tex_path = workdir / "documento.tex"
    tex_path.write_text(tex, encoding="utf-8")

    pdf_path = workdir / "documento.pdf"
    errors = []
    for engine in engines:
        try:
            res = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory", str(workdir), str(tex_path)],
                capture_output=True,
                timeout=60,
                cwd=str(workdir),
            )
            if pdf_path.exists():
                return pdf_path
            err_msg = res.stderr.decode("utf-8", errors="ignore") or res.stdout.decode("utf-8", errors="ignore")
            errors.append(f"[{engine}] Exit code {res.returncode}:\n{err_msg[:1000]}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            errors.append(f"[{engine}] Excepción: {exc}")
            continue

    if not pdf_path.exists() and errors:
        (workdir / "compile_error.txt").write_text("\n\n".join(errors), encoding="utf-8")

    return pdf_path if pdf_path.exists() else None



# ---------------------------------------------------------------------------
# Redacción asistida por Claude
# ---------------------------------------------------------------------------

async def draft_with_claude(doc_type: str, datos: dict, normativa: List[dict],
                            precision: bool = False,
                            adjunto_texto: str = "",
                            adjunto_nombre: str = "",
                            contexto_normativa: str = "") -> Optional[dict]:
    """Pide al modelo una redacción formal estructurada (JSON).

    Con `precision=True` se antepone Anthropic, recomendable en escritos que se
    presentan ante una Administración: un fundamento de derecho mal citado
    puede costar la inadmisión. Si su cuota se agota, se sigue con el local.

    Devuelve un dict con claves `exposicion`, `fundamentos`, `solicitud`, o None
    si no hay backend de IA o falla la llamada.
    """
    import asyncio
    import llm

    if not llm.available():
        return None

    cfg = DOC_TYPES[doc_type]

    ctx_parts: List[str] = []
    for i, doc in enumerate(normativa[:6], 1):
        ctx_parts.append(
            f"[{i}] {doc.get('title', 'Sin título')} "
            f"(BOE {doc.get('identifier', '')}, {doc.get('publication_date', '')})"
        )
    # Con solo los títulos, el modelo no puede fundamentar nada: deja vacíos
    # los fundamentos aunque la norma correcta esté en la lista, porque no ve
    # su articulado. Si el llamante aporta pasajes literales, se usan esos.
    contexto = (contexto_normativa
                or "\n".join(ctx_parts)
                or "No se han encontrado normas específicas.")

    system_prompt = (
        "Eres un jurista que redacta escritos administrativos en español, en "
        "estilo formal y respetuoso. Redactas en primera persona del interesado. "
        "Tu objetivo es dar el MÁXIMO peso jurídico a la solicitud, apoyándola en "
        "la normativa vigente de mayor jerarquía de las proporcionadas (prioriza "
        "Leyes Orgánicas como la LOPDGDD 3/2018 o Leyes de Procedimiento como la Ley 39/2015). "
        "No utilices ni fundamentes en instrucciones antiguas de los años 90 ni en reglamentos "
        "de sectores ajenos (como casinos, bingos, o telecomunicaciones específicas) salvo "
        "que el caso trate explícitamente de esos sectores. "
        "IMPORTANTE: PROHIBIDO utilizar marcado o símbolos de Markdown (como **, __, #, -, *, ``). "
        "Escribe exclusivamente en castellano formal y limpio. Si necesitas estructurar o listar puntos, "
        "utiliza exclusivamente párrafos o listas numeradas normales (1., 2., 3.). "
        "Cita ÚNICAMENTE identificadores BOE y artículos que aparezcan en la normativa proporcionada. "
        "Devuelve EXCLUSIVAMENTE un objeto JSON válido con las claves: exposicion, fundamentos, solicitud. "
        "Sin texto adicional ni envoltorios."
    )

    # Pasajes del documento que aporta el ciudadano (una resolución que
    # recurre, un contrato, un requerimiento). Los selecciona el llamante,
    # que es quien tiene el buscador; aquí solo se insertan en el prompt.
    bloque_adjunto = ""
    if adjunto_texto:
        bloque_adjunto = (
            f"DOCUMENTO APORTADO POR EL CIUDADANO "
            f"({adjunto_nombre or 'documento'}):\n{adjunto_texto}\n\n"
            "Extrae de él los hechos, fechas, importes, números de expediente y "
            "el órgano actuante, y úsalos literalmente en la exposición en vez "
            "de generalidades. No inventes datos que no figuren en él.\n\n"
        )

    user_message = (
        f"Tipo de documento: {cfg['label']}\n"
        f"Datos aportados por el ciudadano:\n{json.dumps(datos, ensure_ascii=False, indent=2)}\n\n"
        f"{bloque_adjunto}"
        f"Normativa aplicable encontrada en la base de datos (cita SÓLO de aquí):\n{contexto}\n\n"
        "Redacta:\n"
        "- exposicion: los hechos/motivos de forma ordenada y formal.\n"
        "- fundamentos: OBLIGATORIO, nunca vacío. Escribe uno o varios párrafos "
        "numerados. En cada uno: enuncia la regla invocada, indica de qué norma "
        "sale citando su identificador BOE, y —si el pasaje aportado trae "
        "encabezado de artículo— cita ese artículo tal como aparece. Después "
        "relaciona la regla con los hechos del caso. Trabaja con las normas más "
        "cercanas al asunto de entre las aportadas; si ninguna encaja del todo, "
        "usa la más próxima y acótala explícitamente en lugar de callar. No "
        "reproduzcas la lista de normas: argumenta con ellas.\n"
        "- solicitud: lo que se pide, de forma concreta y firme.\n"
        "Responde sólo con el JSON."
    )

    # Con los fundamentos ya obligatorios el JSON es más largo; 1500 tokens
    # lo cortaban a media respuesta y el bloque quedaba sin cerrar.
    raw = await asyncio.to_thread(llm.complete, system_prompt, user_message, max_tokens=2500, precision=precision)
    if not raw:
        return None

    # Extraer el primer bloque JSON aunque venga con texto alrededor.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    def _to_text(val) -> str:
        if isinstance(val, list):
            return "\n\n".join(str(x) for x in val if str(x).strip())
        return str(val or "")

    return {
        "exposicion": _to_text(data.get("exposicion")),
        "fundamentos": _to_text(data.get("fundamentos")),
        "solicitud": _to_text(data.get("solicitud")),
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def generate(
    doc_type: str,
    datos: dict,
    ai_sections: Optional[dict],
    normativa: Optional[List[dict]] = None,
) -> dict:
    """Construye el .tex, lo compila y lo guarda. Devuelve metadatos + token."""
    _clean_old_docs()
    doc_id = uuid.uuid4().hex[:16]

    workdir = DOCS_DIR / doc_id
    workdir.mkdir(parents=True, exist_ok=True)

    tex = build_latex(doc_type, datos, ai_sections, normativa)
    (workdir / "documento.tex").write_text(tex, encoding="utf-8")

    pdf_path = compile_pdf(tex, workdir)

    citada = [
        {"identifier": d.get("identifier", ""), "title": d.get("title", "")}
        for d in (normativa or [])[:6]
        if d.get("identifier")
    ]

    err_file = workdir / "compile_error.txt"
    compile_err = err_file.read_text(encoding="utf-8") if err_file.exists() else None

    return {
        "doc_id": doc_id,
        "latex": tex,
        "pdf_available": pdf_path is not None,
        "compile_error": compile_err,
        "used_ai": ai_sections is not None,
        "normativa": citada,
    }


def file_path(doc_id: str, fmt: str) -> Optional[Path]:
    """Ruta de un fichero generado (validando el doc_id contra path traversal)."""
    if not re.fullmatch(r"[0-9a-f]{16}", doc_id):
        return None
    name = "documento.tex" if fmt == "tex" else "documento.pdf"
    path = DOCS_DIR / doc_id / name
    return path if path.exists() else None
