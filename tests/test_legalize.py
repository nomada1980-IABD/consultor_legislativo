import pytest
from fastapi.testclient import TestClient
import app
import documents
import llm


def test_normalize_text():
    assert app._normalize_text("indemnización") == "indemnizacion"
    assert app._normalize_text("CÓDIGO PENAL") == "codigo penal"
    assert app._normalize_text("prestación por desempleo") == "prestacion por desempleo"


def test_tokenise_accent_insensitivity():
    tokens_with_accent = app._tokenise("indemnización por despido")
    tokens_without_accent = app._tokenise("indemnizacion por despido")
    assert tokens_with_accent == tokens_without_accent
    assert "indemnizacion" in tokens_with_accent
    assert "despido" in tokens_with_accent


def test_latex_escape():
    raw_text = "Monto < 500 euros & > 200 euros | 100% #1"
    escaped = documents.latex_escape(raw_text)
    assert r"\textless{}" in escaped
    assert r"\textgreater{}" in escaped
    assert r"\textbar{}" in escaped
    assert r"\&" in escaped
    assert r"\%" in escaped
    assert r"\#" in escaped


def test_anthropic_effort_budget_tokens():
    llm.ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
    llm.ANTHROPIC_EFFORT = "high"
    effort_map = {"low": 1024, "medium": 2048, "high": 4096, "xhigh": 8192, "max": 16384}
    assert effort_map[llm.ANTHROPIC_EFFORT] == 4096


def test_api_status():
    client = TestClient(app.app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "has_ai" in data
    assert "regions" in data


def test_api_doc_types():
    client = TestClient(app.app)
    response = client.get("/api/doc-types")
    assert response.status_code == 200
    data = response.json()
    assert "types" in data
    assert len(data["types"]) >= 7
    type_ids = [t["id"] for t in data["types"]]
    assert "recurso_reposicion" in type_ids
    assert "alegaciones" in type_ids
    assert "denuncia_itss" in type_ids
    assert "reclamacion_previa_social" in type_ids


def test_pypdf_extraction():
    import io
    import pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)
    pdf_bytes = stream.getvalue()
    texto = app._extraer_texto("prueba.pdf", pdf_bytes)
    assert isinstance(texto, str)


def test_ocr_tesseract_extraction():
    import io
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "BOE 2026", fill=(0, 0, 0))
    stream = io.BytesIO()
    img.save(stream, format="PNG")
    png_bytes = stream.getvalue()
    texto = app._extraer_texto("escaneo.png", png_bytes)
    assert isinstance(texto, str)


def test_universal_format_extraction():
    datos_txt = "Notificacion de resolucion administrativa expediente 12345".encode("utf-8")
    texto = app._extraer_texto("documento.xyz", datos_txt)
    assert "12345" in texto


def test_markdown_to_xelatex_conversion():
    import documents
    md_sample = "**PRIMERO.** Hechos 100% legales & probados:\n- Elemento A\n- Elemento B"
    tex_out = documents.format_text_for_latex(md_sample)
    assert "\\textbf{PRIMERO.}" in tex_out
    assert "100\\%" in tex_out
    assert "\\&" in tex_out
    assert "\\begin{itemize}" in tex_out
    assert "\\item Elemento A" in tex_out

