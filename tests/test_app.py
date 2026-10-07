from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from test_summarization import pdf_bytes


def test_sample_summary_persists_without_loading_ai():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Summascribe.py")).run()
    assert not app.exception
    app.button[0].click().run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["summary"]["method"] == "Extractive"
    assert app.session_state["summary"]["evidence"]
    app.slider[0].set_value(4).run()
    assert not app.exception
    assert app.session_state["summary"]["summary"]


def test_blank_document_cannot_be_submitted():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Summascribe.py")).run()
    assert app.button[1].disabled
    assert not app.exception


def test_pdf_upload_summarizes_and_detects_source_change():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Summascribe.py")).run()
    if not hasattr(app.file_uploader[0], "set_value"):
        pytest.skip("Upload testing requires a recent Streamlit release")
    app.file_uploader[0].set_value(("garden.pdf", pdf_bytes(), "application/pdf")).run()
    app.button[1].click().run()
    assert not app.exception
    assert "garden" in app.session_state["summary"]["summary"]
    app.file_uploader[0].set_value(("note.txt", b"A different document about cloud computing.", "text/plain")).run()
    assert not app.exception
    assert any("source has changed" in message.value for message in app.info)


def test_invalid_pdf_displays_error_without_crashing():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Summascribe.py")).run()
    if not hasattr(app.file_uploader[0], "set_value"):
        pytest.skip("Upload testing requires a recent Streamlit release")
    app.file_uploader[0].set_value(("bad.pdf", b"not a PDF", "application/pdf")).run()
    assert app.error
    assert app.button[1].disabled
    assert not app.exception
