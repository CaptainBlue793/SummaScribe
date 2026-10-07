from io import BytesIO
import threading

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from summarization import LocalSummarizer, extractive_summary, read_pdf, read_text, special_token_template


def pdf_bytes(text="The garden provides fresh food to the community.", password=None, algorithm="RC4-128"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password, algorithm=algorithm)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_text_and_passwords():
    assert "garden" in read_pdf(pdf_bytes())[0]["text"]
    data = pdf_bytes(password="secret")
    with pytest.raises(ValueError, match="password"):
        read_pdf(data, "wrong")
    assert read_pdf(data, "secret")[0]["page"] == 1


def test_aes_encrypted_pdf():
    data = pdf_bytes(password="secret", algorithm="AES-256")
    assert "garden" in read_pdf(data, "secret")[0]["text"]


def test_blank_and_malformed_pdf_are_rejected():
    with pytest.raises(ValueError):
        read_pdf(b"not a PDF")
    with pytest.raises(ValueError, match="No readable"):
        read_pdf(pdf_bytes(""))


def test_text_validation():
    assert read_text(b"\xef\xbb\xbfHello world.")[0]["text"] == "Hello world."
    with pytest.raises(ValueError, match="UTF-8"):
        read_text(b"\xff")
    with pytest.raises(ValueError):
        read_text(b" ")


def test_evidence_is_original_deduplicated_and_ordered():
    pages = [{"page": 1, "text": "Gardens provide fresh food. Gardens provide fresh food. Rainwater supports irrigation."}, {"page": 2, "text": "Community volunteers care for gardens. Soil testing improves safety."}]
    result = extractive_summary(pages, 3)
    assert len(result["evidence"]) == 3
    assert len({row["text"] for row in result["evidence"]}) == 3
    assert [row["position"] for row in result["evidence"]] == sorted(row["position"] for row in result["evidence"])
    for row in result["evidence"]:
        assert row["text"] in pages[row["page"] - 1]["text"]


def test_unpunctuated_text_is_bounded():
    result = extractive_summary([{"page": 1, "text": "garden " * 1000}], 3)
    assert len(result["summary"].split()) <= 270


class Tokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return [int(word) for word in text.split()]


def fake_model(budget=20):
    model = LocalSummarizer.__new__(LocalSummarizer)
    model.tokenizer = Tokenizer()
    model.budget = budget
    model.lock = threading.Lock()
    model.source = "test-model"
    model.device = "cpu"
    model.calls = []

    def generate(tokens, limit):
        assert len(tokens) <= budget
        model.calls.append((tokens, limit))
        return " ".join(str(token) for token in tokens[:2])

    model._generate = generate
    return model


def test_local_model_reads_every_token_before_combining():
    model = fake_model()
    tokens = list(range(200))
    result = model.summarize([{"page": 1, "text": " ".join(map(str, tokens))}], 80)
    assert [token for chunk, _ in model.calls[:10] for token in chunk] == tokens
    assert len(result["sections"]) == 10
    assert result["summary"]


def test_single_chunk_respects_requested_output_budget():
    model = fake_model()
    model.summarize([{"page": 1, "text": "1 2 3"}], 300)
    assert model.calls[0][1] == 300


def test_noncontracting_model_fails_instead_of_truncating():
    model = fake_model()
    model._generate = lambda tokens, limit: " ".join(map(str, tokens))
    with pytest.raises(ValueError, match="could not compress"):
        model.summarize([{"page": 1, "text": " ".join(map(str, range(100)))}])


def test_special_token_envelope_preserves_content():
    class Wrapper:
        def encode(self, text, add_special_tokens=False):
            return [0, 7, 8, 1] if add_special_tokens else [7, 8]
    assert special_token_template(Wrapper()) == ([0], [1])
