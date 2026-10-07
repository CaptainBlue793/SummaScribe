"""In-memory document extraction, evidence summaries, and token-safe local T5 inference."""
from io import BytesIO
from pathlib import Path
import os
import re
import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_CHARACTERS = 300_000
MAX_FILE_BYTES = 20 * 1024 * 1024


def validate_pages(pages):
    if not pages or not any(page["text"].strip() for page in pages):
        raise ValueError("No readable text found. Scanned PDFs need OCR before upload.")
    if sum(len(page["text"]) for page in pages) > MAX_CHARACTERS:
        raise ValueError("Document is too large. Select fewer pages or use at most 300,000 characters.")
    return pages


def read_pdf(data, password=""):
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PdfReadError
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Choose a PDF smaller than 20 MB.")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(password):
            raise ValueError("This PDF is locked. Enter its password in the sidebar.")
        if len(reader.pages) > 500:
            raise ValueError("Use a PDF with at most 500 pages.")
        pages = []
        total = 0
        for number, page in enumerate(reader.pages, 1):
            # Bound compressed content streams before asking pypdf to extract text.
            contents = page.get_contents()
            if contents and len(contents.get_data()) > 10 * 1024 * 1024:
                raise ValueError("A PDF page is too complex to extract. Export a simpler PDF or upload text.")
            text = page.extract_text() or ""
            total += len(text)
            if total > MAX_CHARACTERS:
                raise ValueError("PDF text exceeds 300,000 characters. Split the PDF before uploading.")
            pages.append({"page": number, "text": text.strip()})
        return validate_pages(pages)
    except DependencyError as exc:
        raise ValueError("This PDF requires an encryption dependency. Install requirements.txt in starGPU and retry.") from exc
    except (PdfReadError, OSError, KeyError, TypeError, NotImplementedError) as exc:
        raise ValueError("Could not read this PDF. Try exporting it again or uploading plain text.") from exc


def read_text(data):
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Choose a text file smaller than 20 MB.")
    try:
        return validate_pages([{"page": 1, "text": data.decode("utf-8-sig").strip()}])
    except UnicodeDecodeError as exc:
        raise ValueError("Save your text as UTF-8 before uploading.") from exc


def sentence_records(pages):
    records = []
    seen = set()
    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            # Long unpunctuated sections are bounded so one line cannot become an entire summary.
            words = sentence.split()
            for start in range(0, len(words), 90):
                segment = " ".join(words[start:start + 90])
                if segment and segment.casefold() not in seen:
                    records.append({"text": segment, "page": page["page"], "position": len(records)})
                    seen.add(segment.casefold())
    return records


def extractive_summary(pages, sentence_count=8):
    validate_pages(pages)
    if not 1 <= sentence_count <= 30:
        raise ValueError("Choose 1–30 summary sentences.")
    records = sentence_records(pages)
    if not records:
        raise ValueError("No readable sentences found.")
    try:
        matrix = TfidfVectorizer(stop_words="english", max_features=8000, sublinear_tf=True).fit_transform([row["text"] for row in records])
        centroid = np.asarray(matrix.mean(axis=0)).ravel()
        relevance = np.asarray(matrix @ centroid).ravel()
        relevance /= max(float(relevance.max()), 1e-12)
        # Maximal marginal relevance balances centrality with non-redundant evidence.
        selected = []
        redundancy = np.zeros(len(records))
        for _ in range(min(sentence_count, len(records))):
            scores = 0.75 * relevance - 0.25 * redundancy
            scores[selected] = -np.inf
            index = int(np.argmax(scores))
            selected.append(index)
            redundancy = np.maximum(redundancy, (matrix @ matrix[index].T).toarray().ravel())
    except ValueError:
        selected = list(range(min(sentence_count, len(records))))
    evidence = [records[index] for index in sorted(selected)]
    return {"summary": "\n\n".join(row["text"] for row in evidence), "evidence": evidence, "method": "Extractive", "pages": len(pages), "source_words": sum(len(p["text"].split()) for p in pages)}


def model_source():
    configured = os.environ.get("SUMMASCRIBE_MODEL", "").strip()
    local = Path(__file__).with_name("Lamini-1")
    return configured or (str(local) if local.is_dir() else "MBZUAI/LaMini-Flan-T5-248M")


def special_token_template(tokenizer):
    """Infer the tokenizer's envelope using encode, shared by Transformers 4 and 5.

    Content token IDs stay intact, including at chunk boundaries. Decoding and
    re-tokenizing chunks can change their length and lose that guarantee.
    """
    raw = tokenizer.encode("A short tokenizer probe.", add_special_tokens=False)
    wrapped = tokenizer.encode("A short tokenizer probe.", add_special_tokens=True)
    if not raw:
        raise ValueError("Tokenizer could not encode the input probe.")
    for start in range(len(wrapped) - len(raw) + 1):
        if wrapped[start:start + len(raw)] == raw:
            return wrapped[:start], wrapped[start + len(raw):]
    raise ValueError("This tokenizer changes content when adding special tokens. Use a compatible T5 tokenizer.")


class LocalSummarizer:
    """Load cached model weights only; serialize calls to the shared GPU model."""
    def __init__(self, source, device="Auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.source = source
        self.device = "cuda" if device == "Auto" and torch.cuda.is_available() else device.lower()
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this environment. Choose Auto or CPU.")
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, local_files_only=True).to(self.device).eval()
        configured_limit = getattr(self.tokenizer, "model_max_length", 512)
        self.input_limit = min(512, configured_limit) if 16 <= configured_limit < 100_000 else 512
        self.prefix = self.tokenizer.encode("Summarize the following text: ", add_special_tokens=False)
        self.special_prefix, self.special_suffix = special_token_template(self.tokenizer)
        self.budget = self.input_limit - len(self.prefix) - len(self.special_prefix) - len(self.special_suffix)
        if self.budget < 8:
            raise ValueError("The selected model has too little input space for summarization.")
        self.lock = threading.Lock()

    def _generate(self, token_ids, max_tokens):
        torch = self.torch
        ids = self.special_prefix + self.prefix + token_ids + self.special_suffix
        if len(ids) > self.input_limit:
            raise ValueError("Summary input exceeds the model's token window.")
        inputs = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_tokens, num_beams=2, do_sample=False, no_repeat_ngram_size=3)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def summarize(self, pages, max_tokens=180, progress=None):
        validate_pages(pages)
        if not 40 <= max_tokens <= 400:
            raise ValueError("Choose 40–400 output tokens.")
        text = "\n\n".join(page["text"] for page in pages)
        # Full-source encoding only measures/splits tokens; no overlong sequence is sent to the model.
        tokens = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        sections = []
        with self.lock:
            chunks = [tokens[i:i + self.budget] for i in range(0, len(tokens), self.budget)]
            for index, chunk in enumerate(chunks):
                sections.append(self._generate(chunk, max_tokens if len(chunks) == 1 else min(max_tokens, 128)))
                if progress:
                    progress((index + 1) / len(chunks), f"Reading section {index + 1} of {len(chunks)}")
            # Hierarchical compression processes every section; no silent tokenizer truncation.
            combined = self.tokenizer.encode("\n\n".join(sections), add_special_tokens=False, verbose=False)
            for _ in range(12):
                if len(combined) <= self.budget:
                    break
                reduced = [self._generate(combined[i:i + self.budget], max(8, min(96, self.budget // 4))) for i in range(0, len(combined), self.budget)]
                next_tokens = self.tokenizer.encode("\n\n".join(reduced), add_special_tokens=False, verbose=False)
                if len(next_tokens) >= len(combined):
                    raise ValueError("The model could not compress this document. Use extractive mode or summarize fewer pages.")
                combined = next_tokens
            if len(combined) > self.budget:
                raise ValueError("Too many sections to combine. Summarize fewer pages.")
            summary = self._generate(combined, max_tokens) if len(chunks) > 1 else sections[0]
        if not summary:
            raise ValueError("The model returned an empty summary. Try extractive mode.")
        return {"summary": summary, "sections": sections, "method": "Local AI", "device": self.device, "model": self.source, "evidence": [], "pages": len(pages), "source_words": sum(len(p["text"].split()) for p in pages)}
