# SummaScribe

A document studio for turning PDFs, text files, and pasted notes into readable summaries. The default extractive mode works immediately, with no model download or API key. Optional local AI uses the original LaMini model and automatically selects CUDA when available.

## What's inside

- In-memory PDF extraction, encrypted-PDF password support, UTF-8 text uploads, and pasted notes.
- Page-range selection and original text inspection alongside the digest.
- Extractive summaries built with TF-IDF relevance and redundancy reduction. Selected sentences retain their page references and source order.
- Optional lazy-loaded, cached local T5 inference. Long inputs are split by token count and hierarchically summarized instead of silently truncated.
- A GPU lock for shared inference, progress updates, useful validation messages, and session-persistent results.
- Markdown and JSON exports, including evidence or intermediate AI section summaries.

## Run locally with starGPU

From this directory in PowerShell:

```powershell
conda activate starGPU
python -m pip install -r requirements.txt
python -m streamlit run Summascribe.py --server.port 8502
```

Alternatively, `./run.ps1` uses `conda run -n starGPU`. Open http://localhost:8502. Extractive mode needs neither PyTorch nor model weights. The theme is configured in `.streamlit/config.toml`.

## Enable local AI

Keep your existing CUDA-enabled PyTorch in starGPU. Install the optional requirements and download weights once:

```powershell
conda activate starGPU
python -m pip install -r requirements-ai.txt
python download_model.py
```

The downloader writes the model to `Lamini-1/` and its download cache to `models/`, both ignored by Git. If this checkout already contains `Lamini-1/`, you can skip the download. Use **Local AI** in the sidebar and select Auto, CPU, or CUDA.

To use another compatible local sequence-to-sequence model:

```powershell
$env:SUMMASCRIBE_MODEL = 'C:\path\to\local-model'
python -m streamlit run Summascribe.py --server.port 8502
```

Model resolution uses `SUMMASCRIBE_MODEL`, then this repository's `Lamini-1/`, then the existing Hugging Face cache for `MBZUAI/LaMini-Flan-T5-248M`. Runtime loading uses `local_files_only=True`; a document request never downloads weights. `.env.example` documents the variable; the app does not automatically read `.env` files.

The original [LaMini-Flan-T5-248M model](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) uses CC BY-NC 4.0. Check its model card when choosing weights for a commercial deployment. The app itself does not impose a model license on extractive mode.

## Limits and interpretation

Uploads are limited to 20 MB and 500 PDF pages; extracted/pasted text to 300,000 characters. Very complex PDF content streams are rejected before extraction. Scanned PDFs need OCR; this app does not perform OCR. A document exceeding the extracted-text limit must be split before upload.

Extractive mode selects original text rather than rewriting it, with up to 30 selected segments in the engine (20 in the interface). Long unpunctuated passages are split into bounded segments. References correspond to PDF page numbers; pasted text has page 1. Duplicate sentences are removed.

AI inputs stay within the tokenizer's input limit, capped at 512 tokens including the instruction and special tokens. Every input chunk is summarized; intermediate results are repeatedly compressed to fit the final pass. The output slider is a maximum token budget, not a guaranteed length. AI rewrites can be inaccurate and do not have sentence-level source citations. If a model fails to compress, the app reports it instead of discarding the remaining input.

Document text remains in memory and the browser's app session; it is not written to `data/` or logged. Processing happens on the app host: when you run locally, that is your machine; a hosted deployment processes documents on its server. Exports include document text and should be handled as you would the original document. Results stay associated with the last submitted document and settings.

## Files and checks

`Summascribe.py` is the Streamlit interface; `summarization.py` is the extraction/summarization engine; `download_model.py` handles explicit model setup. The old LangChain dependency and overlapping character-chunk concatenation are removed. Legacy visual assets and the root `config.toml` remain unused by the new interface.

```powershell
python -m pip install pytest
python -m pytest tests -q
python smoke_ai.py
```

Tests cover PDF extraction/passwords, malformed sources, evidence integrity, token coverage, compression failure, and app interaction. `smoke_ai.py` separately validates real local-model inference; it needs the optional dependencies and weights.
