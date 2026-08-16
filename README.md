# SummaScribe ✧

**Upload a PDF, get a summary.** A local LLM reads the whole document and condenses it — no API keys, no data
leaving your machine.

🚀 **Live demo:** [huggingface.co/spaces/Scarletta975/SummaScribe](https://huggingface.co/spaces/Scarletta975/SummaScribe)

---

## How it works

1. **Load** — the uploaded PDF is parsed page by page with LangChain's `PyPDFLoader`.
2. **Chunk** — text is split via `RecursiveCharacterTextSplitter` (200-character chunks, 50-character overlap) so
   page breaks don't cut sentences in half, then reassembled into one clean text body.
3. **Summarize** — a HuggingFace `summarization` pipeline runs
   [**LaMini-Flan-T5-248M**](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M), a 248M-parameter instruction-tuned
   T5. Small enough to run on CPU, good enough to write readable prose.
4. **Display** — the original PDF renders in an embedded viewer on the left, the summary appears on the right.

A sidebar slider sets **summary strength** (50–1000 tokens), so you can dial between a one-paragraph gist and a
detailed digest.

## Repo layout

| File | What it is |
|------|-----------|
| `Summascribe.py` | The whole app — PDF loading, chunking, LLM pipeline, Streamlit UI |
| `data/` | Where uploaded PDFs are written |
| `Images/` | Background and sidebar art |
| `config.toml` | Streamlit dark theme |
| `font.css` | Custom font styling |
| `PythonPackages.txt` | Dependency list |
| `LLM Folder download.txt` | Pointer to the model weights |

## Running it locally

**1. Clone and install**

```bash
git clone https://github.com/CaptainBlue793/Summascribe.git
cd Summascribe

pip install streamlit langchain transformers torch tiktoken accelerate \
            sentencepiece sentence_transformers pypdf python-multipart
```

**2. Download the model**

The app loads its weights from a local folder named `Lamini-1` in the project root:

```bash
git lfs install
git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M Lamini-1
```

**3. Run**

```bash
streamlit run Summascribe.py
```

Upload a PDF, set the summary strength, and hit **Summarize**.

## Notes & limitations

- **Text-only PDFs.** Scanned or image-based documents produce nothing — there's no OCR step.
- LaMini-Flan-T5 has a **512-token input window**, so very long documents get truncated rather than summarized
  section by section. A map-reduce chain over the chunks would fix this.
- Runs on CPU via `device_map="auto"`; a GPU makes it noticeably faster but isn't required.

## Tech stack

Python · LangChain · HuggingFace Transformers · PyTorch · LaMini-Flan-T5-248M · Streamlit
