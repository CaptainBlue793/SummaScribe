"""Document reading and summarization workspace."""
import hashlib
import json
from pathlib import Path

import streamlit as st

from summarization import LocalSummarizer, extractive_summary, model_source, read_pdf, read_text, validate_pages

SAMPLE = """Urban gardens can improve access to fresh food in neighborhoods with limited grocery stores. Community volunteers maintain shared planting beds and organize seasonal harvests.

Rainwater collection reduces the need for municipal irrigation. Composting turns kitchen scraps into nutrients, reducing waste and improving soil health.

Successful projects need secure land access and clear maintenance responsibilities. Schools can use gardens to teach biology, nutrition, and teamwork.

City planners should consider accessibility, water availability, and soil contamination before approving a site. Regular soil testing and raised beds help reduce exposure to pollutants.

Long-term funding remains a challenge. Partnerships with local businesses can cover supplies, while volunteer training supports continuity between growing seasons."""


@st.cache_resource(show_spinner=False, max_entries=2)
def load_model(source, device):
    return LocalSummarizer(source, device)


def main():
    st.set_page_config(page_title="SummaScribe · Document studio", page_icon="✦", layout="wide")
    st.markdown(f"<style>{Path(__file__).with_name('style.css').read_text()}</style>", unsafe_allow_html=True)
    st.caption("SUMMASCRIBE / DOCUMENT STUDIO")
    st.title("Less reading. More understanding.")
    st.write("Turn a document into a focused summary, with its source close at hand.")
    with st.sidebar:
        st.header("Make it yours")
        method = st.radio("Summary method", ["Extractive", "Local AI"], help="Extractive selects original sentences with page references. Local AI rewrites using cached model weights.")
        sentences = st.slider("Summary sentences", 3, 20, 8)
        max_tokens = st.slider("AI output token limit", 40, 400, 180, disabled=method != "Local AI")
        device = st.selectbox("AI device", ["Auto", "CPU", "CUDA"], disabled=method != "Local AI")
        password = st.text_input("PDF password (if needed)", type="password")
        st.caption("Text is processed in memory on the app's host. Local AI uses existing model weights; it does not download them during a request.")
    upload_col, info_col = st.columns([3, 2], gap="large")
    with upload_col:
        uploaded = st.file_uploader("Drop in your document", type=["pdf", "txt", "md"])
        if st.button("Try a sample document"):
            st.session_state["notes"] = SAMPLE
        notes = st.text_area("Or paste your text", key="notes", height=200, max_chars=300_000)
    with info_col:
        st.subheader("Your reading companion")
        st.markdown("**Read across the whole source**  \nLong documents are processed in sections.\n\n**Keep the evidence**  \nExtractive summaries preserve original wording and page references.\n\n**Save your digest**  \nExport a summary or a structured report.")
        st.caption("Text-based PDFs only. OCR is required for scans. Maximum: 20 MB, 500 pages, 300,000 extracted characters.")
    pages = None
    name = "Pasted text"
    fingerprint = None
    try:
        if uploaded is not None:
            data = uploaded.getvalue()
            fingerprint = hashlib.sha256(data + password.encode()).hexdigest()
            name = uploaded.name
            previous = st.session_state.get("document")
            if previous and previous["fingerprint"] == fingerprint:
                pages = previous["pages"]
            else:
                with st.spinner("Reading your document…"):
                    pages = read_pdf(data, password) if uploaded.name.lower().endswith(".pdf") else read_text(data)
                st.session_state["document"] = {"fingerprint": fingerprint, "pages": pages}
        elif notes.strip():
            pages = validate_pages([{"page": 1, "text": notes.strip()}])
            fingerprint = hashlib.sha256(notes.encode()).hexdigest()
    except (ValueError, ImportError) as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else "PDF support needs pypdf. Install requirements.txt in starGPU.")
    selected_pages = pages
    if pages and len(pages) > 1:
        first, last = st.slider("Pages to summarize", 1, len(pages), (1, len(pages)))
        selected_pages = [page for page in pages if first <= page["page"] <= last]
    if st.button("Create summary", type="primary", disabled=not pages, use_container_width=True):
        try:
            with st.spinner("Building your summary…"):
                if method == "Extractive":
                    result = extractive_summary(selected_pages, sentences)
                else:
                    model = load_model(model_source(), device)
                    progress = st.progress(0, text="Starting local model…")
                    try:
                        result = model.summarize(selected_pages, max_tokens, lambda value, text: progress.progress(value, text=text))
                    finally:
                        progress.empty()
                result.update({"document": name, "fingerprint": fingerprint, "source_pages": selected_pages})
                st.session_state["summary"] = result
        except (OSError, ImportError) as exc:
            st.error("Local AI weights or dependencies are unavailable. Use Extractive, or follow the model setup in README.md.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    result = st.session_state.get("summary")
    if not result:
        return
    st.divider()
    st.caption(f"Last summary: {result['document']} · {result['method']}. Create a new summary after changing source or settings.")
    if fingerprint != result["fingerprint"]:
        st.info("The source has changed. The summary below still belongs to the previous document.")
    metrics = st.columns(4)
    words = len(result["summary"].split())
    for col, label, value in zip(metrics, ["Pages read", "Source words", "Summary words", "Length retained"], [result["pages"], result["source_words"], words, f"{words / max(result['source_words'], 1):.0%}"]):
        col.metric(label, value)
    digest, evidence, source = st.tabs(["Your digest", "Evidence & sections", "Original text"])
    with digest:
        with st.container(border=True):
            st.write(result["summary"])
        if result["method"] == "Local AI":
            st.caption("AI summaries can omit or misstate details. Check the original before relying on a claim.")
        report = f"# {result['document']}\n\nMethod: {result['method']}\n\n{result['summary']}\n"
        if result["evidence"]:
            report += "\n## Source references\n\n" + "\n".join(f"- Page {row['page']}: {row['text']}" for row in result["evidence"])
        a, b = st.columns(2)
        a.download_button("Download Markdown", report, "summascribe-summary.md", "text/markdown")
        b.download_button("Download JSON", json.dumps(result, indent=2, ensure_ascii=False), "summascribe-summary.json", "application/json")
    with evidence:
        for row in result["evidence"]:
            with st.container(border=True):
                st.caption(f"SOURCE / PAGE {row['page']}")
                st.write(row["text"])
        for index, section in enumerate(result.get("sections", []), 1):
            with st.expander(f"AI section {index}"):
                st.write(section)
        if not result["evidence"]:
            st.caption("AI sections summarize token-sized portions; they are not sentence-level citations.")
    with source:
        for page in result["source_pages"]:
            with st.expander(f"Page {page['page']}", expanded=len(result["source_pages"]) == 1):
                st.text(page["text"])


if __name__ == "__main__":
    main()
