"""Opt-in real-model validation, separate from the lightweight test suite."""
import json

from Summascribe import SAMPLE
from summarization import LocalSummarizer, model_source


def main():
    model = LocalSummarizer(model_source(), "Auto")
    short = model.summarize([{"page": 1, "text": SAMPLE}], max_tokens=80)
    long_text = "\n\n".join(f"Section {i + 1}. {SAMPLE}" for i in range(5))
    long = model.summarize([{"page": 1, "text": long_text}], max_tokens=80)
    assert short["summary"] and long["summary"]
    assert len(long["sections"]) > 1
    print(json.dumps({"device": model.device, "input_limit": model.input_limit, "long_sections": len(long["sections"]), "short_summary": short["summary"], "long_summary": long["summary"]}, indent=2))


if __name__ == "__main__":
    main()
