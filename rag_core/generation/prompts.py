"""Prompt templates for QA generation and LLM-as-judge evaluation."""

from __future__ import annotations

QA_PROMPT = """\
You are answering a question about financial document pages.
The pages are shown as images. Relevant text extracts are also provided.

Extracted text chunks:
{text_chunks}

Rules:
1. Answer ONLY from what is visible in the images and text chunks.
2. If the answer is not present, say "The retrieved pages do not contain this information."
3. For numeric answers, quote the exact number and its row/column context.
4. End with "Sources: [page IDs used]"

Question: {question}
"""

JUDGE_PROMPT = """\
You are evaluating the faithfulness of an answer to a question about financial documents.

Question: {question}
Ground truth answer: {ground_truth}
System answer: {answer}

Rate the system answer on a scale of 1-5:
1 = Completely wrong or fabricated
2 = Partially correct but with significant errors
3 = Mostly correct but missing key details
4 = Correct with minor omissions
5 = Fully correct and faithful to the ground truth

Respond with ONLY a single integer (1-5), nothing else.
"""
