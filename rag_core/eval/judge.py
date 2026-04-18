"""LLM-as-judge faithfulness scoring using Gemini Flash."""

from __future__ import annotations

import logging
import time

from google import genai

from rag_core.api_retry import with_retry
from rag_core.config import GEMINI_API_KEY, GENERATION_MODEL
from rag_core.generation.prompts import JUDGE_PROMPT

logger = logging.getLogger(__name__)


class LLMJudge:
    """Score answer faithfulness using Gemini Flash as a judge."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def score_faithfulness(
        self, query: str, answer: str, ground_truth: str
    ) -> float:
        """Rate faithfulness 1-5, returned normalized to 0-1.

        Returns 0.0 if the model response can't be parsed.
        """
        prompt = JUDGE_PROMPT.format(
            question=query, ground_truth=ground_truth, answer=answer
        )

        response = with_retry(
            lambda: self._client.models.generate_content(
                model=GENERATION_MODEL,
                contents=[prompt],
            )
        )

        text = (response.text or "").strip()
        try:
            score = int(text[0])  # First character should be 1-5
            return (score - 1) / 4.0  # Normalize to 0-1
        except (ValueError, IndexError):
            logger.warning("Could not parse judge score from: %r", text)
            return 0.0

    def batch_judge(
        self,
        items: list[dict],
        sleep_s: float = 1.0,
    ) -> float:
        """Score a batch of (query, answer, ground_truth) dicts.

        Returns mean faithfulness score (0-1).
        """
        scores = []
        for i, item in enumerate(items):
            score = self.score_faithfulness(
                item["query"], item["answer"], item["ground_truth"]
            )
            scores.append(score)
            if i < len(items) - 1:
                time.sleep(sleep_s)

            if (i + 1) % 20 == 0:
                logger.info("Judged %d / %d (mean so far: %.3f)", i + 1, len(items), sum(scores) / len(scores))

        return sum(scores) / len(scores) if scores else 0.0
