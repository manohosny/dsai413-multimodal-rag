"""Shared test fixtures."""

from __future__ import annotations

import pytest

from rag_core.models import PageRecord, Segment
from pathlib import Path


SAMPLE_MARKDOWN = """\
# Quarterly Financial Report

Revenue increased by 15% year-over-year, driven by strong demand
in the enterprise segment. Operating expenses remained flat.

| Metric | Q1 2023 | Q2 2023 | Change |
|--------|---------|---------|--------|
| Revenue | $4.2B | $4.8B | +14% |
| Net Income | $1.1B | $1.3B | +18% |

Figure 3: Revenue breakdown by segment showing enterprise growth
outpacing consumer by 2x.

[1] Excludes one-time restructuring charges of $200M.
"""

SAMPLE_MARKDOWN_TABLES_ONLY = """\
| Year | Revenue |
|------|---------|
| 2022 | $3.5B |
| 2023 | $4.2B |
"""

SAMPLE_MARKDOWN_NO_SPECIAL = """\
This is a plain text paragraph with no tables, charts, or footnotes.

Another paragraph of plain text discussing financial performance
in the current fiscal year.
"""


@pytest.fixture
def sample_markdown() -> str:
    return SAMPLE_MARKDOWN


@pytest.fixture
def sample_page_record(tmp_path: Path) -> PageRecord:
    img_path = tmp_path / "00001.png"
    img_path.write_bytes(b"fake-png-data")
    return PageRecord(
        corpus_id=1,
        doc_id="test_doc_001",
        page_number=1,
        image_path=img_path,
        markdown=SAMPLE_MARKDOWN,
    )
