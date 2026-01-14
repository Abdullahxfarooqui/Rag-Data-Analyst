"""
PDF processor DISABLED - pdfplumber dependencies cause segfaults on Streamlit Cloud.
Focus on Excel/CSV for data analysis.
"""
from typing import Dict, Any, Optional, Callable


def extract_pdf_complete(
    file_content: bytes,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    PDF extraction disabled to prevent segfaults.
    Users should upload Excel or CSV files instead.
    """
    return {
        "text": "⚠️ PDF extraction is temporarily disabled.\n\nPlease upload your data as Excel (.xlsx, .xls) or CSV files instead.",
        "pages": [],
        "tables": [],
        "type": "pdf",
        "num_pages": 0,
        "num_tables": 0,
        "total_table_rows": 0,
        "is_dataset": False,
        "total_chars": 0,
        "error": "PDF extraction disabled to prevent segfaults on Streamlit Cloud. Use Excel/CSV instead."
    }


def get_page_count(file_content: bytes) -> int:
    """PDF disabled."""
    return 0


def clean_text(text: str) -> str:
    """Clean text utility."""
    if not text:
        return ""
    return text.strip()
