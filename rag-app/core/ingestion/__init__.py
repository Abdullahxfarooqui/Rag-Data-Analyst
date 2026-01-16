"""Ingestion package for dynamic document processing."""
from .dynamic_ingestor import (
    DynamicIngestor,
    DocumentSchema,
    TableSchema,
    ColumnMetadata,
    DocumentType,
    ColumnType,
    detect_document_type,
    analyze_column,
)

__all__ = [
    "DynamicIngestor",
    "DocumentSchema",
    "TableSchema",
    "ColumnMetadata",
    "DocumentType",
    "ColumnType",
    "detect_document_type",
    "analyze_column",
]
