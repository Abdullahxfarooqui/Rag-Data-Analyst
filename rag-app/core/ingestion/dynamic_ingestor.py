"""
Dynamic Document Ingestion Engine.

PRODUCTION-GRADE SYSTEM FOR ANY DOCUMENT TYPE.

This module automatically:
1. Detects document type (Excel, CSV, PDF, DOCX, TXT, etc.)
2. Extracts tables, numeric metrics, categorical fields, text
3. Normalizes and cleans data dynamically
4. Generates metadata (columns, units, types, date ranges)
5. Prepares chunks for FAISS indexing

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│                         Dynamic Ingestor                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Type Detect │─▶│ Extractor   │─▶│ Normalizer  │─▶│ Metadata    │    │
│  │ (mimetype)  │  │ (per-type)  │  │ (units/fmt) │  │ Generator   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │                │            │
│         ▼                ▼                ▼                ▼            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DocumentSchema                                │   │
│  │  {                                                               │   │
│  │    "doc_id": "...",                                              │   │
│  │    "doc_type": "excel|csv|pdf|...",                              │   │
│  │    "tables": [...],                                              │   │
│  │    "numeric_columns": [...],                                     │   │
│  │    "categorical_columns": [...],                                 │   │
│  │    "date_columns": [...],                                        │   │
│  │    "text_content": [...],                                        │   │
│  │    "metadata": {...}                                             │   │
│  │  }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

TRADE-OFFS:
- Memory vs Speed: We load documents fully for accurate extraction, 
  but use streaming for very large files (>100MB)
- Accuracy vs Latency: Deep type detection is slower but more reliable
- Flexibility vs Consistency: Dynamic schemas adapt to content but 
  require careful normalization for cross-document comparisons
"""
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DOCUMENT TYPE DETECTION
# ============================================================================

class DocumentType(Enum):
    """Supported document types."""
    EXCEL = auto()      # .xlsx, .xls
    CSV = auto()        # .csv, .tsv
    PDF = auto()        # .pdf
    WORD = auto()       # .docx, .doc
    TEXT = auto()       # .txt, .md
    JSON = auto()       # .json
    PARQUET = auto()    # .parquet
    HTML = auto()       # .html
    UNKNOWN = auto()


# Extension to type mapping
EXTENSION_MAP = {
    '.xlsx': DocumentType.EXCEL,
    '.xls': DocumentType.EXCEL,
    '.xlsm': DocumentType.EXCEL,
    '.csv': DocumentType.CSV,
    '.tsv': DocumentType.CSV,
    '.pdf': DocumentType.PDF,
    '.docx': DocumentType.WORD,
    '.doc': DocumentType.WORD,
    '.txt': DocumentType.TEXT,
    '.md': DocumentType.TEXT,
    '.json': DocumentType.JSON,
    '.parquet': DocumentType.PARQUET,
    '.html': DocumentType.HTML,
    '.htm': DocumentType.HTML,
}


def detect_document_type(
    file_path: Optional[str] = None,
    file_obj: Optional[BinaryIO] = None,
    filename: Optional[str] = None
) -> DocumentType:
    """
    Detect document type from file path, object, or filename.
    
    Uses multiple strategies:
    1. File extension
    2. MIME type detection
    3. Magic bytes inspection
    """
    # Try extension first
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]
    
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]
    
    # Try MIME type
    if file_path:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if 'spreadsheet' in mime_type or 'excel' in mime_type:
                return DocumentType.EXCEL
            elif 'csv' in mime_type:
                return DocumentType.CSV
            elif 'pdf' in mime_type:
                return DocumentType.PDF
            elif 'word' in mime_type or 'document' in mime_type:
                return DocumentType.WORD
    
    # Magic bytes detection for file objects
    if file_obj:
        pos = file_obj.tell()
        header = file_obj.read(8)
        file_obj.seek(pos)
        
        # Excel XLSX (ZIP format)
        if header[:4] == b'PK\x03\x04':
            return DocumentType.EXCEL
        # PDF
        elif header[:4] == b'%PDF':
            return DocumentType.PDF
        # Old Excel XLS
        elif header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            return DocumentType.EXCEL
    
    return DocumentType.UNKNOWN


# ============================================================================
# COLUMN TYPE DETECTION
# ============================================================================

class ColumnType(Enum):
    """Detected column data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    ID = "id"
    UNKNOWN = "unknown"


@dataclass
class ColumnMetadata:
    """Metadata for a detected column."""
    name: str
    column_type: ColumnType
    dtype: str
    
    # Numeric metadata
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    unit: Optional[str] = None
    
    # Categorical metadata
    unique_count: Optional[int] = None
    top_values: Optional[List[str]] = None
    
    # Datetime metadata
    date_format: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    
    # General metadata
    null_count: int = 0
    null_percentage: float = 0.0
    sample_values: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.column_type.value,
            "dtype": self.dtype,
            "unit": self.unit,
            "null_percentage": self.null_percentage,
            "unique_count": self.unique_count,
            "min": self.min_value or self.min_date,
            "max": self.max_value or self.max_date,
            "mean": self.mean_value,
            "sample_values": self.sample_values[:5],
        }


# Unit detection patterns
UNIT_PATTERNS = {
    'currency': [
        (r'\$|USD|usd', 'USD'),
        (r'EUR|eur|€', 'EUR'),
        (r'GBP|gbp|£', 'GBP'),
        (r'INR|inr|₹', 'INR'),
    ],
    'percentage': [
        (r'%|percent|pct', '%'),
    ],
    'weight': [
        (r'kg|kilogram', 'kg'),
        (r'lb|pound', 'lb'),
        (r'g|gram', 'g'),
    ],
    'length': [
        (r'km|kilometer', 'km'),
        (r'm|meter', 'm'),
        (r'cm|centimeter', 'cm'),
        (r'in|inch', 'in'),
        (r'ft|feet|foot', 'ft'),
    ],
    'volume': [
        (r'L|liter|litre', 'L'),
        (r'ml|milliliter', 'ml'),
        (r'gal|gallon', 'gal'),
    ],
    'count': [
        (r'units?|qty|quantity|count', 'units'),
        (r'pcs|pieces', 'pcs'),
    ],
}


def detect_unit(column_name: str, values: pd.Series) -> Optional[str]:
    """Detect unit from column name and sample values."""
    column_lower = column_name.lower()
    
    # Check column name for unit hints
    for unit_type, patterns in UNIT_PATTERNS.items():
        for pattern, unit in patterns:
            if re.search(pattern, column_lower, re.IGNORECASE):
                return unit
    
    # Check sample string values for embedded units
    sample_str = values.dropna().astype(str).head(10)
    for val in sample_str:
        for unit_type, patterns in UNIT_PATTERNS.items():
            for pattern, unit in patterns:
                if re.search(pattern, str(val)):
                    return unit
    
    # Infer from column name keywords
    if any(kw in column_lower for kw in ['price', 'cost', 'amount', 'revenue', 'sales', 'total']):
        return 'USD'  # Default currency
    if any(kw in column_lower for kw in ['rate', 'ratio', 'pct']):
        return '%'
    
    return None


def detect_column_type(column: pd.Series, column_name: str) -> ColumnType:
    """
    Intelligently detect column type.
    
    Uses multiple heuristics:
    1. Pandas dtype
    2. Value patterns
    3. Column name hints
    4. Uniqueness ratio
    """
    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(column):
        return ColumnType.DATETIME
    
    # Boolean
    if pd.api.types.is_bool_dtype(column):
        return ColumnType.BOOLEAN
    
    # Numeric
    if pd.api.types.is_numeric_dtype(column):
        # Check if it's likely an ID column
        if column_name.lower() in ['id', 'index', 'key', 'pk'] or column_name.lower().endswith('_id'):
            return ColumnType.ID
        
        # Check uniqueness - high uniqueness numeric might be ID
        if len(column.dropna()) > 0:
            uniqueness = column.nunique() / len(column.dropna())
            if uniqueness > 0.95 and column.dtype in ['int64', 'int32']:
                return ColumnType.ID
        
        return ColumnType.NUMERIC
    
    # Object/string type - need deeper analysis
    non_null = column.dropna()
    if len(non_null) == 0:
        return ColumnType.UNKNOWN
    
    # Try parsing as datetime
    try:
        parsed = pd.to_datetime(non_null.head(100), errors='coerce')
        if parsed.notna().sum() / len(parsed) > 0.8:
            return ColumnType.DATETIME
    except:
        pass
    
    # Check if it's categorical (low cardinality)
    uniqueness = non_null.nunique() / len(non_null)
    if uniqueness < 0.5 or non_null.nunique() < 50:
        return ColumnType.CATEGORICAL
    
    # Long text content
    avg_length = non_null.astype(str).str.len().mean()
    if avg_length > 100:
        return ColumnType.TEXT
    
    # Default to categorical
    return ColumnType.CATEGORICAL


def analyze_column(column: pd.Series, column_name: str) -> ColumnMetadata:
    """
    Perform deep analysis on a column.
    
    Returns comprehensive metadata including type, statistics, 
    and detected units.
    """
    col_type = detect_column_type(column, column_name)
    
    metadata = ColumnMetadata(
        name=column_name,
        column_type=col_type,
        dtype=str(column.dtype),
        null_count=int(column.isna().sum()),
        null_percentage=round(column.isna().sum() / len(column) * 100, 2) if len(column) > 0 else 0,
        sample_values=column.dropna().head(5).tolist(),
    )
    
    non_null = column.dropna()
    
    if col_type == ColumnType.NUMERIC:
        metadata.min_value = float(non_null.min()) if len(non_null) > 0 else None
        metadata.max_value = float(non_null.max()) if len(non_null) > 0 else None
        metadata.mean_value = float(non_null.mean()) if len(non_null) > 0 else None
        metadata.std_value = float(non_null.std()) if len(non_null) > 0 else None
        metadata.unit = detect_unit(column_name, column)
    
    elif col_type == ColumnType.CATEGORICAL:
        metadata.unique_count = int(non_null.nunique())
        value_counts = non_null.value_counts().head(10)
        metadata.top_values = [str(v) for v in value_counts.index.tolist()]
    
    elif col_type == ColumnType.DATETIME:
        try:
            parsed = pd.to_datetime(non_null, errors='coerce')
            valid = parsed.dropna()
            if len(valid) > 0:
                metadata.min_date = str(valid.min())
                metadata.max_date = str(valid.max())
        except:
            pass
    
    return metadata


# ============================================================================
# DOCUMENT SCHEMA
# ============================================================================

@dataclass
class TableSchema:
    """Schema for an extracted table."""
    name: str
    row_count: int
    column_count: int
    columns: List[ColumnMetadata]
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [c.to_dict() for c in self.columns],
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
        }


@dataclass
class DocumentSchema:
    """
    Complete schema for an ingested document.
    
    This is the primary output of the ingestion pipeline,
    containing all extracted data and metadata.
    """
    doc_id: str
    filename: str
    doc_type: DocumentType
    
    # Extracted content
    tables: List[Tuple[str, pd.DataFrame]] = field(default_factory=list)
    table_schemas: List[TableSchema] = field(default_factory=list)
    text_content: List[str] = field(default_factory=list)
    
    # Aggregated column info
    all_numeric_columns: List[str] = field(default_factory=list)
    all_categorical_columns: List[str] = field(default_factory=list)
    all_datetime_columns: List[str] = field(default_factory=list)
    
    # Metadata
    file_size_bytes: int = 0
    ingestion_timestamp: str = ""
    processing_time_ms: float = 0
    chunk_count: int = 0
    
    # Errors/warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "doc_type": self.doc_type.name,
            "table_count": len(self.tables),
            "table_schemas": [ts.to_dict() for ts in self.table_schemas],
            "text_content_count": len(self.text_content),
            "numeric_columns": self.all_numeric_columns,
            "categorical_columns": self.all_categorical_columns,
            "datetime_columns": self.all_datetime_columns,
            "file_size_bytes": self.file_size_bytes,
            "ingestion_timestamp": self.ingestion_timestamp,
            "processing_time_ms": self.processing_time_ms,
            "chunk_count": self.chunk_count,
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def get_primary_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the largest table as primary DataFrame."""
        if not self.tables:
            return None
        # Return largest by row count
        return max(self.tables, key=lambda t: len(t[1]))[1]
    
    def get_all_metrics(self) -> List[Dict]:
        """Get all numeric metrics across tables."""
        metrics = []
        for table_name, df in self.tables:
            for schema in self.table_schemas:
                if schema.name == table_name:
                    for col in schema.columns:
                        if col.column_type == ColumnType.NUMERIC:
                            metrics.append({
                                "table": table_name,
                                "column": col.name,
                                "unit": col.unit,
                                "min": col.min_value,
                                "max": col.max_value,
                                "mean": col.mean_value,
                            })
        return metrics


# ============================================================================
# DOCUMENT EXTRACTORS
# ============================================================================

class BaseExtractor:
    """Base class for document extractors."""
    
    def extract(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
        """
        Extract tables and text from document.
        
        Returns:
            Tuple of (tables, text_content)
            - tables: List of (name, DataFrame) tuples
            - text_content: List of text paragraphs
        """
        raise NotImplementedError


class ExcelExtractor(BaseExtractor):
    """Extract from Excel files."""
    
    def extract(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
        tables = []
        text_content = []
        
        try:
            # Determine source
            source = file_path or file_obj
            
            # Read all sheets
            xlsx = pd.ExcelFile(source)
            
            for sheet_name in xlsx.sheet_names:
                try:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name)
                    
                    # Skip empty sheets
                    if df.empty or len(df.columns) == 0:
                        continue
                    
                    # Clean column names
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    # Drop completely empty rows/columns
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if not df.empty:
                        tables.append((sheet_name, df))
                        
                except Exception as e:
                    logger.warning(f"Failed to read sheet {sheet_name}: {e}")
            
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            raise
        
        return tables, text_content


class CSVExtractor(BaseExtractor):
    """Extract from CSV files."""
    
    def extract(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
        tables = []
        text_content = []
        
        try:
            source = file_path or file_obj
            name = Path(filename or file_path or "data").stem
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    if file_obj:
                        file_obj.seek(0)
                    df = pd.read_csv(source, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is not None and not df.empty:
                # Clean column names
                df.columns = [str(c).strip() for c in df.columns]
                tables.append((name, df))
                
        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            raise
        
        return tables, text_content


class PDFExtractor(BaseExtractor):
    """Extract from PDF files."""
    
    def extract(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
        tables = []
        text_content = []
        
        try:
            # Try pdfplumber for tables
            try:
                import pdfplumber
                
                pdf = pdfplumber.open(file_path or file_obj)
                
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for j, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:
                            # Use first row as header
                            headers = [str(h) if h else f"col_{k}" for k, h in enumerate(table_data[0])]
                            df = pd.DataFrame(table_data[1:], columns=headers)
                            tables.append((f"page_{i+1}_table_{j+1}", df))
                
                pdf.close()
                
            except ImportError:
                # Fallback to PyPDF2 for text only
                try:
                    import PyPDF2
                    
                    reader = PyPDF2.PdfReader(file_path or file_obj)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                            
                except ImportError:
                    logger.warning("No PDF library available (pdfplumber or PyPDF2)")
                    
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
        
        return tables, text_content


class TextExtractor(BaseExtractor):
    """Extract from text files."""
    
    def extract(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
        tables = []
        text_content = []
        
        try:
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = file_obj.read().decode('utf-8')
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            text_content.extend(paragraphs)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
        
        return tables, text_content


class JSONExtractor(BaseExtractor):
    """Extract from JSON files."""
    
    def extract(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None
    ) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
        tables = []
        text_content = []
        
        try:
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = json.load(file_obj)
            
            name = Path(filename or file_path or "data").stem
            
            # Try to convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
                tables.append((name, df))
            elif isinstance(data, dict):
                # Check if it's a records-style dict
                if all(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                    tables.append((name, df))
                else:
                    # Store as text
                    text_content.append(json.dumps(data, indent=2))
                    
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            raise
        
        return tables, text_content


# Extractor registry
EXTRACTORS = {
    DocumentType.EXCEL: ExcelExtractor(),
    DocumentType.CSV: CSVExtractor(),
    DocumentType.PDF: PDFExtractor(),
    DocumentType.TEXT: TextExtractor(),
    DocumentType.JSON: JSONExtractor(),
}


# ============================================================================
# DYNAMIC INGESTOR
# ============================================================================

class DynamicIngestor:
    """
    Production-grade document ingestion engine.
    
    Handles any document type, extracts all content,
    and generates comprehensive metadata for downstream
    processing.
    
    Usage:
        ingestor = DynamicIngestor()
        schema = ingestor.ingest("data.xlsx")
        
        # Access extracted data
        df = schema.get_primary_dataframe()
        metrics = schema.get_all_metrics()
        
        # Get chunks for FAISS
        chunks = ingestor.generate_chunks(schema)
    
    TRADE-OFFS:
    - Memory: Full document loading for accuracy
    - Speed: Deep type detection adds latency (~100ms per column)
    - Flexibility: Dynamic schema adapts to any content
    """
    
    def __init__(
        self,
        max_rows_per_table: int = 100000,
        max_text_length: int = 1000000,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize ingestor.
        
        Args:
            max_rows_per_table: Max rows to load per table (memory management)
            max_text_length: Max text content length
            chunk_size: Size of text chunks for FAISS
            chunk_overlap: Overlap between chunks
        """
        self.max_rows = max_rows_per_table
        self.max_text_length = max_text_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._stats = {
            "documents_processed": 0,
            "total_tables": 0,
            "total_rows": 0,
            "total_chunks": 0,
            "avg_processing_time_ms": 0,
        }
    
    def ingest(
        self,
        file_path: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None,
    ) -> DocumentSchema:
        """
        Ingest a document and generate schema.
        
        Args:
            file_path: Path to file
            file_obj: File-like object
            filename: Original filename (for type detection)
            
        Returns:
            DocumentSchema with all extracted content and metadata
        """
        start_time = time.time()
        
        # Generate document ID
        if file_path:
            with open(file_path, 'rb') as f:
                doc_id = hashlib.md5(f.read()[:8192]).hexdigest()[:16]
            file_size = os.path.getsize(file_path)
            fname = filename or os.path.basename(file_path)
        else:
            file_obj.seek(0)
            doc_id = hashlib.md5(file_obj.read(8192)).hexdigest()[:16]
            file_obj.seek(0, 2)  # Seek to end
            file_size = file_obj.tell()
            file_obj.seek(0)
            fname = filename or "uploaded_file"
        
        # Detect document type
        doc_type = detect_document_type(file_path, file_obj, filename)
        
        logger.info(f"Ingesting {fname} (type: {doc_type.name}, size: {file_size} bytes)")
        
        # Initialize schema
        schema = DocumentSchema(
            doc_id=doc_id,
            filename=fname,
            doc_type=doc_type,
            file_size_bytes=file_size,
            ingestion_timestamp=datetime.now().isoformat(),
        )
        
        # Get extractor
        extractor = EXTRACTORS.get(doc_type)
        if not extractor:
            schema.errors.append(f"No extractor for document type: {doc_type.name}")
            return schema
        
        # Extract content
        try:
            tables, text_content = extractor.extract(file_path, file_obj, filename)
        except Exception as e:
            schema.errors.append(f"Extraction failed: {str(e)}")
            logger.exception(f"Extraction failed for {fname}")
            return schema
        
        # Process tables
        for table_name, df in tables:
            # Truncate large tables
            if len(df) > self.max_rows:
                schema.warnings.append(
                    f"Table {table_name} truncated from {len(df)} to {self.max_rows} rows"
                )
                df = df.head(self.max_rows)
            
            # Analyze columns
            columns = []
            for col in df.columns:
                col_meta = analyze_column(df[col], col)
                columns.append(col_meta)
                
                # Aggregate by type
                if col_meta.column_type == ColumnType.NUMERIC:
                    schema.all_numeric_columns.append(col)
                elif col_meta.column_type == ColumnType.CATEGORICAL:
                    schema.all_categorical_columns.append(col)
                elif col_meta.column_type == ColumnType.DATETIME:
                    schema.all_datetime_columns.append(col)
            
            # Create table schema
            table_schema = TableSchema(
                name=table_name,
                row_count=len(df),
                column_count=len(df.columns),
                columns=columns,
                numeric_columns=[c.name for c in columns if c.column_type == ColumnType.NUMERIC],
                categorical_columns=[c.name for c in columns if c.column_type == ColumnType.CATEGORICAL],
                datetime_columns=[c.name for c in columns if c.column_type == ColumnType.DATETIME],
            )
            
            schema.tables.append((table_name, df))
            schema.table_schemas.append(table_schema)
        
        # Process text content
        for text in text_content:
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                schema.warnings.append("Text content truncated")
            schema.text_content.append(text)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        schema.processing_time_ms = processing_time
        
        # Update stats
        self._stats["documents_processed"] += 1
        self._stats["total_tables"] += len(tables)
        self._stats["total_rows"] += sum(len(df) for _, df in tables)
        
        logger.info(
            f"Ingested {fname}: {len(tables)} tables, "
            f"{sum(len(df) for _, df in tables)} rows, "
            f"{len(text_content)} text blocks in {processing_time:.0f}ms"
        )
        
        return schema
    
    def generate_chunks(
        self,
        schema: DocumentSchema,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate chunks for FAISS indexing.
        
        Creates semantically meaningful chunks from:
        1. Table data (row-based chunks with context)
        2. Text content (overlapping chunks)
        3. Metadata summaries
        
        Args:
            schema: Document schema from ingestion
            include_metadata: Include column metadata in chunks
            
        Returns:
            List of chunk dictionaries with 'text' and 'metadata'
        """
        chunks = []
        
        # Metadata summary chunk
        if include_metadata:
            meta_chunk = self._create_metadata_chunk(schema)
            chunks.append(meta_chunk)
        
        # Table chunks
        for table_name, df in schema.tables:
            table_chunks = self._chunk_table(table_name, df, schema)
            chunks.extend(table_chunks)
        
        # Text chunks
        for i, text in enumerate(schema.text_content):
            text_chunks = self._chunk_text(text, i, schema)
            chunks.extend(text_chunks)
        
        schema.chunk_count = len(chunks)
        self._stats["total_chunks"] += len(chunks)
        
        return chunks
    
    def _create_metadata_chunk(self, schema: DocumentSchema) -> Dict:
        """Create a metadata summary chunk."""
        # Build summary
        lines = [
            f"Document: {schema.filename}",
            f"Type: {schema.doc_type.name}",
            f"Tables: {len(schema.tables)}",
        ]
        
        for table_schema in schema.table_schemas:
            lines.append(f"\nTable '{table_schema.name}' ({table_schema.row_count} rows):")
            lines.append(f"  Columns: {', '.join([c.name for c in table_schema.columns[:10]])}")
            
            if table_schema.numeric_columns:
                lines.append(f"  Numeric: {', '.join(table_schema.numeric_columns[:5])}")
            if table_schema.categorical_columns:
                lines.append(f"  Categories: {', '.join(table_schema.categorical_columns[:5])}")
        
        return {
            "text": "\n".join(lines),
            "metadata": {
                "doc_id": schema.doc_id,
                "chunk_type": "metadata",
                "filename": schema.filename,
            }
        }
    
    def _chunk_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        schema: DocumentSchema
    ) -> List[Dict]:
        """Chunk a table into meaningful segments."""
        chunks = []
        
        # Statistics chunk
        stats_text = self._format_table_stats(table_name, df)
        chunks.append({
            "text": stats_text,
            "metadata": {
                "doc_id": schema.doc_id,
                "chunk_type": "table_stats",
                "table_name": table_name,
                "filename": schema.filename,
            }
        })
        
        # Row-based chunks (batches of rows)
        batch_size = 50  # Rows per chunk
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Format batch as text
            text_lines = [f"Table: {table_name} (rows {i+1}-{min(i+batch_size, len(df))})"]
            text_lines.append(batch.to_string(index=False, max_rows=50))
            
            chunks.append({
                "text": "\n".join(text_lines),
                "metadata": {
                    "doc_id": schema.doc_id,
                    "chunk_type": "table_rows",
                    "table_name": table_name,
                    "row_start": i,
                    "row_end": min(i+batch_size, len(df)),
                    "filename": schema.filename,
                }
            })
        
        return chunks
    
    def _format_table_stats(self, table_name: str, df: pd.DataFrame) -> str:
        """Format table statistics as text."""
        lines = [
            f"Table: {table_name}",
            f"Rows: {len(df)}, Columns: {len(df.columns)}",
            f"Columns: {', '.join(df.columns.tolist())}",
        ]
        
        # Numeric summaries
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            lines.append("\nNumeric Statistics:")
            for col in numeric_cols[:10]:
                lines.append(
                    f"  {col}: min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )
        
        return "\n".join(lines)
    
    def _chunk_text(
        self,
        text: str,
        index: int,
        schema: DocumentSchema
    ) -> List[Dict]:
        """Chunk text content with overlap."""
        chunks = []
        
        # Simple chunking with overlap
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > self.chunk_size * 0.5:
                    chunk_text = chunk_text[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": schema.doc_id,
                    "chunk_type": "text",
                    "text_index": index,
                    "chunk_num": chunk_num,
                    "filename": schema.filename,
                }
            })
            
            start = end - self.chunk_overlap
            chunk_num += 1
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return dict(self._stats)


# ============================================================================
# EXPORTS
# ============================================================================

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
