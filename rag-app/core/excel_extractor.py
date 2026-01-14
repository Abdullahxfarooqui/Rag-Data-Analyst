"""
Pure Python Excel extractor - NO pandas, NO numpy, NO segfaults.
Supports both .xlsx and .xls files.
"""
import io
from typing import Dict, Any, List
from pathlib import Path


def extract_excel_pure(file_content: bytes, filename: str) -> str:
    """
    Extract text from Excel files using PURE PYTHON only.
    NO pandas, NO numpy, NO compiled dependencies.
    
    Args:
        file_content: Raw file bytes
        filename: Original filename for extension detection
        
    Returns:
        Plain text representation of all sheets
    """
    ext = Path(filename).suffix.lower()
    
    if ext == '.xlsx':
        return _extract_xlsx(file_content)
    elif ext == '.xls':
        return _extract_xls(file_content)
    else:
        return f"Error: Unsupported file format '{ext}'. Please use .xlsx or .xls files."


def _extract_xlsx(file_content: bytes) -> str:
    """Extract from .xlsx using openpyxl (pure Python)."""
    try:
        import openpyxl
        from openpyxl import load_workbook
    except ImportError:
        return "Error: openpyxl not installed. Cannot read .xlsx files."
    
    try:
        workbook = load_workbook(
            filename=io.BytesIO(file_content),
            read_only=True,
            data_only=True  # Get values, not formulas
        )
        
        all_sheets_text = []
        
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            
            # Get all rows
            rows = []
            headers = []
            
            for row_idx, row in enumerate(sheet.iter_rows(values_only=True), 1):
                # Skip completely empty rows
                if not any(cell is not None and str(cell).strip() for cell in row):
                    continue
                
                # First non-empty row is headers
                if not headers:
                    headers = [str(cell).strip() if cell is not None else f"Column{i+1}" 
                              for i, cell in enumerate(row)]
                    continue
                
                # Build row data
                row_data = []
                for header, cell in zip(headers, row):
                    if cell is not None and str(cell).strip():
                        value = str(cell).strip()
                        row_data.append(f"{header}: {value}")
                
                if row_data:
                    rows.append(" | ".join(row_data))
            
            if rows:
                sheet_text = f"## Sheet: {sheet_name}\n\n"
                sheet_text += f"Headers: {', '.join(headers)}\n\n"
                sheet_text += "\n".join(rows)
                all_sheets_text.append(sheet_text)
        
        workbook.close()
        
        if not all_sheets_text:
            return "No data found in Excel file."
        
        return "\n\n---\n\n".join(all_sheets_text)
        
    except Exception as e:
        return f"Error reading .xlsx file: {str(e)}"


def _extract_xls(file_content: bytes) -> str:
    """Extract from .xls using xlrd 1.2.0 (pure Python)."""
    try:
        import xlrd
    except ImportError:
        return "Error: xlrd not installed. Cannot read .xls files."
    
    try:
        workbook = xlrd.open_workbook(file_contents=file_content)
        
        all_sheets_text = []
        
        for sheet_idx in range(workbook.nsheets):
            sheet = workbook.sheet_by_index(sheet_idx)
            sheet_name = sheet.name
            
            if sheet.nrows == 0:
                continue
            
            # Get headers from first row
            headers = []
            for col_idx in range(sheet.ncols):
                cell = sheet.cell(0, col_idx)
                headers.append(str(cell.value).strip() if cell.value else f"Column{col_idx+1}")
            
            # Get data rows
            rows = []
            for row_idx in range(1, sheet.nrows):
                row_data = []
                has_data = False
                
                for col_idx in range(sheet.ncols):
                    cell = sheet.cell(row_idx, col_idx)
                    if cell.value is not None and str(cell.value).strip():
                        value = str(cell.value).strip()
                        row_data.append(f"{headers[col_idx]}: {value}")
                        has_data = True
                
                if has_data:
                    rows.append(" | ".join(row_data))
            
            if rows:
                sheet_text = f"## Sheet: {sheet_name}\n\n"
                sheet_text += f"Headers: {', '.join(headers)}\n\n"
                sheet_text += "\n".join(rows)
                all_sheets_text.append(sheet_text)
        
        if not all_sheets_text:
            return "No data found in Excel file."
        
        return "\n\n---\n\n".join(all_sheets_text)
        
    except Exception as e:
        return f"Error reading .xls file: {str(e)}"


def get_excel_metadata(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Get basic metadata without pandas.
    
    Returns:
        Dict with sheet names and row counts
    """
    ext = Path(filename).suffix.lower()
    
    try:
        if ext == '.xlsx':
            import openpyxl
            workbook = openpyxl.load_workbook(
                filename=io.BytesIO(file_content),
                read_only=True,
                data_only=True
            )
            sheets = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                row_count = sum(1 for _ in sheet.iter_rows())
                sheets.append({"name": sheet_name, "rows": row_count})
            workbook.close()
            return {"sheets": sheets, "format": "xlsx"}
            
        elif ext == '.xls':
            import xlrd
            workbook = xlrd.open_workbook(file_contents=file_content)
            sheets = []
            for sheet_idx in range(workbook.nsheets):
                sheet = workbook.sheet_by_index(sheet_idx)
                sheets.append({"name": sheet.name, "rows": sheet.nrows})
            return {"sheets": sheets, "format": "xls"}
            
    except Exception as e:
        return {"error": str(e), "sheets": []}
    
    return {"error": "Unsupported format", "sheets": []}
