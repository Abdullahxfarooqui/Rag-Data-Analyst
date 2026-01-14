# 📊 Excel File Format Guide

## Supported Format: .xls (Excel 97-2003)

Your app now supports **Excel .xls files** using a pure Python library that doesn't crash! 

## If You Have .xlsx Files

Modern Excel files (.xlsx) still cause issues. Here's how to convert:

### Quick Conversion in Excel

1. **Open your .xlsx file** in Microsoft Excel
2. Click **File → Save As**
3. Choose save location
4. In the "Save as type" dropdown, select:
   - **Excel 97-2003 Workbook (*.xls)**
5. Click **Save**
6. Excel may warn about compatibility - click **Yes** or **Continue**
7. Upload the new .xls file to your app ✅

### What Gets Preserved

✅ **Preserved:**
- All data values
- All rows and columns
- Headers
- Multiple sheets
- Numbers, text, dates

⚠️ **May Be Lost:**
- Some advanced Excel features
- Certain formatting styles
- Very large files (>65,536 rows - Excel 2003 limit)

### If Your File Is Too Large for .xls

Excel 97-2003 format has a limit of 65,536 rows. If your file exceeds this:

**Option 1: Split into multiple .xls files**
- Filter data by date/category
- Save each as separate .xls file
- Upload multiple files to the app

**Option 2: Use CSV instead**
- File → Save As → CSV
- CSV has no row limits
- Works perfectly with the app

## Alternative: CSV Files

CSV files work great and have no limitations:
- File → Save As → CSV (Comma delimited)
- One file per sheet if you have multiple sheets
- No row limits, no crashes, fast upload

## Need Help?

The app will show a clear error if you upload .xlsx by mistake, telling you to convert it first.

---

**Bottom line:** Save as .xls (Excel 97-2003) and your Excel files will work! 🎉
