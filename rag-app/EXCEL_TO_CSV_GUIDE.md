# 📊 How to Convert Excel to CSV

## Why CSV Only?

Excel processing libraries (openpyxl and xlrd) have native C extensions that cause server crashes on Streamlit Cloud with Python 3.13. CSV files work perfectly and are more reliable!

## Quick Conversion Guide

### Method 1: Using Excel/Google Sheets

1. **Open your Excel file** (.xlsx or .xls)
2. Click **File → Save As**
3. Choose location
4. In "Save as type" dropdown, select **CSV (Comma delimited) (*.csv)**
5. Click **Save**
6. If you have multiple sheets, repeat for each sheet

### Method 2: Using Google Sheets (Online)

1. Upload your Excel file to Google Drive
2. Right-click and select **Open with → Google Sheets**
3. Click **File → Download → Comma Separated Values (.csv)**
4. Your CSV file will download automatically

### Method 3: Using Python (Batch Conversion)

If you have many Excel files to convert:

```python
import pandas as pd

# Read Excel file
df = pd.read_excel('your_file.xlsx', sheet_name='Sheet1')

# Save as CSV
df.to_csv('your_file.csv', index=False)

print("✅ Converted successfully!")
```

## Multiple Sheets?

If your Excel file has multiple sheets:

### Option 1: Save each sheet separately
- In Excel: Select each sheet tab, then File → Save As → CSV
- Name each CSV file clearly (e.g., `sales_2023.csv`, `sales_2024.csv`)

### Option 2: Combine sheets using Python
```python
import pandas as pd

# Read all sheets
excel_file = pd.ExcelFile('your_file.xlsx')

# Save each sheet as CSV
for sheet_name in excel_file.sheet_names:
    df = excel_file.parse(sheet_name)
    df.to_csv(f'{sheet_name}.csv', index=False)
    print(f"✅ Saved {sheet_name}.csv")
```

## Benefits of CSV

✅ **Faster:** CSV files load 2-3x faster than Excel  
✅ **Smaller:** CSV files are typically 30-50% smaller  
✅ **Universal:** Works everywhere, no compatibility issues  
✅ **Reliable:** No crashes, no segfaults, no library issues  
✅ **Version Control:** Easy to track changes in Git  

## Your Data is Safe

**CSV files preserve:**
- ✅ All rows and columns
- ✅ All data values
- ✅ Headers and column names
- ✅ Numbers, text, and dates

**CSV files don't preserve:**
- ❌ Formulas (only the calculated values are saved)
- ❌ Formatting (colors, fonts, borders)
- ❌ Multiple sheets (need separate files)
- ❌ Charts and images

## Need Help?

If you need your formulas or formatting preserved, consider:
1. Taking screenshots of formatted tables for reference
2. Documenting formulas separately
3. Using CSV for the raw data analysis (which is what this app does)

---

**Questions?** The app works great with CSV files - they contain all your data and load reliably every time! 🎉
