"""
Minimal RAG Data Analyst - Test Version
Stripped down to find segfault source
"""
import streamlit as st

st.set_page_config(
    page_title="RAG Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Data Analyst")
st.write("✅ App loaded successfully!")
st.write("This is a minimal test version to verify Streamlit Cloud deployment works.")

# Test basic imports one by one
import_status = []

try:
    import pandas as pd
    import_status.append(("pandas", "✅"))
except Exception as e:
    import_status.append(("pandas", f"❌ {e}"))

try:
    import numpy as np
    import_status.append(("numpy", "✅"))
except Exception as e:
    import_status.append(("numpy", f"❌ {e}"))

try:
    import plotly.express as px
    import_status.append(("plotly", "✅"))
except Exception as e:
    import_status.append(("plotly", f"❌ {e}"))

try:
    import requests
    import_status.append(("requests", "✅"))
except Exception as e:
    import_status.append(("requests", f"❌ {e}"))

try:
    import json
    import_status.append(("json", "✅"))
except Exception as e:
    import_status.append(("json", f"❌ {e}"))

st.subheader("Import Status")
for lib, status in import_status:
    st.write(f"- {lib}: {status}")

st.success("All basic imports successful!")
