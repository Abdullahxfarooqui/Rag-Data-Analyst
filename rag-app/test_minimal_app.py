"""
Ultra-minimal Streamlit app to test if basic libraries work.
No imports from core/ modules.
"""
import streamlit as st

st.title("🔬 Minimal Test App")
st.write("✅ Streamlit loaded successfully!")

# Test imports one by one
try:
    import numpy as np
    st.write("✅ NumPy loaded successfully!")
    st.write(f"NumPy version: {np.__version__}")
except Exception as e:
    st.error(f"❌ NumPy failed: {e}")

try:
    import pandas as pd
    st.write("✅ Pandas loaded successfully!")
    st.write(f"Pandas version: {pd.__version__}")
except Exception as e:
    st.error(f"❌ Pandas failed: {e}")

try:
    import plotly
    st.write("✅ Plotly loaded successfully!")
    st.write(f"Plotly version: {plotly.__version__}")
except Exception as e:
    st.error(f"❌ Plotly failed: {e}")

try:
    import requests
    st.write("✅ Requests loaded successfully!")
    st.write(f"Requests version: {requests.__version__}")
except Exception as e:
    st.error(f"❌ Requests failed: {e}")

try:
    import openpyxl
    st.write("✅ OpenPyXL loaded successfully!")
    st.write(f"OpenPyXL version: {openpyxl.__version__}")
except Exception as e:
    st.error(f"❌ OpenPyXL failed: {e}")

st.success("🎉 All base libraries loaded!")
