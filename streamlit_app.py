"""Streamlit Cloud entry point (auto-detected at repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 13):
    import streamlit as st

    st.set_page_config(page_title="WC 2026 Predictor", layout="wide")
    st.error(
        "This app requires **Python 3.11 or 3.12**. Streamlit Cloud is using "
        f"**Python {sys.version_info.major}.{sys.version_info.minor}**, which breaks ML dependencies.\n\n"
        "Fix: **Manage app → Settings → Python version → 3.11** → Save → Reboot."
    )
    st.stop()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import main

main()
