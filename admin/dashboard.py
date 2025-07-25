import streamlit as st
from pathlib import Path

REPORT_DIR = Path("reports")

def main():
    st.title("📑 Admin Report Console")
    for pdf in REPORT_DIR.glob("*.pdf"):
        parts = pdf.stem.split("_")
        cand = parts[1]
        st.write(f"**{cand}** – {pdf.name}")
        with open(pdf, "rb") as f:
            st.download_button(
                "Download", data=f, file_name=pdf.name, mime="application/pdf")

if __name__ == "__main__":
    main()
