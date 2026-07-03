"""
app.py — MediVision Streamlit UI (improved)
"""

import os
import sys
import tempfile
import streamlit as st
from PIL import Image
from fpdf import FPDF
import io

sys.path.append(os.path.dirname(__file__))

from extractor import load_vit, extract_single_image
from retriever import load_index, retrieve_similar_reports, format_rag_context
from generator import generate_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediVision — AI Radiology Report Generator",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #2a3550;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8b9fc1;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }
    .badge {
        display: inline-block;
        background: #0f3460;
        color: #4da6ff;
        border: 1px solid #4da6ff33;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 10px;
    }

    .report-card {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1.8rem;
        margin-top: 1rem;
    }
    .report-section {
        margin-bottom: 1.4rem;
    }
    .report-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .label-findings  { color: #4da6ff; }
    .label-impression { color: #56d364; }
    .label-recommendations { color: #f0a500; }
    .report-text {
        color: #cdd6e8;
        font-size: 0.95rem;
        line-height: 1.7;
        margin: 0;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #2a3550;
        margin: 1.2rem 0;
    }

    section[data-testid="stSidebar"] {
        background-color: #12151f;
        border-right: 1px solid #2a3550;
    }
    .sidebar-title {
        color: #4da6ff;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .tech-chip {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 6px;
        color: #cdd6e8;
        font-size: 0.85rem;
    }

    .warning-box {
        background: #2a1f0e;
        border: 1px solid #f0a50044;
        border-radius: 8px;
        padding: 10px 14px;
        color: #f0a500;
        font-size: 0.82rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    model, image_processor = load_vit()
    index, image_ids = load_index()
    return model, image_processor, index, image_ids


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
def generate_pdf(report_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "", ln=True)
    pdf.cell(0, 10, "MediVision", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 230)
    pdf.cell(0, 6, "AI-Powered Radiology Report", ln=True, align="C")

    pdf.ln(10)

    # Parse sections
    sections = {"FINDINGS": "", "IMPRESSION": "", "RECOMMENDATIONS": ""}
    current = None
    for line in report_text.split("\n"):
        line = line.strip().replace("**", "")
        if line.startswith("FINDINGS"):
            current = "FINDINGS"
        elif line.startswith("IMPRESSION"):
            current = "IMPRESSION"
        elif line.startswith("RECOMMENDATIONS"):
            current = "RECOMMENDATIONS"
        elif current and line:
            sections[current] += line + " "

    colors = {
        "FINDINGS":        (77, 166, 255),
        "IMPRESSION":      (86, 211, 100),
        "RECOMMENDATIONS": (240, 165, 0),
    }

    for section, text in sections.items():
        if not text.strip():
            continue
        r, g, b = colors[section]

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 8, section, ln=True)

        pdf.set_draw_color(r, g, b)
        pdf.set_line_width(0.4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, text.strip())
        pdf.ln(5)

    # Footer — use hyphen instead of em dash to avoid encoding error
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6,
        "Generated by MediVision AI | For educational purposes only - not for clinical use",
        align="C"
    )

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Parse report into sections
# ---------------------------------------------------------------------------
def parse_report_sections(report_text: str) -> dict:
    sections = {"FINDINGS": "", "IMPRESSION": "", "RECOMMENDATIONS": ""}
    current = None
    for line in report_text.split("\n"):
        clean = line.strip().replace("**", "")
        if clean.startswith("FINDINGS"):
            current = "FINDINGS"
        elif clean.startswith("IMPRESSION"):
            current = "IMPRESSION"
        elif clean.startswith("RECOMMENDATIONS"):
            current = "RECOMMENDATIONS"
        elif current and clean:
            sections[current] += clean + " "
    return sections


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🫁 MediVision</h1>
    <p>AI-Powered Chest X-Ray Report Generator</p>
    <span class="badge">ViT Vision Model</span>
    <span class="badge">FAISS Retrieval</span>
    <span class="badge">Gemini 2.5 Flash</span>
    <span class="badge">IU X-Ray Dataset</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">Technology Stack</p>', unsafe_allow_html=True)
    st.markdown('<div class="tech-chip">🔬 ViT - Visual feature extraction</div>', unsafe_allow_html=True)
    st.markdown('<div class="tech-chip">🗂️ FAISS - Similar case retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="tech-chip">🤖 Gemini 2.5 Flash - Report generation</div>', unsafe_allow_html=True)
    st.markdown('<div class="tech-chip">📊 7,430 training X-rays</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-title">Settings</p>', unsafe_allow_html=True)
    top_k = st.slider("Similar cases to retrieve", 1, 10, 5)
    show_similar = st.checkbox("Show retrieved similar cases", value=False)

    st.markdown("""
    <div class="warning-box">
        For educational purposes only.<br>Not intended for clinical use.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "PNG, JPG or JPEG up to 200MB",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    indication = st.text_input(
        "Clinical Indication (optional)",
        placeholder="e.g. shortness of breath, chest pain, routine check"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

with col2:
    st.markdown("#### Generated Report")

    if not uploaded_file:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px dashed #2a3550;border-radius:12px;
        padding:3rem;text-align:center;color:#4a5568;margin-top:0.5rem;">
            <div style="font-size:2.5rem;">🫁</div>
            <div style="margin-top:0.8rem;font-size:0.95rem;">
                Upload a chest X-ray to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        generate_btn = st.button(
            "🔍 Generate Report",
            type="primary",
            use_container_width=True
        )

        if generate_btn:
            with st.spinner("Analysing X-ray and generating report..."):
                try:
                    model, image_processor, index, image_ids = load_models()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    report = generate_report(
                        tmp_path,
                        indication=indication,
                        model=model,
                        image_processor=image_processor,
                    )

                    st.session_state["report"] = report
                    st.session_state["tmp_path"] = tmp_path

                except Exception as e:
                    st.error(f"Error: {e}")

        if "report" in st.session_state:
            report = st.session_state["report"]
            sections = parse_report_sections(report)

            st.markdown("""<div class="report-card">""", unsafe_allow_html=True)

            if sections["FINDINGS"]:
                st.markdown(f"""
                <div class="report-section">
                    <div class="report-label label-findings">Findings</div>
                    <p class="report-text">{sections["FINDINGS"].strip()}</p>
                </div>
                <hr class="section-divider">
                """, unsafe_allow_html=True)

            if sections["IMPRESSION"]:
                st.markdown(f"""
                <div class="report-section">
                    <div class="report-label label-impression">Impression</div>
                    <p class="report-text">{sections["IMPRESSION"].strip()}</p>
                </div>
                <hr class="section-divider">
                """, unsafe_allow_html=True)

            if sections["RECOMMENDATIONS"]:
                st.markdown(f"""
                <div class="report-section">
                    <div class="report-label label-recommendations">Recommendations</div>
                    <p class="report-text">{sections["RECOMMENDATIONS"].strip()}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Download as PDF
            pdf_bytes = generate_pdf(report)
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name="medivision_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            # Similar cases
            if show_similar and "tmp_path" in st.session_state:
                import pandas as pd
                st.markdown("---")
                st.markdown("#### Similar Cases Retrieved")
                query_vec = extract_single_image(
                    st.session_state["tmp_path"],
                    *load_models()[:2]
                )
                df = pd.read_csv(os.path.join("data", "dataset_index.csv"))
                similar = retrieve_similar_reports(
                    query_vec, index, image_ids, df, top_k=top_k
                )
                for i, r in enumerate(similar, 1):
                    with st.expander(
                        f"Case {i} - {r['image_id']} (similarity: {r['score']:.3f})"
                    ):
                        st.markdown(f"**Impression:** {r['impression']}")
                        st.markdown(f"**Findings:** {r['findings']}")