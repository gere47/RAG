"""
LexTemporal AI - Professional Legal Intelligence Platform
Graph-Grounded Temporal RAG with Contradiction-Resilient QA
Production-Grade Streamlit Application
"""

import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import time
import uuid
import json
from dataclasses import dataclass, field

from src.query_engine import QueryEngine, QueryResult
from src.config import config
from src.logger import get_logger
from src.agentic_engine import AgenticQueryEngine
from src.agentic_engine import MasterLegalAgent


logger = get_logger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LexTemporal AI | Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/gere47/RAG',
        'Report a bug': 'https://github.com/gere47/RAG/issues',
        'About': '''
        # LexTemporal AI
        
        **Graph-Grounded Temporal RAG for Legal Documents**
        
        Production-grade legal intelligence platform that resolves 
        contradictions across evolving documents using graph-based 
        temporal reasoning.
        
        **Core Capabilities:**
        - Hybrid Search (Vector + BM25)
        - Cross-Encoder Reranking
        - Neo4j Graph Integration
        - Temporal Contradiction Resolution
        - Local LLM (Llama 3.2)
        
        **Version:** 2.0.0
        **License:** MIT
        '''
    }
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* ===== GLOBAL ===== */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* ===== MAIN CONTAINER ===== */
    .main > div {
        padding: 1rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* ===== HERO HEADER ===== */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
        border-radius: 24px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        color: #ffffff;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-badges {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 8px 18px;
        border-radius: 40px;
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.2s ease;
    }
    
    .hero-badge:hover {
        background: rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    
    /* ===== CHAT MESSAGES ===== */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .user-message-wrapper {
        display: flex;
        justify-content: flex-end;
        margin: 20px 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 24px 24px 6px 24px;
        max-width: 75%;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        font-weight: 500;
        line-height: 1.6;
        animation: slideInRight 0.3s ease;
    }
    
    .assistant-message-wrapper {
        display: flex;
        justify-content: flex-start;
        margin: 20px 0;
    }
    
    .assistant-message {
        background: #1e293b;
        color: #f1f5f9;
        padding: 20px 28px;
        border-radius: 24px 24px 24px 6px;
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155;
        line-height: 1.7;
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .typing-indicator {
        display: flex;
        gap: 6px;
        padding: 16px 24px;
        background: #1e293b;
        border-radius: 24px;
        width: fit-content;
        border: 1px solid #334155;
    }
    
    .typing-indicator span {
        width: 10px;
        height: 10px;
        background: #6366f1;
        border-radius: 50%;
        animation: pulse 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    /* ===== SOURCE CARDS ===== */
    .source-card {
        background: #0f172a;
        border-radius: 16px;
        padding: 18px;
        margin: 12px 0;
        border: 1px solid #334155;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: #6366f1;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .source-title {
        font-weight: 700;
        color: #f1f5f9;
        font-size: 1.1rem;
    }
    
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-current {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-historical {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .badge-graph {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    .source-meta {
        display: flex;
        gap: 20px;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #334155;
    }
    
    /* ===== METRICS CARDS ===== */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 24px 0;
    }
    
    .metric-card {
        background: #1e293b;
        border-radius: 20px;
        padding: 24px 20px;
        border: 1px solid #334155;
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        border-color: #6366f1;
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    
    .metric-trend {
        font-size: 0.8rem;
        color: #10b981;
        margin-top: 6px;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    section[data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* ===== INPUT STYLING ===== */
    .stChatInput > div {
        border-radius: 40px !important;
        border: 2px solid #334155 !important;
        background: #1e293b !important;
        padding: 8px 8px 8px 20px !important;
    }
    
    .stChatInput input {
        color: #f1f5f9 !important;
    }
    
    .stChatInput input::placeholder {
        color: #64748b !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader > div {
        border: 2px dashed #334155 !important;
        border-radius: 20px !important;
        background: #1e293b !important;
        padding: 30px !important;
        transition: all 0.2s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #6366f1 !important;
        background: #1e293b !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 20px !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6366f1;
    }
    
    /* ===== WELCOME SCREEN ===== */
    .welcome-container {
        text-align: center;
        padding: 60px 40px;
        background: #1e293b;
        border-radius: 32px;
        border: 1px solid #334155;
        margin: 40px 0;
    }
    
    .welcome-icon {
        font-size: 5rem;
        margin-bottom: 24px;
    }
    
    .welcome-title {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
    
    .welcome-text {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto 32px;
    }
    
    .example-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .example-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-card:hover {
        border-color: #6366f1;
        background: #1e293b;
        transform: translateY(-4px);
    }
    
    .example-card-icon {
        font-size: 1.8rem;
        margin-bottom: 12px;
    }
    
    .example-card-title {
        color: #f1f5f9;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .example-card-text {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* ===== TIMELINE ===== */
    .timeline-container {
        background: #1e293b;
        border-radius: 20px;
        padding: 24px;
        margin: 24px 0;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
@dataclass
class Message:
    role: str
    content: str
    sources: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: int = 0


def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "engine" not in st.session_state:
        with st.spinner("🚀 Initializing LexTemporal AI..."):
            st.session_state.engine = QueryEngine()
    
        if "manifest_df" not in st.session_state:
            manifest_path = config.paths.project_root / "document_manifest.csv"
        if manifest_path.exists():
            st.session_state.manifest_df = pd.read_csv(manifest_path)
        else:
            st.session_state.manifest_df = pd.DataFrame()
    
    if "show_timeline" not in st.session_state:
        st.session_state.show_timeline = False
    
    if "target_date" not in st.session_state:
        st.session_state.target_date = None
    
    if "use_specific_date" not in st.session_state:
        st.session_state.use_specific_date = False
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = True
    if "engine" not in st.session_state:
        with st.spinner("🚀 Initializing LexTemporal AI Agent..."):
            st.session_state.engine = AgenticQueryEngine()


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_hero():
    """Render professional hero header."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">⚖️ LexTemporal AI</div>
        <div class="hero-subtitle">Graph-Grounded Temporal RAG for Contradiction-Resilient Legal Intelligence</div>
        <div class="hero-badges">
            <span class="hero-badge">🕸️ Graph-Grounded</span>
            <span class="hero-badge">⏳ Temporal-Aware</span>
            <span class="hero-badge">🔍 Hybrid Search</span>
            <span class="hero-badge">🎯 Cross-Encoder Reranking</span>
            <span class="hero-badge">🤖 Llama 3.2</span>
            <span class="hero-badge">📊 3108 Chunks</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    """Render metrics dashboard."""
    engine = st.session_state.engine
    manifest = st.session_state.manifest_df
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-value">{}</div>
            <div class="metric-label">Documents</div>
            <div class="metric-trend">↑ 8 indexed</div>
        </div>
        """.format(len(manifest)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🔍</div>
            <div class="metric-value">{}</div>
            <div class="metric-label">Vector Chunks</div>
            <div class="metric-trend">all-MiniLM-L6-v2</div>
        </div>
        """.format(engine.collection.count()), unsafe_allow_html=True)
    
    with col3:
        status = "🟢 Connected" if engine.graph_enabled else "🟡 Limited"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🕸️</div>
            <div class="metric-value">Neo4j</div>
            <div class="metric-label">{}</div>
            <div class="metric-trend">SUPERSEDES enabled</div>
        </div>
        """.format(status), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <div class="metric-value">Llama 3.2</div>
            <div class="metric-label">3B Parameters</div>
            <div class="metric-trend">Local • Private</div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render professional sidebar."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #f1f5f9; font-weight: 700; margin-bottom: 5px;">⚖️ LexTemporal</h1>
            <p style="color: #94a3b8; font-size: 0.9rem;">v2.0.0 • Production</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Temporal Controls
        st.markdown("### ⏳ Temporal Context")
        
        st.session_state.use_specific_date = st.checkbox(
            "Use specific effective date",
            value=st.session_state.use_specific_date,
            help="Filter documents effective on or before this date"
        )
        
        if st.session_state.use_specific_date:
            target_date = st.date_input(
                "Effective Date",
                value=date(2026, 4, 13),
                help="Documents effective after this date are treated as future/inactive"
            )
            st.session_state.target_date = str(target_date)
            st.info(f"📅 Showing documents effective ≤ {target_date}")
        else:
            st.session_state.target_date = None
            st.success("📅 Showing current/latest versions")
        
        st.divider()
        
        # Document Timeline
        st.markdown("### 📅 Document Evolution")
        
        if st.button("📈 Show Timeline", use_container_width=True):
            st.session_state.show_timeline = not st.session_state.show_timeline
        
        if st.session_state.show_timeline and not st.session_state.manifest_df.empty:
            render_timeline()
        
        st.divider()
        
        # Upload Section
        st.markdown("### 📤 Upload New Version")
        
        uploaded_file = st.file_uploader(
            "Add amendment or new document",
            type=["pdf"],
            help="Upload a new version (amendment, restatement, etc.)",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            with st.expander("📝 Document Details", expanded=True):
                doc_title = st.text_input("Title", value=uploaded_file.name.replace('.pdf', ''))
                effective_date = st.date_input("Effective Date", value=date.today())
                
                manifest = st.session_state.manifest_df
                options = ["None"] + (manifest['doc_id'].tolist() if not manifest.empty else [])
                supersedes = st.selectbox("Supersedes", options=options)
                
                if st.button("🚀 Process Document", use_container_width=True):
                    process_upload(uploaded_file, doc_title, str(effective_date), 
                                  None if supersedes == "None" else supersedes)
        
        st.divider()
        
        # Display Options
        st.markdown("### 🎛️ Display Options")
        st.session_state.show_sources = st.checkbox("Show source citations", value=True)
        st.session_state.show_metrics = st.checkbox("Show response metrics", value=True)
        
        st.divider()
        
        # System Stats
        st.markdown("### 📊 System Stats")
        
        engine = st.session_state.engine
        st.metric("Vector DB", f"{engine.collection.count():,} chunks")
        st.metric("Graph DB", "Connected" if engine.graph_enabled else "Disabled")
        st.metric("Model", config.ollama.model)
        
        st.divider()
        
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; color: #64748b; font-size: 0.8rem;">
            © 2026 LexTemporal AI<br/>
            Graph-Grounded Temporal RAG
        </div>
        """, unsafe_allow_html=True)


def render_timeline():
    """Render interactive document timeline."""
    df = st.session_state.manifest_df.copy()
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df = df.sort_values('effective_date')
    
    fig = px.timeline(
        df,
        x_start="effective_date",
        x_end=df['effective_date'] + pd.Timedelta(days=90),
        y="doc_title",
        color="doc_id",
        title="Document Evolution Timeline",
        labels={"effective_date": "Effective Date", "doc_title": "Document"},
        height=350,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(15, 23, 42, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color="#f1f5f9"),
        title=dict(font=dict(size=16, color="#f1f5f9")),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def process_upload(file, title: str, effective_date: str, supersedes: Optional[str]):
    """Process uploaded document."""
    from src.ingest import ingest_single_document
    
    with st.spinner("📄 Processing document..."):
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        
        pdf_path = config.paths.raw_pdfs_dir / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(file.getbuffer())
        
        new_row = pd.DataFrame([{
            'doc_id': doc_id,
            'doc_title': title,
            'effective_date': effective_date,
            'supersedes_doc_id': supersedes
        }])
        
        manifest = st.session_state.manifest_df
        manifest = pd.concat([manifest, new_row], ignore_index=True)
        manifest_path = config.paths.project_root / "document_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        st.session_state.manifest_df = manifest
        
        try:
            ingest_single_document(doc_id, effective_date)
            st.success(f"✅ Document {doc_id} processed successfully!")
            time.sleep(1.5)
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")


def render_welcome():
    """Render welcome screen with examples."""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">⚖️</div>
        <div class="welcome-title">Welcome to LexTemporal AI</div>
        <div class="welcome-text">
            Ask questions about your legal documents. I provide accurate, 
            contradiction-resilient answers using graph-based temporal reasoning.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 💡 Try these examples:")
    
    examples = [
        ("📄", "Current State", "What is the effective date of the original agreement?"),
        ("⏳", "Temporal Query", "What was the penalty fee in 2021?"),
        ("🔄", "Contradiction Test", "Has the penalty clause changed over time?"),
        ("🔍", "Specific Clause", "What does Section 1.1 state?"),
        ("⚖️", "Governing Law", "What is the governing law?"),
        ("👥", "Parties", "Who are the parties involved?"),
    ]
    
    cols = st.columns(3)
    for i, (icon, title, question) in enumerate(examples):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="example-card" onclick="this.style.transform='scale(0.98)'">
                <div class="example-card-icon">{icon}</div>
                <div class="example-card-title">{title}</div>
                <div class="example-card-text">{question}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(question, key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append(Message(role="user", content=question))
                st.rerun()


def render_message(msg: Message):
    """Render a single chat message."""
    if msg.role == "user":
        st.markdown(f"""
        <div class="user-message-wrapper">
            <div class="user-message">{msg.content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message-wrapper">
            <div class="assistant-message">{msg.content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.show_sources and msg.sources:
            with st.expander("📎 View Sources & Citations", expanded=False):
                for src in msg.sources:
                    is_current = src.get('is_current', True)
                    badge_class = "badge-current" if is_current else "badge-historical"
                    badge_text = "CURRENT" if is_current else "HISTORICAL"
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-title">📄 {src['doc_id']}</span>
                            <span class="badge {badge_class}">{badge_text}</span>
                        </div>
                        <div class="source-meta">
                            <span>📅 Effective: {src['effective_date']}</span>
                            <span>🔍 Chunk: {src['chunk_id']}</span>
                            <span>📊 Score: {src.get('score', 0):.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        if st.session_state.show_metrics and msg.response_time_ms:
            st.caption(f"⚡ Response time: {msg.response_time_ms / 1000:.2f}s")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def init_session_state():
    if "agent" not in st.session_state:
        with st.spinner("🚀 Initializing Master Legal Agent..."):
            st.session_state.agent = MasterLegalAgent()
    
    # Add mode selector to sidebar
    st.sidebar.selectbox(
        "Agent Mode",
        ["auto", "verify", "compare", "scenario", "monitor"],
        key="agent_mode"
    )


def main():
    """Main application entry point."""
    init_session_state()
    
    render_hero()
    render_sidebar()

    if prompt := st.chat_input("Ask anything..."):
        result = st.session_state.agent.process(
        prompt, 
        mode=st.session_state.agent_mode
        )
    
    if st.session_state.show_metrics:
        render_metrics()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            render_welcome()
        else:
            for msg in st.session_state.messages:
                render_message(msg)
    
    # Chat input
    if prompt := st.chat_input("Ask about your legal documents...", key="chat_input"):
        user_msg = Message(role="user", content=prompt)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        
        with st.chat_message("assistant"):
            with st.spinner(""):
                st.markdown("""
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
                """, unsafe_allow_html=True)
                
                result = st.session_state.engine.answer(
                    prompt,
                    target_date=st.session_state.target_date
                )
                
                assistant_msg = Message(
                    role="assistant",
                    content=result["answer"],
                    sources=result.get("sources", []),
                    response_time_ms=result.get("total_time_ms", 0)
                )
                
                st.session_state.messages.append(assistant_msg)
                st.rerun()


if __name__ == "__main__":
    main()

# # """
# # Streamlit web interface for Graph-Grounded Temporal RAG.
# # Usage: streamlit run app.py
# # """

# # import streamlit as st
# # from datetime import date
# # from pathlib import Path
# # import sys

# # # Add project root to path
# # sys.path.insert(0, str(Path(__file__).parent))

# # from src.query_engine import QueryEngine
# # from src.config import config
# # from src.logger import get_logger

# # logger = get_logger(__name__)

# # # Page config
# # st.set_page_config(
# #     page_title="Legal Graph RAG",
# #     page_icon="📜",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Custom CSS
# # st.markdown("""
# # <style>
# #     .stChatMessage { padding: 1rem; border-radius: 10px; }
# #     .source-box { 
# #         background-color: #f0f2f6; 
# #         padding: 10px; 
# #         border-radius: 5px; 
# #         font-size: 0.9em;
# #         margin: 5px 0;
# #     }
# #     .current { border-left: 4px solid #00cc66; }
# #     .historical { border-left: 4px solid #ff9900; opacity: 0.8; }
# # </style>
# # """, unsafe_allow_html=True)

# # # Header
# # st.title("📜 Graph‑Grounded Temporal RAG")
# # st.markdown("*Contradiction‑Resilient Question Answering over Evolving Legal Documents*")

# # # In the sidebar, before the settings section
# # with st.sidebar:
# #     st.header("📤 Upload New Document")
    
# #     uploaded_file = st.file_uploader(
# #         "Upload PDF",
# #         type=["pdf"],
# #         help="Upload a new version of a legal document"
# #     )
    
# #     if uploaded_file:
# #         doc_title = st.text_input("Document Title", value=uploaded_file.name)
# #         effective_date = st.date_input("Effective Date", value=date.today())
        
# #         # Find which document this supersedes
# #         import pandas as pd
# #         manifest = pd.read_csv(config.manifest_path)
# #         supersedes_options = ["None"] + manifest['doc_id'].tolist()
# #         supersedes = st.selectbox("Supersedes Document", options=supersedes_options)
        
# #         if st.button("Process Document"):
# #             with st.spinner("Processing..."):
# #                 # Save PDF
# #                 import uuid
# #                 new_id = f"doc_{str(uuid.uuid4())[:8]}"
# #                 pdf_path = config.RAW_PDFS_DIR / f"{new_id}.pdf"
                
# #                 with open(pdf_path, "wb") as f:
# #                     f.write(uploaded_file.getbuffer())
                
# #                 # Update manifest
# #                 new_row = pd.DataFrame([{
# #                     'doc_id': new_id,
# #                     'doc_title': doc_title,
# #                     'effective_date': str(effective_date),
# #                     'supersedes_doc_id': None if supersedes == "None" else supersedes
# #                 }])
# #                 manifest = pd.concat([manifest, new_row], ignore_index=True)
# #                 manifest.to_csv(config.manifest_path, index=False)
                
# #                 # Trigger pipeline for this single document
# #                 from src.ingest import ingest_single_document
# #                 ingest_single_document(new_id, str(effective_date))
                
# #                 st.success(f"✅ Document {new_id} processed!")
# #                 st.rerun()

# # # Sidebar
# # with st.sidebar:
# #     st.header("⚙️ Settings")
    
# #     # Date selector
# #     target_date = st.date_input(
# #         "📅 Effective Date (for temporal queries)",
# #         value=date(2026, 4, 8),
# #         help="The system will treat documents effective after this date as future/inactive"
# #     )
    
# #     st.divider()
    
# #     # System status
# #     st.header("📊 System Status")
    
# #     # Check components
# #     col1, col2 = st.columns(2)
    
# #     # Vector DB
# #     vectors_path = config.VECTORS_DIR
# #     if vectors_path.exists() and list(vectors_path.iterdir()):
# #         col1.success("✅ Vector DB")
# #     else:
# #         col1.warning("⚠️ No vectors")
    
# #     # Graph DB
# #     engine = QueryEngine()
# #     # Graph DB
# #     if "engine" not in st.session_state:
# #         with st.spinner("Loading models..."):
# #             st.session_state.engine = QueryEngine()

# #     if st.session_state.engine.graph_enabled:
# #         col2.success("✅ Neo4j Graph")
# #     else:
# #         col2.warning("⚠️ Graph disabled")
    
# #     # Document count
# #     manifest_path = config.manifest_path
# #     if manifest_path.exists():
# #         import pandas as pd
# #         df = pd.read_csv(manifest_path)
# #         st.metric("📄 Documents", len(df))
    
# #     st.divider()
    
# #     # Display options
# #     st.header("🔍 Display Options")
# #     show_sources = st.checkbox("Show source documents", value=True)
# #     show_graph_path = st.checkbox("Show graph traversal path", value=False)
# #     show_scores = st.checkbox("Show relevance scores", value=False)
    
# #     st.divider()
# #     st.caption("GraphRAG v1.0 | Local Llama 3.2 | Neo4j AuraDB")


# # # Initialize session state
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []
# #     # Add welcome message
# #     st.session_state.messages.append({
# #         "role": "assistant",
# #         "content": "👋 Hello! I'm a legal document assistant with temporal awareness. Ask me questions about the documents, and I'll provide answers based on the **current** versions (or historical ones if you specify a date).",
# #         "sources": []
# #     })

# # if "engine" not in st.session_state:
# #     with st.spinner("Loading models..."):
# #         st.session_state.engine = QueryEngine()

# # # Display chat history
# # for msg in st.session_state.messages:
# #     with st.chat_message(msg["role"]):
# #         st.markdown(msg["content"])
        
# #         # Show sources if available
# #         if show_sources and msg.get("sources"):
# #             with st.expander("📎 Sources", expanded=False):
# #                 for src in msg["sources"]:
# #                     status_class = "current" if src.get("is_current", True) else "historical"
# #                     status_emoji = "🟢" if src.get("is_current", True) else "🟡"
                    
# #                     st.markdown(f"""
# #                     <div class="source-box {status_class}">
# #                         {status_emoji} <strong>{src['doc_id']}</strong> | Effective: {src['effective_date']}<br/>
# #                         <small>Chunk: {src['chunk_id']}</small>
# #                         {f"<br/><small>Score: {src['score']:.3f}</small>" if show_scores else ""}
# #                     </div>
# #                     """, unsafe_allow_html=True)

# # # Chat input
# # if prompt := st.chat_input("Ask about the documents..."):
# #     # Add user message
# #     st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)
    
# #     # Generate response
# #     with st.chat_message("assistant"):
# #         with st.spinner("🔍 Retrieving and analyzing..."):
# #             result = st.session_state.engine.answer(
# #                 prompt,
# #                 target_date=str(target_date)
# #             )
            
# #             answer = result["answer"]
# #             sources = result["sources"]
            
# #             st.markdown(answer)
            
# #             # Show sources
# #             if show_sources and sources:
# #                 with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
# #                     for src in sources:
# #                         status_emoji = "🟢" if src["is_current"] else "🟡"
# #                         status_text = "CURRENT" if src["is_current"] else "HISTORICAL"
                        
# #                         st.markdown(f"""
# #                         **{status_emoji} {src['doc_id']}** ({status_text})  
# #                         Effective: `{src['effective_date']}`  
# #                         Chunk: `{src['chunk_id']}`
# #                         {f"Relevance: `{src['score']:.3f}`" if show_scores else ""}
# #                         """)
# #                         st.divider()
    
# #     # Save response
# #     st.session_state.messages.append({
# #         "role": "assistant",
# #         "content": answer,
# #         "sources": sources
# #     })

# # # Footer
# # st.divider()
# # col1, col2, col3 = st.columns(3)
# # with col1:
# #     st.metric("🔍 Retrieval", f"Hybrid (α={config.retrieval.hybrid_alpha})")
# # with col2:
# #     st.metric("🤖 Model", config.ollama.model)
# # with col3:
# #     st.metric("🔄 Graph", "Enabled" if st.session_state.engine.graph_enabled else "Disabled")



# """
# Graph-Grounded Temporal RAG - Professional Legal Document Intelligence Platform
# Production-ready Streamlit frontend with interactive features and dynamic UI.
# """

# import streamlit as st
# import pandas as pd
# from datetime import date, datetime, timedelta
# from pathlib import Path
# import sys
# import time
# import json
# from typing import Optional, List, Dict, Any
# import plotly.graph_objects as go
# import plotly.express as px

# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent))

# from src.query_engine import QueryEngine
# from src.config import config
# from src.logger import get_logger
# from src.ingest import ingest_single_document
# import os
# os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
# os.environ["ANONYMIZED_TELEMETRY"] = "false"

# logger = get_logger(__name__)

# # ============================================================================
# # Page Configuration
# # ============================================================================
# st.set_page_config(
#     page_title="LexTemporal AI | Legal Intelligence",
#     page_icon="⚖️",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://github.com/gere47/RAG',
#         'Report a bug': 'https://github.com/gere47/RAG/issues',
#         'About': '# Graph-Grounded Temporal RAG\n*Contradiction-Resilient Legal QA*'
#     }
# )

# # ============================================================================
# # Custom CSS for Professional Styling
# # ============================================================================
# st.markdown("""
# <style>
#     /* Import Google Fonts */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
    
#     /* Global Styles */
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#     }
    
#     /* Main Container */
#     .main > div {
#         padding-top: 1rem;
#     }
    
#     /* Header Gradient */
#     .header-gradient {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 16px;
#         margin-bottom: 2rem;
#         box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
#     }
    
#     .header-title {
#         color: white;
#         font-size: 2.5rem;
#         font-weight: 700;
#         margin-bottom: 0.5rem;
#         letter-spacing: -0.02em;
#     }
    
#     .header-subtitle {
#         color: rgba(255,255,255,0.9);
#         font-size: 1.1rem;
#         font-weight: 400;
#     }
    
#     /* Chat Messages */
#     .chat-container {
#         max-width: 900px;
#         margin: 0 auto;
#     }
    
#     .user-message {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 16px 20px;
#         border-radius: 20px 20px 4px 20px;
#         margin: 12px 0;
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
#         font-weight: 500;
#         animation: slideIn 0.3s ease;
#     }
    
#     .assistant-message {
#         background: #f8f9fa;
#         color: #1e1e2f;
#         padding: 20px 24px;
#         border-radius: 20px 20px 20px 4px;
#         margin: 12px 0;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.06);
#         border: 1px solid #e9ecef;
#         line-height: 1.7;
#         animation: slideIn 0.3s ease;
#     }
    
#     @keyframes slideIn {
#         from {
#             opacity: 0;
#             transform: translateY(10px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     /* Source Cards */
#     .source-card {
#         background: white;
#         border-radius: 12px;
#         padding: 16px;
#         margin: 10px 0;
#         border-left: 4px solid #667eea;
#         box-shadow: 0 2px 6px rgba(0,0,0,0.04);
#         transition: all 0.2s ease;
#     }
    
#     .source-card:hover {
#         box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
#         transform: translateY(-2px);
#     }
    
#     .source-card.historical {
#         border-left-color: #ffa500;
#         opacity: 0.85;
#     }
    
#     .badge {
#         display: inline-block;
#         padding: 4px 12px;
#         border-radius: 20px;
#         font-size: 0.75rem;
#         font-weight: 600;
#         text-transform: uppercase;
#         letter-spacing: 0.5px;
#     }
    
#     .badge-current {
#         background: #d4edda;
#         color: #155724;
#     }
    
#     .badge-historical {
#         background: #fff3cd;
#         color: #856404;
#     }
    
#     .badge-graph {
#         background: #667eea;
#         color: white;
#     }
    
#     /* Metrics Cards */
#     .metric-card {
#         background: white;
#         border-radius: 16px;
#         padding: 20px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.04);
#         border: 1px solid #e9ecef;
#         text-align: center;
#         transition: transform 0.2s ease;
#     }
    
#     .metric-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 8px 24px rgba(0,0,0,0.08);
#     }
    
#     .metric-value {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1e1e2f;
#         line-height: 1.2;
#     }
    
#     .metric-label {
#         font-size: 0.9rem;
#         color: #6c757d;
#         font-weight: 500;
#         text-transform: uppercase;
#         letter-spacing: 0.5px;
#     }
    
#     /* Sidebar Styling */
#     .sidebar .stButton > button {
#         width: 100%;
#         border-radius: 12px;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         font-weight: 600;
#         padding: 12px;
#         border: none;
#         transition: all 0.2s ease;
#     }
    
#     .sidebar .stButton > button:hover {
#         transform: scale(1.02);
#         box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
#     }
    
#     /* File Uploader */
#     .upload-box {
#         border: 2px dashed #dee2e6;
#         border-radius: 16px;
#         padding: 30px;
#         text-align: center;
#         background: #f8f9fa;
#         transition: all 0.2s ease;
#     }
    
#     .upload-box:hover {
#         border-color: #667eea;
#         background: #f0f4ff;
#     }
    
#     /* Input Box */
#     .stChatInput > div {
#         border-radius: 30px !important;
#         border: 2px solid #e9ecef !important;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;
#     }
    
#     .stChatInput > div:focus-within {
#         border-color: #667eea !important;
#         box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15) !important;
#     }
    
#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 8px;
#         height: 8px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: #f1f1f1;
#         border-radius: 10px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: #c1c1c1;
#         border-radius: 10px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: #a8a8a8;
#     }
    
#     /* Timeline Visualization */
#     .timeline-container {
#         background: white;
#         border-radius: 16px;
#         padding: 20px;
#         margin: 20px 0;
#         border: 1px solid #e9ecef;
#     }
    
#     /* Dark mode support */
#     @media (prefers-color-scheme: dark) {
#         .assistant-message {
#             background: #2d2d3f;
#             color: #f0f0f0;
#             border-color: #3d3d4f;
#         }
#         .source-card {
#             background: #2d2d3f;
#             border-color: #3d3d4f;
#         }
#         .metric-card {
#             background: #2d2d3f;
#             border-color: #3d3d4f;
#         }
#         .metric-value {
#             color: #f0f0f0;
#         }
#     }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================================
# # Session State Initialization
# # ============================================================================
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "engine" not in st.session_state:
#     with st.spinner("🚀 Initializing LexTemporal AI..."):
#         st.session_state.engine = QueryEngine()

# if "manifest_df" not in st.session_state:
#     if config.manifest_path.exists():
#         st.session_state.manifest_df = pd.read_csv(config.manifest_path)
#     else:
#         st.session_state.manifest_df = pd.DataFrame()

# if "show_timeline" not in st.session_state:
#     st.session_state.show_timeline = False

# if "upload_success" not in st.session_state:
#     st.session_state.upload_success = False

# # ============================================================================
# # Helper Functions
# # ============================================================================
# def render_header():
#     """Render the professional header with gradient."""
#     col1, col2, col3 = st.columns([2, 3, 1])
    
#     with col2:
#         st.markdown("""
#         <div class="header-gradient">
#             <div class="header-title">⚖️ LexTemporal AI</div>
#             <div class="header-subtitle">Graph-Grounded Temporal RAG for Legal Intelligence</div>
#             <div style="margin-top: 15px; display: flex; gap: 10px;">
#                 <span class="badge badge-current">Contradiction-Resilient</span>
#                 <span class="badge badge-graph">Temporal-Aware</span>
#                 <span class="badge badge-graph">Hybrid Search</span>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)


# def render_sidebar():
#     """Render the professional sidebar with all controls."""
#     with st.sidebar:
#         # Logo and Title
#         st.markdown("""
#         <div style="text-align: center; padding: 20px 0;">
#             <h1 style="color: #1e1e2f; font-weight: 700; margin-bottom: 5px;">⚖️ LexTemporal</h1>
#             <p style="color: #6c757d; font-size: 0.9rem;">Legal Intelligence Platform</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.divider()
        
#         # System Status Section
#         st.markdown("### 📊 System Status")
        
#         engine = st.session_state.engine
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric(
#                 "📄 Documents",
#                 len(st.session_state.manifest_df),
#                 delta=None
#             )
#         with col2:
#             st.metric(
#                 "🔍 Chunks",
#                 engine.collection.count(),
#                 delta=None
#             )
        
#         col3, col4 = st.columns(2)
#         with col3:
#             graph_status = "🟢 Online" if engine.graph_enabled else "🟡 Limited"
#             st.metric("🕸️ Graph", graph_status)
#         with col4:
#             st.metric("🤖 LLM", "Llama 3.2")
        
#         st.divider()
        
#         # Temporal Controls
#         st.markdown("### ⏳ Temporal Context")
        
#         use_specific_date = st.checkbox("Use specific effective date", value=False)
        
#         if use_specific_date:
#             target_date = st.date_input(
#                 "Effective Date",
#                 value=date(2026, 4, 12),
#                 help="The system will prioritize documents effective on or before this date"
#             )
#             st.session_state.target_date = str(target_date)
#         else:
#             st.session_state.target_date = None
#             st.info("Using current/latest documents")
        
#         st.divider()
        
#         # Document Timeline
#         st.markdown("### 📅 Document Timeline")
        
#         if st.button("📈 Show Evolution Timeline", use_container_width=True):
#             st.session_state.show_timeline = not st.session_state.show_timeline
        
#         if st.session_state.show_timeline and len(st.session_state.manifest_df) > 0:
#             render_timeline()
        
#         st.divider()
        
#         # Upload Section
#         st.markdown("### 📤 Upload Document")
        
#         uploaded_file = st.file_uploader(
#             "Add new version",
#             type=["pdf"],
#             help="Upload a new version of a legal document",
#             label_visibility="collapsed"
#         )
        
#         if uploaded_file:
#             with st.expander("📝 Document Details", expanded=True):
#                 doc_title = st.text_input("Document Title", value=uploaded_file.name)
#                 effective_date = st.date_input("Effective Date", value=date.today())
                
#                 manifest = st.session_state.manifest_df
#                 supersedes_options = ["None"] + (manifest['doc_id'].tolist() if not manifest.empty else [])
#                 supersedes = st.selectbox("Supersedes Document", options=supersedes_options)
                
#                 if st.button("🚀 Process Document", use_container_width=True):
#                     process_uploaded_document(
#                         uploaded_file, 
#                         doc_title, 
#                         str(effective_date),
#                         None if supersedes == "None" else supersedes
#                     )
        
#         if st.session_state.upload_success:
#             st.success("✅ Document processed!")
#             st.session_state.upload_success = False
#             time.sleep(1)
#             st.rerun()
        
#         st.divider()
        
#         # Display Options
#         st.markdown("### 🎛️ Display Options")
#         show_sources = st.checkbox("Show source citations", value=True)
#         show_confidence = st.checkbox("Show confidence scores", value=False)
        
#         st.divider()
        
#         # Footer
#         st.markdown("""
#         <div style="text-align: center; padding: 20px 0; color: #6c757d; font-size: 0.8rem;">
#             Graph-Grounded Temporal RAG v1.0<br/>
#             © 2026 LexTemporal AI
#         </div>
#         """, unsafe_allow_html=True)
        
#         return show_sources, show_confidence


# def render_timeline():
#     """Render interactive document evolution timeline."""
#     df = st.session_state.manifest_df.copy()
#     df['effective_date'] = pd.to_datetime(df['effective_date'])
#     df = df.sort_values('effective_date')
    
#     fig = px.timeline(
#         df,
#         x_start="effective_date",
#         x_end=df['effective_date'] + pd.Timedelta(days=30),
#         y="doc_title",
#         color="doc_id",
#         title="Document Evolution Timeline",
#         labels={"effective_date": "Effective Date", "doc_title": "Document"},
#         height=300
#     )
    
#     fig.update_layout(
#         showlegend=False,
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(family="Inter", size=12),
#         margin=dict(l=10, r=10, t=40, b=10)
#     )
    
#     st.plotly_chart(fig, use_container_width=True)


# def process_uploaded_document(uploaded_file, title: str, effective_date: str, supersedes: Optional[str]):
#     """Process and ingest an uploaded document."""
#     import uuid
#     from src.ingest import ingest_single_document
    
#     with st.spinner("📄 Processing document..."):
#         # Generate new ID
#         doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        
#         # Save PDF
#         pdf_path = config.RAW_PDFS_DIR / f"{doc_id}.pdf"
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Update manifest
#         new_row = pd.DataFrame([{
#             'doc_id': doc_id,
#             'doc_title': title,
#             'effective_date': effective_date,
#             'supersedes_doc_id': supersedes
#         }])
        
#         manifest = st.session_state.manifest_df
#         manifest = pd.concat([manifest, new_row], ignore_index=True)
#         manifest.to_csv(config.manifest_path, index=False)
#         st.session_state.manifest_df = manifest
        
#         # Ingest into system
#         try:
#             ingest_single_document(doc_id, effective_date)
#             st.session_state.upload_success = True
#         except Exception as e:
#             st.error(f"Failed to ingest document: {e}")


# def render_chat_message(role: str, content: str, sources: List[Dict] = None, show_sources: bool = True):
#     """Render a beautifully styled chat message."""
#     if role == "user":
#         st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
        
#         if show_sources and sources:
#             with st.expander("📎 View Sources & Citations", expanded=False):
#                 for i, src in enumerate(sources, 1):
#                     is_current = src.get('is_current', True)
#                     card_class = "source-card" if is_current else "source-card historical"
#                     badge_class = "badge-current" if is_current else "badge-historical"
#                     badge_text = "CURRENT" if is_current else "HISTORICAL"
                    
#                     st.markdown(f"""
#                     <div class="{card_class}">
#                         <div style="display: flex; justify-content: space-between; align-items: center;">
#                             <strong style="font-size: 1rem;">📄 {src['doc_id']}</strong>
#                             <span class="badge {badge_class}">{badge_text}</span>
#                         </div>
#                         <div style="margin-top: 8px; color: #6c757d; font-size: 0.9rem;">
#                             Effective: {src['effective_date']} | Chunk: {src['chunk_id']}
#                         </div>
#                         <div style="margin-top: 8px; font-size: 0.9rem; opacity: 0.8;">
#                             Relevance Score: {src.get('score', 0):.3f}
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)


# def render_welcome_screen():
#     """Render welcome screen with example queries."""
#     st.markdown("""
#     <div style="max-width: 800px; margin: 40px auto; text-align: center;">
#         <h2 style="color: #1e1e2f; font-weight: 600; margin-bottom: 30px;">
#             👋 Welcome to LexTemporal AI
#         </h2>
#         <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 40px;">
#             Ask questions about your legal documents. I'll provide accurate, 
#             contradiction-resilient answers based on the current effective versions.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Example queries
#     st.markdown("#### 💡 Try these example queries:")
    
#     cols = st.columns(3)
#     examples = [
#         "What is the effective date of the original agreement?",
#         "What are the payment terms?",
#         "What is the governing law?",
#         "Who are the parties involved?",
#         "What is the penalty fee?",
#         "Show me the amendment history"
#     ]
    
#     for i, example in enumerate(examples):
#         with cols[i % 3]:
#             if st.button(example, key=f"example_{i}", use_container_width=True):
#                 st.session_state.messages.append({"role": "user", "content": example})
#                 st.rerun()


# # ============================================================================
# # Main Application
# # ============================================================================
# def main():
#     """Main application entry point."""
    
#     # Render header
#     render_header()
    
#     # Render sidebar and get display options
#     show_sources, show_confidence = render_sidebar()
    
#     # Main chat area
#     chat_container = st.container()
    
#     with chat_container:
#         if not st.session_state.messages:
#             render_welcome_screen()
#         else:
#             for msg in st.session_state.messages:
#                 with st.chat_message(msg["role"]):
#                     if msg["role"] == "user":
#                         st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
#                     else:
#                         st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
#                         if show_sources and msg.get("sources"):
#                             with st.expander("📎 View Sources", expanded=False):
#                                 for src in msg["sources"]:
#                                     is_current = src.get('is_current', True)
#                                     card_class = "source-card" if is_current else "source-card historical"
#                                     badge_class = "badge-current" if is_current else "badge-historical"
#                                     badge_text = "CURRENT" if is_current else "HISTORICAL"
                                    
#                                     st.markdown(f"""
#                                     <div class="{card_class}">
#                                         <div style="display: flex; justify-content: space-between;">
#                                             <strong>📄 {src['doc_id']}</strong>
#                                             <span class="badge {badge_class}">{badge_text}</span>
#                                         </div>
#                                         <div style="margin-top: 8px; color: #6c757d;">
#                                             Effective: {src['effective_date']} | Chunk: {src['chunk_id']}
#                                         </div>
#                                     </div>
#                                     """, unsafe_allow_html=True)
    
#     # Chat input
#     if prompt := st.chat_input("Ask about your legal documents..."):
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("🔍 Analyzing documents and resolving temporal context..."):
#                 # Progress animation
#                 progress_bar = st.progress(0)
                
#                 for i in range(100):
#                     time.sleep(0.01)
#                     progress_bar.progress(i + 1)
                
#                 progress_bar.empty()
                
#                 # Get answer
#                 result = st.session_state.engine.answer(
#                     prompt,
#                     target_date=st.session_state.get('target_date')
#                 )
                
#                 answer = result["answer"]
#                 sources = result["sources"]
#                 graph_used = result["graph_used"]
                
#                 # Display answer
#                 st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
                
#                 # Display sources
#                 if show_sources and sources:
#                     with st.expander("📎 View Sources & Citations", expanded=False):
#                         # Graph status
#                         if graph_used:
#                             st.success("🕸️ Graph-based temporal resolution applied")
                        
#                         for src in sources:
#                             is_current = src.get('is_current', True)
#                             card_class = "source-card" if is_current else "source-card historical"
#                             badge_class = "badge-current" if is_current else "badge-historical"
#                             badge_text = "CURRENT" if is_current else "HISTORICAL"
                            
#                             st.markdown(f"""
#                             <div class="{card_class}">
#                                 <div style="display: flex; justify-content: space-between;">
#                                     <strong>📄 {src['doc_id']}</strong>
#                                     <span class="badge {badge_class}">{badge_text}</span>
#                                 </div>
#                                 <div style="margin-top: 8px; color: #6c757d;">
#                                     Effective: {src['effective_date']} | Chunk: {src['chunk_id']}
#                                 </div>
#                             </div>
#                             """, unsafe_allow_html=True)
        
#         # Save response
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": answer,
#             "sources": sources
#         })


# if __name__ == "__main__":
#     main()