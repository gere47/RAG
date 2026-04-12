# """
# Streamlit web interface for Graph-Grounded Temporal RAG.
# Usage: streamlit run app.py
# """

# import streamlit as st
# from datetime import date
# from pathlib import Path
# import sys

# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent))

# from src.query_engine import QueryEngine
# from src.config import config
# from src.logger import get_logger

# logger = get_logger(__name__)

# # Page config
# st.set_page_config(
#     page_title="Legal Graph RAG",
#     page_icon="📜",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .stChatMessage { padding: 1rem; border-radius: 10px; }
#     .source-box { 
#         background-color: #f0f2f6; 
#         padding: 10px; 
#         border-radius: 5px; 
#         font-size: 0.9em;
#         margin: 5px 0;
#     }
#     .current { border-left: 4px solid #00cc66; }
#     .historical { border-left: 4px solid #ff9900; opacity: 0.8; }
# </style>
# """, unsafe_allow_html=True)

# # Header
# st.title("📜 Graph‑Grounded Temporal RAG")
# st.markdown("*Contradiction‑Resilient Question Answering over Evolving Legal Documents*")

# # In the sidebar, before the settings section
# with st.sidebar:
#     st.header("📤 Upload New Document")
    
#     uploaded_file = st.file_uploader(
#         "Upload PDF",
#         type=["pdf"],
#         help="Upload a new version of a legal document"
#     )
    
#     if uploaded_file:
#         doc_title = st.text_input("Document Title", value=uploaded_file.name)
#         effective_date = st.date_input("Effective Date", value=date.today())
        
#         # Find which document this supersedes
#         import pandas as pd
#         manifest = pd.read_csv(config.manifest_path)
#         supersedes_options = ["None"] + manifest['doc_id'].tolist()
#         supersedes = st.selectbox("Supersedes Document", options=supersedes_options)
        
#         if st.button("Process Document"):
#             with st.spinner("Processing..."):
#                 # Save PDF
#                 import uuid
#                 new_id = f"doc_{str(uuid.uuid4())[:8]}"
#                 pdf_path = config.RAW_PDFS_DIR / f"{new_id}.pdf"
                
#                 with open(pdf_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())
                
#                 # Update manifest
#                 new_row = pd.DataFrame([{
#                     'doc_id': new_id,
#                     'doc_title': doc_title,
#                     'effective_date': str(effective_date),
#                     'supersedes_doc_id': None if supersedes == "None" else supersedes
#                 }])
#                 manifest = pd.concat([manifest, new_row], ignore_index=True)
#                 manifest.to_csv(config.manifest_path, index=False)
                
#                 # Trigger pipeline for this single document
#                 from src.ingest import ingest_single_document
#                 ingest_single_document(new_id, str(effective_date))
                
#                 st.success(f"✅ Document {new_id} processed!")
#                 st.rerun()

# # Sidebar
# with st.sidebar:
#     st.header("⚙️ Settings")
    
#     # Date selector
#     target_date = st.date_input(
#         "📅 Effective Date (for temporal queries)",
#         value=date(2026, 4, 8),
#         help="The system will treat documents effective after this date as future/inactive"
#     )
    
#     st.divider()
    
#     # System status
#     st.header("📊 System Status")
    
#     # Check components
#     col1, col2 = st.columns(2)
    
#     # Vector DB
#     vectors_path = config.VECTORS_DIR
#     if vectors_path.exists() and list(vectors_path.iterdir()):
#         col1.success("✅ Vector DB")
#     else:
#         col1.warning("⚠️ No vectors")
    
#     # Graph DB
#     engine = QueryEngine()
#     # Graph DB
#     if "engine" not in st.session_state:
#         with st.spinner("Loading models..."):
#             st.session_state.engine = QueryEngine()

#     if st.session_state.engine.graph_enabled:
#         col2.success("✅ Neo4j Graph")
#     else:
#         col2.warning("⚠️ Graph disabled")
    
#     # Document count
#     manifest_path = config.manifest_path
#     if manifest_path.exists():
#         import pandas as pd
#         df = pd.read_csv(manifest_path)
#         st.metric("📄 Documents", len(df))
    
#     st.divider()
    
#     # Display options
#     st.header("🔍 Display Options")
#     show_sources = st.checkbox("Show source documents", value=True)
#     show_graph_path = st.checkbox("Show graph traversal path", value=False)
#     show_scores = st.checkbox("Show relevance scores", value=False)
    
#     st.divider()
#     st.caption("GraphRAG v1.0 | Local Llama 3.2 | Neo4j AuraDB")


# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     # Add welcome message
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": "👋 Hello! I'm a legal document assistant with temporal awareness. Ask me questions about the documents, and I'll provide answers based on the **current** versions (or historical ones if you specify a date).",
#         "sources": []
#     })

# if "engine" not in st.session_state:
#     with st.spinner("Loading models..."):
#         st.session_state.engine = QueryEngine()

# # Display chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
        
#         # Show sources if available
#         if show_sources and msg.get("sources"):
#             with st.expander("📎 Sources", expanded=False):
#                 for src in msg["sources"]:
#                     status_class = "current" if src.get("is_current", True) else "historical"
#                     status_emoji = "🟢" if src.get("is_current", True) else "🟡"
                    
#                     st.markdown(f"""
#                     <div class="source-box {status_class}">
#                         {status_emoji} <strong>{src['doc_id']}</strong> | Effective: {src['effective_date']}<br/>
#                         <small>Chunk: {src['chunk_id']}</small>
#                         {f"<br/><small>Score: {src['score']:.3f}</small>" if show_scores else ""}
#                     </div>
#                     """, unsafe_allow_html=True)

# # Chat input
# if prompt := st.chat_input("Ask about the documents..."):
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("🔍 Retrieving and analyzing..."):
#             result = st.session_state.engine.answer(
#                 prompt,
#                 target_date=str(target_date)
#             )
            
#             answer = result["answer"]
#             sources = result["sources"]
            
#             st.markdown(answer)
            
#             # Show sources
#             if show_sources and sources:
#                 with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
#                     for src in sources:
#                         status_emoji = "🟢" if src["is_current"] else "🟡"
#                         status_text = "CURRENT" if src["is_current"] else "HISTORICAL"
                        
#                         st.markdown(f"""
#                         **{status_emoji} {src['doc_id']}** ({status_text})  
#                         Effective: `{src['effective_date']}`  
#                         Chunk: `{src['chunk_id']}`
#                         {f"Relevance: `{src['score']:.3f}`" if show_scores else ""}
#                         """)
#                         st.divider()
    
#     # Save response
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": answer,
#         "sources": sources
#     })

# # Footer
# st.divider()
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("🔍 Retrieval", f"Hybrid (α={config.retrieval.hybrid_alpha})")
# with col2:
#     st.metric("🤖 Model", config.ollama.model)
# with col3:
#     st.metric("🔄 Graph", "Enabled" if st.session_state.engine.graph_enabled else "Disabled")



"""
Graph-Grounded Temporal RAG - Professional Legal Document Intelligence Platform
Production-ready Streamlit frontend with interactive features and dynamic UI.
"""

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import time
import json
from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_engine import QueryEngine
from src.config import config
from src.logger import get_logger
from src.ingest import ingest_single_document

logger = get_logger(__name__)

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="LexTemporal AI | Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/gere47/RAG',
        'Report a bug': 'https://github.com/gere47/RAG/issues',
        'About': '# Graph-Grounded Temporal RAG\n*Contradiction-Resilient Legal QA*'
    }
)

# ============================================================================
# Custom CSS for Professional Styling
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header Gradient */
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Chat Messages */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 12px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
        font-weight: 500;
        animation: slideIn 0.3s ease;
    }
    
    .assistant-message {
        background: #f8f9fa;
        color: #1e1e2f;
        padding: 20px 24px;
        border-radius: 20px 20px 20px 4px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        line-height: 1.7;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Source Cards */
    .source-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .source-card.historical {
        border-left-color: #ffa500;
        opacity: 0.85;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-current {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-historical {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-graph {
        background: #667eea;
        color: white;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e1e2f;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Styling */
    .sidebar .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .sidebar .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader */
    .upload-box {
        border: 2px dashed #dee2e6;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.2s ease;
    }
    
    .upload-box:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    /* Input Box */
    .stChatInput > div {
        border-radius: 30px !important;
        border: 2px solid #e9ecef !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Timeline Visualization */
    .timeline-container {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #e9ecef;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .assistant-message {
            background: #2d2d3f;
            color: #f0f0f0;
            border-color: #3d3d4f;
        }
        .source-card {
            background: #2d2d3f;
            border-color: #3d3d4f;
        }
        .metric-card {
            background: #2d2d3f;
            border-color: #3d3d4f;
        }
        .metric-value {
            color: #f0f0f0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    with st.spinner("🚀 Initializing LexTemporal AI..."):
        st.session_state.engine = QueryEngine()

if "manifest_df" not in st.session_state:
    if config.manifest_path.exists():
        st.session_state.manifest_df = pd.read_csv(config.manifest_path)
    else:
        st.session_state.manifest_df = pd.DataFrame()

if "show_timeline" not in st.session_state:
    st.session_state.show_timeline = False

if "upload_success" not in st.session_state:
    st.session_state.upload_success = False

# ============================================================================
# Helper Functions
# ============================================================================
def render_header():
    """Render the professional header with gradient."""
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col2:
        st.markdown("""
        <div class="header-gradient">
            <div class="header-title">⚖️ LexTemporal AI</div>
            <div class="header-subtitle">Graph-Grounded Temporal RAG for Legal Intelligence</div>
            <div style="margin-top: 15px; display: flex; gap: 10px;">
                <span class="badge badge-current">Contradiction-Resilient</span>
                <span class="badge badge-graph">Temporal-Aware</span>
                <span class="badge badge-graph">Hybrid Search</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render the professional sidebar with all controls."""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #1e1e2f; font-weight: 700; margin-bottom: 5px;">⚖️ LexTemporal</h1>
            <p style="color: #6c757d; font-size: 0.9rem;">Legal Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # System Status Section
        st.markdown("### 📊 System Status")
        
        engine = st.session_state.engine
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "📄 Documents",
                len(st.session_state.manifest_df),
                delta=None
            )
        with col2:
            st.metric(
                "🔍 Chunks",
                engine.collection.count(),
                delta=None
            )
        
        col3, col4 = st.columns(2)
        with col3:
            graph_status = "🟢 Online" if engine.graph_enabled else "🟡 Limited"
            st.metric("🕸️ Graph", graph_status)
        with col4:
            st.metric("🤖 LLM", "Llama 3.2")
        
        st.divider()
        
        # Temporal Controls
        st.markdown("### ⏳ Temporal Context")
        
        use_specific_date = st.checkbox("Use specific effective date", value=False)
        
        if use_specific_date:
            target_date = st.date_input(
                "Effective Date",
                value=date(2026, 4, 12),
                help="The system will prioritize documents effective on or before this date"
            )
            st.session_state.target_date = str(target_date)
        else:
            st.session_state.target_date = None
            st.info("Using current/latest documents")
        
        st.divider()
        
        # Document Timeline
        st.markdown("### 📅 Document Timeline")
        
        if st.button("📈 Show Evolution Timeline", use_container_width=True):
            st.session_state.show_timeline = not st.session_state.show_timeline
        
        if st.session_state.show_timeline and len(st.session_state.manifest_df) > 0:
            render_timeline()
        
        st.divider()
        
        # Upload Section
        st.markdown("### 📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Add new version",
            type=["pdf"],
            help="Upload a new version of a legal document",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            with st.expander("📝 Document Details", expanded=True):
                doc_title = st.text_input("Document Title", value=uploaded_file.name)
                effective_date = st.date_input("Effective Date", value=date.today())
                
                manifest = st.session_state.manifest_df
                supersedes_options = ["None"] + (manifest['doc_id'].tolist() if not manifest.empty else [])
                supersedes = st.selectbox("Supersedes Document", options=supersedes_options)
                
                if st.button("🚀 Process Document", use_container_width=True):
                    process_uploaded_document(
                        uploaded_file, 
                        doc_title, 
                        str(effective_date),
                        None if supersedes == "None" else supersedes
                    )
        
        if st.session_state.upload_success:
            st.success("✅ Document processed!")
            st.session_state.upload_success = False
            time.sleep(1)
            st.rerun()
        
        st.divider()
        
        # Display Options
        st.markdown("### 🎛️ Display Options")
        show_sources = st.checkbox("Show source citations", value=True)
        show_confidence = st.checkbox("Show confidence scores", value=False)
        
        st.divider()
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; color: #6c757d; font-size: 0.8rem;">
            Graph-Grounded Temporal RAG v1.0<br/>
            © 2026 LexTemporal AI
        </div>
        """, unsafe_allow_html=True)
        
        return show_sources, show_confidence


def render_timeline():
    """Render interactive document evolution timeline."""
    df = st.session_state.manifest_df.copy()
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df = df.sort_values('effective_date')
    
    fig = px.timeline(
        df,
        x_start="effective_date",
        x_end=df['effective_date'] + pd.Timedelta(days=30),
        y="doc_title",
        color="doc_id",
        title="Document Evolution Timeline",
        labels={"effective_date": "Effective Date", "doc_title": "Document"},
        height=300
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def process_uploaded_document(uploaded_file, title: str, effective_date: str, supersedes: Optional[str]):
    """Process and ingest an uploaded document."""
    import uuid
    from src.ingest import ingest_single_document
    
    with st.spinner("📄 Processing document..."):
        # Generate new ID
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        
        # Save PDF
        pdf_path = config.RAW_PDFS_DIR / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Update manifest
        new_row = pd.DataFrame([{
            'doc_id': doc_id,
            'doc_title': title,
            'effective_date': effective_date,
            'supersedes_doc_id': supersedes
        }])
        
        manifest = st.session_state.manifest_df
        manifest = pd.concat([manifest, new_row], ignore_index=True)
        manifest.to_csv(config.manifest_path, index=False)
        st.session_state.manifest_df = manifest
        
        # Ingest into system
        try:
            ingest_single_document(doc_id, effective_date)
            st.session_state.upload_success = True
        except Exception as e:
            st.error(f"Failed to ingest document: {e}")


def render_chat_message(role: str, content: str, sources: List[Dict] = None, show_sources: bool = True):
    """Render a beautifully styled chat message."""
    if role == "user":
        st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
        
        if show_sources and sources:
            with st.expander("📎 View Sources & Citations", expanded=False):
                for i, src in enumerate(sources, 1):
                    is_current = src.get('is_current', True)
                    card_class = "source-card" if is_current else "source-card historical"
                    badge_class = "badge-current" if is_current else "badge-historical"
                    badge_text = "CURRENT" if is_current else "HISTORICAL"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="font-size: 1rem;">📄 {src['doc_id']}</strong>
                            <span class="badge {badge_class}">{badge_text}</span>
                        </div>
                        <div style="margin-top: 8px; color: #6c757d; font-size: 0.9rem;">
                            Effective: {src['effective_date']} | Chunk: {src['chunk_id']}
                        </div>
                        <div style="margin-top: 8px; font-size: 0.9rem; opacity: 0.8;">
                            Relevance Score: {src.get('score', 0):.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def render_welcome_screen():
    """Render welcome screen with example queries."""
    st.markdown("""
    <div style="max-width: 800px; margin: 40px auto; text-align: center;">
        <h2 style="color: #1e1e2f; font-weight: 600; margin-bottom: 30px;">
            👋 Welcome to LexTemporal AI
        </h2>
        <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 40px;">
            Ask questions about your legal documents. I'll provide accurate, 
            contradiction-resilient answers based on the current effective versions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example queries
    st.markdown("#### 💡 Try these example queries:")
    
    cols = st.columns(3)
    examples = [
        "What is the effective date of the original agreement?",
        "What are the payment terms?",
        "What is the governing law?",
        "Who are the parties involved?",
        "What is the penalty fee?",
        "Show me the amendment history"
    ]
    
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


# ============================================================================
# Main Application
# ============================================================================
def main():
    """Main application entry point."""
    
    # Render header
    render_header()
    
    # Render sidebar and get display options
    show_sources, show_confidence = render_sidebar()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            render_welcome_screen()
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "user":
                        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
                        if show_sources and msg.get("sources"):
                            with st.expander("📎 View Sources", expanded=False):
                                for src in msg["sources"]:
                                    is_current = src.get('is_current', True)
                                    card_class = "source-card" if is_current else "source-card historical"
                                    badge_class = "badge-current" if is_current else "badge-historical"
                                    badge_text = "CURRENT" if is_current else "HISTORICAL"
                                    
                                    st.markdown(f"""
                                    <div class="{card_class}">
                                        <div style="display: flex; justify-content: space-between;">
                                            <strong>📄 {src['doc_id']}</strong>
                                            <span class="badge {badge_class}">{badge_text}</span>
                                        </div>
                                        <div style="margin-top: 8px; color: #6c757d;">
                                            Effective: {src['effective_date']} | Chunk: {src['chunk_id']}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about your legal documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing documents and resolving temporal context..."):
                # Progress animation
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                progress_bar.empty()
                
                # Get answer
                result = st.session_state.engine.answer(
                    prompt,
                    target_date=st.session_state.get('target_date')
                )
                
                answer = result["answer"]
                sources = result["sources"]
                graph_used = result["graph_used"]
                
                # Display answer
                st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
                
                # Display sources
                if show_sources and sources:
                    with st.expander("📎 View Sources & Citations", expanded=False):
                        # Graph status
                        if graph_used:
                            st.success("🕸️ Graph-based temporal resolution applied")
                        
                        for src in sources:
                            is_current = src.get('is_current', True)
                            card_class = "source-card" if is_current else "source-card historical"
                            badge_class = "badge-current" if is_current else "badge-historical"
                            badge_text = "CURRENT" if is_current else "HISTORICAL"
                            
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div style="display: flex; justify-content: space-between;">
                                    <strong>📄 {src['doc_id']}</strong>
                                    <span class="badge {badge_class}">{badge_text}</span>
                                </div>
                                <div style="margin-top: 8px; color: #6c757d;">
                                    Effective: {src['effective_date']} | Chunk: {src['chunk_id']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()