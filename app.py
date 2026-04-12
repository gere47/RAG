"""
Streamlit web interface for Graph-Grounded Temporal RAG.
Usage: streamlit run app.py
"""

import streamlit as st
from datetime import date
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_engine import QueryEngine
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Legal Graph RAG",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    .source-box { 
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 5px; 
        font-size: 0.9em;
        margin: 5px 0;
    }
    .current { border-left: 4px solid #00cc66; }
    .historical { border-left: 4px solid #ff9900; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📜 Graph‑Grounded Temporal RAG")
st.markdown("*Contradiction‑Resilient Question Answering over Evolving Legal Documents*")

# In the sidebar, before the settings section
with st.sidebar:
    st.header("📤 Upload New Document")
    
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a new version of a legal document"
    )
    
    if uploaded_file:
        doc_title = st.text_input("Document Title", value=uploaded_file.name)
        effective_date = st.date_input("Effective Date", value=date.today())
        
        # Find which document this supersedes
        import pandas as pd
        manifest = pd.read_csv(config.manifest_path)
        supersedes_options = ["None"] + manifest['doc_id'].tolist()
        supersedes = st.selectbox("Supersedes Document", options=supersedes_options)
        
        if st.button("Process Document"):
            with st.spinner("Processing..."):
                # Save PDF
                import uuid
                new_id = f"doc_{str(uuid.uuid4())[:8]}"
                pdf_path = config.RAW_PDFS_DIR / f"{new_id}.pdf"
                
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Update manifest
                new_row = pd.DataFrame([{
                    'doc_id': new_id,
                    'doc_title': doc_title,
                    'effective_date': str(effective_date),
                    'supersedes_doc_id': None if supersedes == "None" else supersedes
                }])
                manifest = pd.concat([manifest, new_row], ignore_index=True)
                manifest.to_csv(config.manifest_path, index=False)
                
                # Trigger pipeline for this single document
                from src.ingest import ingest_single_document
                ingest_single_document(new_id, str(effective_date))
                
                st.success(f"✅ Document {new_id} processed!")
                st.rerun()
                
# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Date selector
    target_date = st.date_input(
        "📅 Effective Date (for temporal queries)",
        value=date(2026, 4, 8),
        help="The system will treat documents effective after this date as future/inactive"
    )
    
    st.divider()
    
    # System status
    st.header("📊 System Status")
    
    # Check components
    col1, col2 = st.columns(2)
    
    # Vector DB
    vectors_path = config.VECTORS_DIR
    if vectors_path.exists() and list(vectors_path.iterdir()):
        col1.success("✅ Vector DB")
    else:
        col1.warning("⚠️ No vectors")
    
    # Graph DB
    engine = QueryEngine()
    # Graph DB
    if "engine" not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.engine = QueryEngine()

    if st.session_state.engine.graph_enabled:
        col2.success("✅ Neo4j Graph")
    else:
        col2.warning("⚠️ Graph disabled")
    
    # Document count
    manifest_path = config.manifest_path
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        st.metric("📄 Documents", len(df))
    
    st.divider()
    
    # Display options
    st.header("🔍 Display Options")
    show_sources = st.checkbox("Show source documents", value=True)
    show_graph_path = st.checkbox("Show graph traversal path", value=False)
    show_scores = st.checkbox("Show relevance scores", value=False)
    
    st.divider()
    st.caption("GraphRAG v1.0 | Local Llama 3.2 | Neo4j AuraDB")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm a legal document assistant with temporal awareness. Ask me questions about the documents, and I'll provide answers based on the **current** versions (or historical ones if you specify a date).",
        "sources": []
    })

if "engine" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.engine = QueryEngine()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show sources if available
        if show_sources and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    status_class = "current" if src.get("is_current", True) else "historical"
                    status_emoji = "🟢" if src.get("is_current", True) else "🟡"
                    
                    st.markdown(f"""
                    <div class="source-box {status_class}">
                        {status_emoji} <strong>{src['doc_id']}</strong> | Effective: {src['effective_date']}<br/>
                        <small>Chunk: {src['chunk_id']}</small>
                        {f"<br/><small>Score: {src['score']:.3f}</small>" if show_scores else ""}
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about the documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and analyzing..."):
            result = st.session_state.engine.answer(
                prompt,
                target_date=str(target_date)
            )
            
            answer = result["answer"]
            sources = result["sources"]
            
            st.markdown(answer)
            
            # Show sources
            if show_sources and sources:
                with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                    for src in sources:
                        status_emoji = "🟢" if src["is_current"] else "🟡"
                        status_text = "CURRENT" if src["is_current"] else "HISTORICAL"
                        
                        st.markdown(f"""
                        **{status_emoji} {src['doc_id']}** ({status_text})  
                        Effective: `{src['effective_date']}`  
                        Chunk: `{src['chunk_id']}`
                        {f"Relevance: `{src['score']:.3f}`" if show_scores else ""}
                        """)
                        st.divider()
    
    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔍 Retrieval", f"Hybrid (α={config.retrieval.hybrid_alpha})")
with col2:
    st.metric("🤖 Model", config.ollama.model)
with col3:
    st.metric("🔄 Graph", "Enabled" if st.session_state.engine.graph_enabled else "Disabled")