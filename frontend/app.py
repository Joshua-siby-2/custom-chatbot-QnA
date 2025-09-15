import streamlit as st
import requests
import time
import logging
import json
from datetime import datetime
import plotly.express as px
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Enhanced RAG Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-ready {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-progress {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Enhanced RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_status_check' not in st.session_state:
        st.session_state.last_status_check = 0
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # System Status
        st.subheader("📊 System Status")
        check_system_status()
        
        st.markdown("---")
        
        # File Management
        st.subheader("📁 Document Management")
        handle_file_management()
        
        st.markdown("---")
        
        # Database Operations
        st.subheader("🗄️ Database Operations")
        handle_database_operations()
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        temperature = st.slider("Response Temperature", 0.0, 1.0, 0.1, 0.1)
        max_results = st.slider("Max Search Results", 1, 10, 5)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        if st.session_state.chat_history:
            for i, (question, answer, sources, timestamp) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f'<div class="chat-message">', unsafe_allow_html=True)
                    st.markdown(f"**You:** {question}")
                    st.markdown(f"**Assistant:** {answer}")
                    if sources:
                        st.markdown("**Sources:**")
                        for j, source in enumerate(sources):
                            st.markdown(f"- {source['source']}: {source['content_preview'][:100]}...")
                    st.markdown(f"*{timestamp}*")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Question input
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area("Ask your question:", height=100, placeholder="Enter your question here...")
            submitted = st.form_submit_button("🚀 Ask", use_container_width=True)
            
            if submitted and question:
                handle_question(question, temperature, max_results)
    
    with col2:
        st.header("🔍 Quick Actions")
        
        # Search functionality
        with st.expander("🔍 Document Search", expanded=False):
            search_query = st.text_input("Search documents:", placeholder="Enter search terms...")
            if st.button("Search Documents") and search_query:
                handle_document_search(search_query, max_results)
        
        # System metrics
        with st.expander("📈 System Metrics", expanded=True):
            display_system_metrics()
        
        # Feature list
        with st.expander("✨ Features", expanded=False):
            display_features()

def check_system_status():
    """Check and display system status"""
    current_time = time.time()
    
    # Check status every 5 seconds to avoid too frequent requests
    if current_time - st.session_state.last_status_check > 5:
        try:
            response = requests.get(f"{BACKEND_URL}/db-status", timeout=5)
            if response.status_code == 200:
                st.session_state.db_status = response.json()
                st.session_state.backend_connected = True
            else:
                st.session_state.backend_connected = False
        except Exception as e:
            st.session_state.backend_connected = False
            logging.error(f"Error checking status: {e}")
        
        st.session_state.last_status_check = current_time
    
    # Display status
    if hasattr(st.session_state, 'backend_connected') and st.session_state.backend_connected:
        if hasattr(st.session_state, 'db_status'):
            db_status = st.session_state.db_status
            
            if db_status.get("in_progress"):
                st.markdown(f'<div class="status-box status-progress">', unsafe_allow_html=True)
                st.markdown(f"🔄 **Creating Database**")
                st.progress(db_status.get("progress", 0) / 100)
                st.markdown(f"Progress: {db_status.get('progress', 0)}%")
                st.markdown(f"Status: {db_status.get('message', 'Processing...')}")
                if db_status.get("files_processed", 0) > 0:
                    st.markdown(f"Files processed: {db_status['files_processed']}")
                st.markdown('</div>', unsafe_allow_html=True)
            elif db_status.get("completed"):
                st.markdown(f'<div class="status-box status-ready">', unsafe_allow_html=True)
                st.markdown("✅ **Database Ready**")
                st.markdown(f"Files processed: {db_status.get('files_processed', 0)}")
                if db_status.get('end_time'):
                    st.markdown(f"Last updated: {db_status['end_time'][:19]}")
                st.markdown('</div>', unsafe_allow_html=True)
            elif db_status.get("error"):
                st.markdown(f'<div class="status-box status-error">', unsafe_allow_html=True)
                st.markdown("❌ **Database Error**")
                st.markdown(f"Error: {db_status['error']}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box">', unsafe_allow_html=True)
                st.markdown("⏳ **Database Not Ready**")
                st.markdown("Please create the database first")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("🟢 Backend Connected")
    else:
        st.error("🔴 Backend Disconnected")

def handle_file_management():
    """Handle file upload and management"""
    # Get supported formats
    try:
        response = requests.get(f"{BACKEND_URL}/supported-formats", timeout=5)
        if response.status_code == 200:
            formats_data = response.json()
            supported_formats = formats_data.get("formats", [".txt"])
            st.info(f"Supported formats: {', '.join(supported_formats)}")
        else:
            supported_formats = [".txt"]
    except:
        supported_formats = [".txt"]
        st.warning("Could not retrieve supported formats")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[fmt[1:] for fmt in supported_formats],  # Remove the dot
        help=f"Supported formats: {', '.join(supported_formats)}"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload File"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                    st.info(f"File size: {result['size']} bytes")
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Upload error: {str(e)}")

def handle_database_operations():
    """Handle database creation and management"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔨 Create Database"):
            create_database(force_recreate=False)
    
    with col2:
        if st.button("🔄 Rebuild Database"):
            create_database(force_recreate=True)
    
    # Clear database
    if st.button("🗑️ Clear Database", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            try:
                response = requests.delete(f"{BACKEND_URL}/clear-db", timeout=10)
                if response.status_code == 200:
                    st.success("Database cleared successfully!")
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.error("Failed to clear database")
            except Exception as e:
                st.error(f"Error clearing database: {e}")
        else:
            st.session_state.confirm_clear = True
            st.warning("⚠️ Click again to confirm deletion")
            st.rerun()

def create_database(force_recreate=False):
    """Create or recreate the database"""
    try:
        payload = {"force_recreate": force_recreate}
        response = requests.post(f"{BACKEND_URL}/create-db", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            st.success(result["message"])
            # Auto-refresh to show progress
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Database creation failed: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error creating database: {e}")

def handle_question(question, temperature, max_results):
    """Handle user question"""
    if not hasattr(st.session_state, 'db_status') or not st.session_state.db_status.get("completed"):
        st.error("Database not ready. Please create the database first.")
        return
    
    with st.spinner("🤔 Thinking..."):
        try:
            payload = {
                "question": question,
                "temperature": temperature,
                "max_results": max_results
            }
            response = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add to chat history
                st.session_state.chat_history.append((
                    question,
                    result["answer"],
                    result.get("sources", []),
                    timestamp
                ))
                
                st.rerun()
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Connection error: {e}")

def handle_document_search(query, max_results):
    """Handle document search"""
    try:
        payload = {"query": query, "max_results": max_results}
        response = requests.post(f"{BACKEND_URL}/search", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            results = result.get("results", [])
            
            if results:
                st.success(f"Found {len(results)} relevant documents:")
                for i, doc in enumerate(results):
                    st.markdown(f"**{i+1}. {doc['source']} ({doc['file_type']})**")
                    st.markdown(f"*{doc['content'][:200]}...*")
                    st.markdown("---")
            else:
                st.warning("No relevant documents found.")
        else:
            st.error("Search failed")
    except Exception as e:
        st.error(f"Search error: {e}")

def display_system_metrics():
    """Display system metrics"""
    if hasattr(st.session_state, 'db_status') and st.session_state.db_status:
        db_status = st.session_state.db_status
        
        # Create metrics
        col1, col2 = st.columns(2)
        
        with col1:
            files_count = db_status.get('files_processed', 0)
            st.metric("Files Processed", files_count)
        
        with col2:
            progress = db_status.get('progress', 0 if not db_status.get('completed') else 100)
            st.metric("Progress", f"{progress}%")
        
        # Database stats
        if db_status.get('database_stats'):
            stats = db_status['database_stats']
            if stats.get('total_chunks'):
                st.metric("Document Chunks", stats['total_chunks'])
        
        # Chat history metrics
        st.metric("Chat Messages", len(st.session_state.chat_history))

def display_features():
    """Display feature list"""
    features = [
        "🔄 **Multi-format Support**: PDF, DOCX, TXT, CSV",
        "⚡ **Fast Processing**: Parallel document loading",
        "🔍 **Smart Search**: Semantic document search",
        "💾 **Incremental Updates**: Only process new/changed files",
        "📊 **Progress Tracking**: Real-time status updates",
        "🔧 **Flexible Configuration**: Adjustable parameters",
        "💬 **Chat History**: Persistent conversation memory",
        "📤 **File Upload**: Direct file upload interface"
    ]
    
    for feature in features:
        st.markdown(f'<div class="feature-card">{feature}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()