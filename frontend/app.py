import streamlit as st
from ui_components import (
    check_system_status,
    handle_file_management,
    handle_database_operations,
    handle_question,
    handle_document_search,
    display_system_metrics,
    display_features,
)
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Enhanced RAG Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(""" 
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .status-box { padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .status-ready { background-color: #d4edda; border-left: 4px solid #28a745; }
    .status-progress { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .status-error { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    .feature-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff; margin: 0.5rem 0; }
    .metric-card { background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; border-left: 4px solid #007bff; background-color: #f8f9fa; }
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
        check_system_status(BACKEND_URL)
        
        st.markdown("---")
        
        # File Management
        st.subheader("📁 Document Management")
        handle_file_management(BACKEND_URL)
        
        st.markdown("---")
        
        # Database Operations
        st.subheader("🗄️ Database Operations")
        handle_database_operations(BACKEND_URL)
        
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
                        for source_file in sources:
                            st.markdown(f"- {source_file}")
                    st.markdown(f"*{timestamp}*")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Question input
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area("Ask your question:", height=100, placeholder="Enter your question here...")
            submitted = st.form_submit_button("🚀 Ask", use_container_width=True)
            
            if submitted and question:
                handle_question(BACKEND_URL, question, temperature, max_results)
    
    with col2:
        st.header("🔍 Quick Actions")
        
        # Search functionality
        with st.expander("🔍 Document Search", expanded=False):
            search_query = st.text_input("Search documents:", placeholder="Enter search terms...")
            if st.button("Search Documents") and search_query:
                handle_document_search(BACKEND_URL, search_query, max_results)
        
        # System metrics
        with st.expander("📈 System Metrics", expanded=True):
            display_system_metrics()
        
        # Feature list
        with st.expander("✨ Features", expanded=False):
            display_features()


if __name__ == "__main__":
    main()
