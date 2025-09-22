import streamlit as st
import requests
import time
import logging
import pandas as pd
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_system_status(BACKEND_URL: str):
    """Check and display system status"""
    current_time = time.time()
    
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


def handle_file_management(BACKEND_URL: str):
    """Handle file upload and management"""
    # Get supported formats
    try:
        response = requests.get(f"{BACKEND_URL}/supported-formats", timeout=5)
        if response.status_code == 200:
            formats_data = response.json()
            supported_formats = formats_data.get("formats", [".txt", ".pdf", ".docx", ".doc", ".csv"])
            format_descriptions = formats_data.get("descriptions", {})
            
            with st.expander("📋 Supported Formats Info"):
                for fmt, desc in format_descriptions.items():
                    st.markdown(f"**{fmt}**: {desc}")
        else:
            supported_formats = [".txt", ".pdf", ".docx", ".doc", ".csv"]
            st.warning("Could not retrieve supported formats from server")
    except Exception as e:
        supported_formats = [".txt", ".pdf", ".docx", ".doc", ".csv"]
        st.warning(f"Could not connect to server for format info: {e}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=[fmt[1:] for fmt in supported_formats],
        help=f"Supported formats: {', '.join(supported_formats)}"
    )
    
    if uploaded_file is not None:
        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            st.info(f"File: {uploaded_file.name} ({format_file_size(uploaded_file.size)})")
            
            if st.button("📤 Upload File"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                    st.info(f"File size: {result['size']} bytes, Format: {result['format']}")
                    if 'documents_data' in st.session_state:
                        del st.session_state.documents_data
                    st.rerun()
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Upload failed: {error_detail}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Document list
    st.markdown("---")
    st.subheader("📄 Document List")
    
    if 'documents_data' not in st.session_state or st.button("🔄 Refresh List"):
        try:
            response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
            if response.status_code == 200:
                st.session_state.documents_data = response.json()
            else:
                st.error("Failed to load documents")
                return
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return
    
    documents_data = st.session_state.get('documents_data', {})
    documents = documents_data.get('documents', [])
    
    if documents:
        st.info(f"Found {len(documents)} documents")
        
        # Group by extension
        doc_types = {}
        for doc in documents:
            ext = doc['extension']
            doc_types.setdefault(ext, []).append(doc)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("TXT", len(doc_types.get('.txt', [])))
        with col2: st.metric("PDF", len(doc_types.get('.pdf', [])))
        with col3: st.metric("DOCX", len(doc_types.get('.docx', [])))
        with col4: st.metric("DOC", len(doc_types.get('.doc', [])))
        with col5: st.metric("CSV", len(doc_types.get('.csv', [])))
        
        st.subheader("All Documents")
        for doc in documents:
            col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
            with col1: st.write(f"**{doc['filename']}**")
            with col2: st.write(f"{format_file_size(doc['size'])}")
            with col3: st.write(f"{doc['extension']}")
            with col4:
                if st.button("🗑️", key=f"delete_{doc['filename']}"):
                    if st.session_state.get(f"confirm_delete_{doc['filename']}", False):
                        delete_document(BACKEND_URL, doc['filename'])
                        st.session_state[f"confirm_delete_{doc['filename']}"] = False
                    else:
                        st.session_state[f"confirm_delete_{doc['filename']}"] = True
                        st.warning(f"⚠️ Click again to confirm deletion of {doc['filename']}")
                        st.rerun()
        
        for doc in documents:
            if st.session_state.get(f"confirm_delete_{doc['filename']}", False):
                st.warning(f"⚠️ Click the trash icon again to confirm deletion of {doc['filename']}")
                with st.expander("📊 Detailed View"):
                    df_data = [{
                        "Filename": d['filename'],
                        "Size": format_file_size(d['size']),
                        "Type": d['extension'],
                        "Modified": d['modified'][:10]
                    } for d in documents]
                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No documents found. Upload files to get started.")


def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"


def delete_document(BACKEND_URL: str, filename: str):
    """Delete a document"""
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{filename}")
        if response.status_code == 200:
            st.success(f"✅ {response.json().get('message', 'Document deleted successfully')}")
            if 'documents_data' in st.session_state:
                del st.session_state.documents_data
            st.rerun()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"❌ Failed to delete file: {error_detail}")
    except Exception as e:
        st.error(f"❌ Error deleting file: {e}")


def handle_database_operations(BACKEND_URL: str):
    """Handle database creation and management"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔨 Create Database"):
            create_database(BACKEND_URL, force_recreate=False)
    
    with col2:
        if st.button("🔄 Rebuild Database"):
            create_database(BACKEND_URL, force_recreate=True)
    
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


def create_database(BACKEND_URL: str, force_recreate=False):
    """Create or recreate the database"""
    try:
        payload = {"force_recreate": force_recreate}
        response = requests.post(f"{BACKEND_URL}/create-db", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            st.success(result["message"])
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Database creation failed: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error creating database: {e}")


def handle_question(BACKEND_URL: str, question: str, temperature: float, max_results: int):
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


def handle_document_search(BACKEND_URL: str, query: str, max_results: int):
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files Processed", db_status.get('files_processed', 0))
        with col2:
            progress = db_status.get('progress', 0 if not db_status.get('completed') else 100)
            st.metric("Progress", f"{progress}%")
        
        if db_status.get('database_stats'):
            stats = db_status['database_stats']
            if stats.get('total_chunks'):
                st.metric("Document Chunks", stats['total_chunks'])
        
        st.metric("Chat Messages", len(st.session_state.chat_history))


def display_features():
    """Display feature list"""
    features = [
        "🔄 **Multi-format Support**: PDF, DOCX, TXT",
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
