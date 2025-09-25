import os
import logging
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, 
    UnstructuredWordDocumentLoader, CSVLoader
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "documents")
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "db")
METADATA_FILE = os.path.join(PERSIST_DIRECTORY, "metadata.json")

# Supported file types and their loaders
FILE_LOADERS = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.doc': UnstructuredWordDocumentLoader,
    '.csv': CSVLoader,
}

class ImprovedRAGHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("--- Initializing ImprovedRAGHandler ---")
        self.embeddings = None
        self.vectordb = None
        self.qa_chain = None
        self.llm = None
        self.retriever = None
        self._initialize_embeddings()
        self._initialize_llm()
        self.logger.info("--- ImprovedRAGHandler initialized ---")
    
    def _initialize_embeddings(self):
        """Initialize embeddings model once"""
        self.logger.info("Step A: Initializing embeddings model...")
        start_time = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {device} for embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
        self.logger.info(f"Step A finished in {time.time() - start_time:.2f} seconds.")
    
    def _initialize_llm(self):
        """Initialize LLM once"""
        self.logger.info("Step B: Initializing LLM...")
        start_time = time.time()
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL, 
            model=OLLAMA_MODEL,
            temperature=0.1,
        )
        self.logger.info(f"Step B finished in {time.time() - start_time:.2f} seconds.")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for a file to track changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about processed files"""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {"files": {}, "version": "1.0"}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata about processed files"""
        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_supported_files(self, directory: str) -> List[str]:
        """Get all supported files from directory"""
        supported_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                if file_ext in FILE_LOADERS:
                    supported_files.append(file_path)
        return supported_files
    
    def _load_document(self, file_path: str) -> List[Document]:
        """Load a single document with appropriate loader"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in FILE_LOADERS:
            logging.warning(f"Unsupported file type: {file_ext}")
            return []
        
        try:
            loader_class = FILE_LOADERS[file_ext]
            
            if file_ext == '.csv':
                loader = loader_class(file_path)
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_type': file_ext,
                    'full_path': file_path
                })
            
            logging.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def _load_documents_parallel(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents in parallel"""
        all_documents = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(self._load_document, fp): fp for fp in file_paths}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as exc:
                    logging.error(f"Error processing {file_path}: {exc}")
        
        return all_documents
    
    def create_vector_db(self, force_recreate: bool = False) -> Dict[str, Any]:
        """
        Creates a ChromaDB vector store from documents in the DOCUMENTS_PATH.
        Returns status information.
        """
        logging.info(f"Starting vector database creation...")
        
        if not os.path.exists(DOCUMENTS_PATH):
            os.makedirs(DOCUMENTS_PATH)
            logging.info(f"Created documents directory at {DOCUMENTS_PATH}")
            return {
                "success": False, 
                "message": "Please add some documents to the documents directory and try again.",
                "files_processed": 0
            }
        
        supported_files = self._get_supported_files(DOCUMENTS_PATH)
        logging.info(f"Found {len(supported_files)} supported files")
        
        if not supported_files:
            return {
                "success": False, 
                "message": f"No supported files found. Supported formats: {list(FILE_LOADERS.keys())}",
                "files_processed": 0
            }
        
        metadata = self._load_metadata()
        
        files_to_process = []
        if force_recreate or not os.path.exists(PERSIST_DIRECTORY):
            files_to_process = supported_files
            logging.info("Full database recreation requested or database doesn't exist")
        else:
            for file_path in supported_files:
                file_hash = self._get_file_hash(file_path)
                relative_path = os.path.relpath(file_path, DOCUMENTS_PATH)
                
                if (relative_path not in metadata["files"] or 
                    metadata["files"][relative_path]["hash"] != file_hash):
                    files_to_process.append(file_path)
        
        if not files_to_process and not force_recreate:
            logging.info("No new or modified files found. Database is up to date.")
            return {
                "success": True, 
                "message": "Database is already up to date.",
                "files_processed": 0
            }
        
        try:
            logging.info(f"Processing {len(files_to_process)} files...")
            
            documents = self._load_documents_parallel(files_to_process)
            
            if not documents:
                return {
                    "success": False, 
                    "message": "No documents could be loaded from the files.",
                    "files_processed": 0
                }
            
            logging.info(f"Loaded {len(documents)} total documents")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            logging.info(f"Split into {len(texts)} text chunks")
            
            if os.path.exists(PERSIST_DIRECTORY) and not force_recreate:
                self.vectordb = Chroma(
                    persist_directory=PERSIST_DIRECTORY, 
                    embedding_function=self.embeddings
                )
                self.vectordb.add_documents(texts)
                logging.info(f"Added {len(texts)} new chunks to existing database")
            else:
                if os.path.exists(PERSIST_DIRECTORY):
                    import shutil
                    shutil.rmtree(PERSIST_DIRECTORY)
                
                self.vectordb = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                logging.info(f"Created new database with {len(texts)} chunks")
            
            self.vectordb.persist()
            
            for file_path in files_to_process:
                relative_path = os.path.relpath(file_path, DOCUMENTS_PATH)
                metadata["files"][relative_path] = {
                    "hash": self._get_file_hash(file_path),
                    "processed_at": str(os.path.getmtime(file_path))
                }
            
            self._save_metadata(metadata)
            
            logging.info(f"Vector store created/updated successfully")
            return {
                "success": True, 
                "message": f"Successfully processed {len(files_to_process)} files with {len(texts)} chunks.",
                "files_processed": len(files_to_process),
                "total_chunks": len(texts),
                "supported_formats": list(FILE_LOADERS.keys())
            }
            
        except Exception as e:
            logging.error(f"Error creating vector database: {str(e)}", exc_info=True)
            return {
                "success": False, 
                "message": f"Error creating database: {str(e)}",
                "files_processed": 0
            }
    
    def get_qa_chain(self, temperature: float = 0.1):
        """
        Creates or retrieves a RetrievalQA chain for question answering.
        """
        self.logger.info("--- Getting QA chain ---")
        start_time = time.time()

        if not os.path.exists(PERSIST_DIRECTORY):
            self.logger.warning("Vector database not found.")
            return None
        
        try:
            if not self.vectordb:
                self.logger.info("Step C: Loading vector database...")
                db_load_start_time = time.time()
                self.vectordb = Chroma(
                    persist_directory=PERSIST_DIRECTORY, 
                    embedding_function=self.embeddings
                )
                self.logger.info(f"Step C finished in {time.time() - db_load_start_time:.2f} seconds.")
            
            if not self.retriever:
                self.logger.info("Step D: Initializing retriever...")
                retriever_start_time = time.time()
                self.retriever = self.vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
                )
                self.logger.info(f"Step D finished in {time.time() - retriever_start_time:.2f} seconds.")
            
            # Recreate the chain only if the temperature has changed
            if not self.qa_chain or self.llm.temperature != temperature:
                self.logger.info(f"Step E: Creating QA chain with temperature: {temperature}")
                chain_creation_start_time = time.time()
                self.llm.temperature = temperature
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True,
                    verbose=False
                )
                self.logger.info(f"Step E finished in {time.time() - chain_creation_start_time:.2f} seconds.")
            
            self.logger.info(f"--- QA chain ready in {time.time() - start_time:.2f} seconds ---")
            return self.qa_chain
            
        except Exception as e:
            self.logger.error(f"Error creating QA chain: {e}", exc_info=True)
            return None
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents without generating an answer"""
        if not self.vectordb:
            if not os.path.exists(PERSIST_DIRECTORY):
                return []
            self.vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY, 
                embedding_function=self.embeddings
            )
        
        try:
            docs = self.vectordb.similarity_search(query, k=k)
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "source": doc.metadata.get('source_file', 'Unknown'),
                    "file_type": doc.metadata.get('file_type', 'Unknown')
                })
            return results
        except Exception as e:
            logging.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the current database"""
        if not os.path.exists(PERSIST_DIRECTORY):
            return {"exists": False}
        
        try:
            metadata = self._load_metadata()
            stats = {
                "exists": True,
                "total_files": len(metadata.get("files", {})),
                "supported_formats": list(FILE_LOADERS.keys()),
                "files_by_type": {}
            }
            
            for file_info in metadata.get("files", {}).values():
                pass
            
            if self.vectordb:
                try:
                    collection = self.vectordb._collection
                    stats["total_chunks"] = collection.count()
                except:
                    stats["total_chunks"] = "Unknown"
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting database stats: {str(e)}")
            return {"exists": True, "error": str(e)}

# Global instance
rag_handler = ImprovedRAGHandler()

# Legacy functions for backward compatibility
def create_vector_db(force_recreate: bool = False):
    """Legacy function for backward compatibility"""
    result = rag_handler.create_vector_db(force_recreate)
    if not result["success"]:
        raise Exception(result["message"])

def get_qa_chain():
    """Legacy function for backward compatibility"""
    return rag_handler.get_qa_chain()

if __name__ == "__main__":
    # Test the improved system
    logging.info("Testing improved RAG system...")
    
    # Create the vector database
    result = rag_handler.create_vector_db()
    print(f"Database creation result: {result}")
    
    if result["success"]:
        # Test QA
        qa_chain = rag_handler.get_qa_chain()
        if qa_chain:
            test_question = "What is the main topic of the documents?"
            logging.info(f"Testing with question: {test_question}")
            result = qa_chain({"query": test_question})
            print(f"Question: {test_question}")
            print(f"Answer: {result['result']}")
            
            # Test document search
            search_results = rag_handler.search_documents(test_question)
            print(f"\nRelevant documents found: {len(search_results)}")
            for i, doc in enumerate(search_results):
                print(f"{i+1}. Source: {doc['source']} ({doc['file_type']})")
                print(f"   Content preview: {doc['content'][:100]}...")