import sys
import os
import threading
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from llm.improved_rag_handler import rag_handler
except ImportError:
    # Fallback to original handler
    from llm.rag_handler import get_qa_chain, create_vector_db

app = FastAPI(
    title="Enhanced RAG API",
    description="A powerful RAG system supporting multiple document formats",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Question(BaseModel):
    question: str
    temperature: Optional[float] = 0.1
    max_results: Optional[int] = 5

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class DatabaseConfig(BaseModel):
    force_recreate: Optional[bool] = False

# Global variables
db_creation_status = {
    "in_progress": False, 
    "completed": False, 
    "error": None,
    "progress": 0,
    "message": "Ready",
    "files_processed": 0,
    "total_files": 0,
    "start_time": None,
    "end_time": None
}

def update_progress(progress: int, message: str, files_processed: int = 0, total_files: int = 0):
    """Update the global progress status"""
    global db_creation_status
    db_creation_status.update({
        "progress": progress,
        "message": message,
        "files_processed": files_processed,
        "total_files": total_files
    })
    logging.info(f"Progress: {progress}% - {message}")

def create_db_background(force_recreate: bool = False):
    """Run database creation in background thread with progress updates"""
    global db_creation_status
    
    logging.info("Starting database creation in background.")
    db_creation_status.update({
        "in_progress": True, 
        "completed": False, 
        "error": None,
        "progress": 0,
        "start_time": datetime.now().isoformat(),
        "end_time": None
    })
    
    try:
        update_progress(10, "Initializing...")
        
        # Use improved handler if available
        if 'rag_handler' in globals():
            update_progress(20, "Scanning documents...")
            result = rag_handler.create_vector_db(force_recreate)
            
            update_progress(60, "Creating embeddings...")
            time.sleep(1)  # Simulate processing time
            
            update_progress(90, "Finalizing database...")
            time.sleep(1)
            
            if result["success"]:
                db_creation_status.update({
                    "in_progress": False, 
                    "completed": True, 
                    "error": None,
                    "progress": 100,
                    "message": result["message"],
                    "files_processed": result["files_processed"],
                    "end_time": datetime.now().isoformat()
                })
                logging.info("Database creation completed successfully with improved handler.")
            else:
                raise Exception(result["message"])
        else:
            # Fallback to original handler
            update_progress(30, "Using legacy document loader...")
            create_vector_db()
            update_progress(100, "Database created successfully")
            
            db_creation_status.update({
                "in_progress": False, 
                "completed": True, 
                "error": None,
                "progress": 100,
                "message": "Database created successfully (legacy mode)",
                "end_time": datetime.now().isoformat()
            })
            logging.info("Database creation completed successfully with legacy handler.")
            
    except Exception as e:
        error_msg = f"Database creation failed: {str(e)}"
        db_creation_status.update({
            "in_progress": False, 
            "completed": False, 
            "error": error_msg,
            "progress": 0,
            "message": error_msg,
            "end_time": datetime.now().isoformat()
        })
        logging.error(f"Database creation failed: {e}", exc_info=True)

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "Enhanced RAG API", 
        "version": "2.0.0",
        "features": [
            "Multiple file format support (TXT, PDF, DOCX, CSV)",
            "Parallel document processing",
            "Incremental updates",
            "Document search",
            "Progress tracking"
        ]
    }

@app.post("/ask")
def ask(question: Question):
    """Ask a question and get an AI-generated answer"""
    logging.info(f"Received question: {question.question}")
    
    if db_creation_status["in_progress"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Database creation in progress ({db_creation_status['progress']}%). Please wait."
        )
   
    if not db_creation_status["completed"]:
        raise HTTPException(
            status_code=400, 
            detail="Vector database not ready. Please create it first."
        )
   
    try:
        logging.info("Getting QA chain.")
        if 'rag_handler' in globals():
            qa_chain = rag_handler.get_qa_chain(temperature=question.temperature)
        else:
            qa_chain = get_qa_chain()
            
        if not qa_chain:
            raise HTTPException(
                status_code=400, 
                detail="QA chain could not be initialized. Please recreate the database."
            )
       
        logging.info("Processing question with QA chain.")
        result = qa_chain({"query": question.question})
        
        # Get source documents info
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"][:3]:  # Limit to top 3 sources
                sources.append({
                    "source": doc.metadata.get('source_file', 'Unknown'),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        logging.info(f"Successfully answered question: {question.question}")
        return {
            "answer": result["result"],
            "sources": sources,
            "question": question.question
        }
        
    except Exception as e:
        logging.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/search")
def search_documents(search_req: SearchRequest):
    """Search for relevant documents without generating an answer"""
    logging.info(f"Received search query: {search_req.query}")
    
    if not db_creation_status["completed"]:
        raise HTTPException(
            status_code=400, 
            detail="Vector database not ready. Please create it first."
        )
    
    try:
        if 'rag_handler' in globals():
            results = rag_handler.search_documents(search_req.query, search_req.max_results)
            return {"results": results, "query": search_req.query}
        else:
            raise HTTPException(
                status_code=501, 
                detail="Search functionality not available in legacy mode."
            )
    except Exception as e:
        logging.error(f"Error searching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.post("/create-db")
def create_db(config: DatabaseConfig = DatabaseConfig()):
    """Trigger database creation"""
    logging.info(f"Received request to create database. Force recreate: {config.force_recreate}")
    
    if db_creation_status["in_progress"]:
        return {
            "message": f"Database creation already in progress ({db_creation_status['progress']}%).",
            "status": db_creation_status
        }
    
    if db_creation_status["completed"] and not config.force_recreate:
        return {
            "message": "Database already exists. Use force_recreate=true to rebuild.",
            "status": db_creation_status
        }
    
    # Start database creation in background thread
    logging.info("Starting database creation thread.")
    thread = threading.Thread(target=create_db_background, args=(config.force_recreate,))
    thread.daemon = True
    thread.start()
    
    return {"message": "Vector database creation process started.", "status": db_creation_status}

@app.get("/db-status")
def get_db_status():
    """Get detailed database status"""
    logging.info("Request for database status.")
    
    status = db_creation_status.copy()
    
    # Add additional stats if available
    if 'rag_handler' in globals() and db_creation_status["completed"]:
        try:
            stats = rag_handler.get_database_stats()
            status.update({"database_stats": stats})
        except Exception as e:
            logging.warning(f"Could not get database stats: {e}")
    
    return status

@app.get("/supported-formats")
def get_supported_formats():
    """Get list of supported document formats"""
    if 'rag_handler' in globals():
        from llm.improved_rag_handler import FILE_LOADERS
        return {
            "formats": list(FILE_LOADERS.keys()),
            "descriptions": {
                ".txt": "Plain text files",
                ".pdf": "PDF documents", 
                ".docx": "Microsoft Word documents (newer format)",
                ".doc": "Microsoft Word documents (older format)",
                ".csv": "Comma-separated values"
            }
        }
    else:
        return {
            "formats": [".txt"],
            "descriptions": {".txt": "Plain text files (legacy mode)"}
        }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file to the documents directory"""
    logging.info(f"Received file upload: {file.filename}")
    
    # Get supported formats
    if 'rag_handler' in globals():
        from llm.improved_rag_handler import FILE_LOADERS, DOCUMENTS_PATH
        supported_formats = list(FILE_LOADERS.keys())
    else:
        supported_formats = [".txt"]
        DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {supported_formats}"
        )
    
    try:
        # Create documents directory if it doesn't exist
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(DOCUMENTS_PATH, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved to: {file_path}")
        return {
            "message": f"File '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "format": file_ext
        }
        
    except Exception as e:
        logging.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.delete("/clear-db")
def clear_database():
    """Clear the vector database"""
    global db_creation_status
    logging.info("Request to clear database.")
    
    if db_creation_status["in_progress"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot clear database while creation is in progress."
        )
    
    try:
        if 'rag_handler' in globals():
            from llm.improved_rag_handler import PERSIST_DIRECTORY
        else:
            PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "db")
        
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            logging.info("Database cleared successfully.")
        
        # Reset status
        db_creation_status = {
            "in_progress": False,
            "completed": False,
            "error": None,
            "progress": 0,
            "message": "Database cleared",
            "files_processed": 0,
            "total_files": 0,
            "start_time": None,
            "end_time": None
        }
        
        return {"message": "Database cleared successfully."}
        
    except Exception as e:
        logging.error(f"Error clearing database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting enhanced backend server.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)