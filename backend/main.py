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
import glob
import PyPDF2
import docx
import pandas as pd
from io import BytesIO
import magic

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

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from various file formats"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.txt':
            return file_content.decode('utf-8')
        
        elif file_ext == '.pdf':
            text = ""
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif file_ext in ['.docx', '.doc']:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        # elif file_ext == '.csv':
        #     df = pd.read_csv(BytesIO(file_content))
        #     return df.to_string(index=False)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    except Exception as e:
        logging.error(f"Error extracting text from {filename}: {e}")
        raise


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
            "Multiple file format support (TXT, PDF, DOCX)",
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
        
        # Get source documents info - only return source file names
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"][:3]:  # Limit to top 3 sources
                source_file = doc.metadata.get('source_file', 'Unknown')
                # Only add if not already in the list (to avoid duplicates)
                if source_file not in sources:
                    sources.append(source_file)
        
        logging.info(f"Successfully answered question: {question.question}")
        return {
            "answer": result["result"],
            "sources": sources,  # Now just a list of file names
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
    return {
        "formats": [".txt", ".pdf", ".docx", ".doc"],
        "descriptions": {
            ".txt": "Plain text files",
            ".pdf": "PDF documents", 
            ".docx": "Microsoft Word documents (newer format)",
            ".doc": "Microsoft Word documents (older format)"        }
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file to the documents directory"""
    logging.info(f"Received file upload: {file.filename}")
    
    # Supported formats
    supported_formats = ['.txt', '.pdf', '.docx', '.doc']
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {supported_formats}"
        )
    
    try:
        # Create documents directory if it doesn't exist
        DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        
        # Read file content
        file_content = await file.read()
        
        # Save uploaded file
        file_path = os.path.join(DOCUMENTS_PATH, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Extract and save text content for processing
        try:
            text_content = extract_text_from_file(file_content, file.filename)
            text_file_path = os.path.join(DOCUMENTS_PATH, f"{os.path.splitext(file.filename)[0]}.txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text_content)
        except Exception as e:
            logging.warning(f"Could not extract text from {file.filename}: {e}")
        
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

@app.get("/documents")
def list_documents():
    """Get list of all documents in the documents directory"""
    try:
        if 'rag_handler' in globals():
            from llm.improved_rag_handler import DOCUMENTS_PATH
        else:
            DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
        
        # Create directory if it doesn't exist
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        
        # Get all files in the documents directory
        files = []
        for file_path in glob.glob(os.path.join(DOCUMENTS_PATH, "*")):
            if os.path.isfile(file_path):
                file_info = {
                    "filename": os.path.basename(file_path),
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    "extension": os.path.splitext(file_path)[1].lower()
                }
                files.append(file_info)
        
        # Sort files by name
        files.sort(key=lambda x: x["filename"])
        
        return {
            "documents": files,
            "count": len(files),
            "path": DOCUMENTS_PATH
        }
        
    except Exception as e:
        logging.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting enhanced backend server.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)