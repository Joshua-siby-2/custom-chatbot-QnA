import sys
import os
import threading
import logging
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the RAG handler
try:
    logger.info("Attempting to import improved_rag_handler...")
    from llm.improved_rag_handler import rag_handler
    logger.info("Successfully imported improved_rag_handler")
except ImportError as e:
    logger.warning(f"Could not import improved_rag_handler: {str(e)}")
    try:
        logger.info("Falling back to original rag_handler...")
        from llm.rag_handler import get_qa_chain, create_vector_db
        logger.info("Successfully imported rag_handler")
    except ImportError as e:
        logger.error(f"Failed to import rag_handler: {str(e)}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Enhanced RAG API",
    description="A powerful RAG system supporting multiple document formats",
    version="2.0.0"
)

# Middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    
    try:
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Request completed in {process_time:.2f}ms - Status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")
        raise

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
    "end_time": None,
    "last_updated": None
}

def update_progress(progress: int, message: str, files_processed: int = 0, total_files: int = 0):
    """Update the global progress status with detailed logging"""
    global db_creation_status
    
    # Log the progress update
    log_message = f"Progress: {progress}% - {message}"
    if total_files > 0:
        log_message += f" (Processed {files_processed} of {total_files} files)"
    
    # Log at appropriate level based on progress
    if progress == 0:
        logger.info(f"Starting database update: {message}")
    elif progress == 100:
        logger.info(f"Database update complete: {message}")
    elif progress % 20 == 0:  # Log every 20%
        logger.info(log_message)
    else:
        logger.debug(log_message)
    
    # Update the status
    db_creation_status.update({
        "progress": progress,
        "message": message,
        "files_processed": files_processed,
        "total_files": total_files,
        "last_updated": datetime.now().isoformat()
    })

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
    
    logger.info("Starting database creation in background.")
    logger.debug(f"Force recreate: {force_recreate}")
    
    # Initialize status with detailed information
    start_time = datetime.now()
    db_creation_status.update({
        "in_progress": True, 
        "completed": False, 
        "error": None,
        "progress": 0,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "message": "Initializing database creation...",
        "files_processed": 0,
        "total_files": 0
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
    
    return {"message": "Database creation started", "status": db_creation_status}

@app.get("/db-status")
def get_db_status():
    """Get detailed database status"""
    logger.debug("Database status requested")
    logger.debug(f"Current status: {db_creation_status}")
    
    # Log a warning if database creation failed
    if db_creation_status.get("error"):
        logger.warning(f"Database creation failed: {db_creation_status['error']}")
    
    # Log if database creation is taking too long
    if db_creation_status["in_progress"] and "start_time" in db_creation_status and db_creation_status["start_time"]:
        try:
            start_time = datetime.fromisoformat(db_creation_status["start_time"])
            duration = (datetime.now() - start_time).total_seconds()
            if duration > 300:  # 5 minutes
                logger.warning(f"Database creation is taking longer than expected: {duration:.1f} seconds")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating database creation duration: {e}")
    
    return db_creation_status

@app.get("/supported-formats")
def get_supported_formats():
    """Get list of supported document formats"""
    logger.info("Fetching list of supported document formats")
    
    formats = [
        {"extension": ".txt", "type": "Plain Text"},
        {"extension": ".pdf", "type": "PDF Document"},
        {"extension": ".docx", "type": "Microsoft Word"},
        {"extension": ".doc", "type": "Microsoft Word (Legacy)"}
    ]
    
    logger.debug(f"Supported formats: {formats}")
    return {"supported_formats": formats}

@app.get("/documents")
def list_documents():
    """List all uploaded documents"""
    logger.info("Listing all documents")
    
    try:
        if 'rag_handler' in globals():
            from llm.improved_rag_handler import DOCUMENTS_PATH
        else:
            DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
        
        # Ensure the documents directory exists
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        
        # Get all files in the documents directory
        documents = []
        for filename in os.listdir(DOCUMENTS_PATH):
            file_path = os.path.join(DOCUMENTS_PATH, filename)
            if os.path.isfile(file_path):
                file_stat = os.stat(file_path)
                documents.append({
                    "filename": filename,
                    "size": file_stat.st_size,
                    "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "extension": os.path.splitext(filename)[1].lower()
                })
        
        logger.info(f"Found {len(documents)} documents")
        return {"documents": documents}
        
    except Exception as e:
        error_msg = f"Error listing documents: {str(e)}"
        logger.error(f"{error_msg}. Error details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document file to the documents directory"""
    logger.info(f"Received file upload request for: {file.filename}")
    logger.debug(f"File size: {file.size} bytes")
    
    try:
        if 'rag_handler' in globals():
            from llm.improved_rag_handler import DOCUMENTS_PATH
            logger.debug("Using improved_rag_handler DOCUMENTS_PATH")
        else:
            DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
            logger.debug(f"Using default DOCUMENTS_PATH: {DOCUMENTS_PATH}")
        
        # Ensure the documents directory exists
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        logger.debug(f"Documents directory ready at: {DOCUMENTS_PATH}")
        
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        logger.debug(f"File extension: {file_ext}")
        
        if file_ext not in ['.txt', '.pdf', '.docx', '.doc']:
            error_msg = f"Unsupported file format: {file_ext}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Save the file
        file_path = os.path.join(DOCUMENTS_PATH, file.filename)
        logger.info(f"Saving file to: {file_path}")
        
        start_time = time.time()
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(file_path)
            logger.info(f"File saved successfully. Size: {file_size} bytes")
            
            return {
                "message": f"File {file.filename} uploaded successfully",
                "path": file_path,
                "size": file_size,
                "upload_time": time.time() - start_time
            }
            
        except IOError as e:
            error_msg = f"Failed to save file: {str(e)}"
            logger.error(f"{error_msg}. Error details: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during file upload: {str(e)}"
        logger.error(f"{error_msg}. Error details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a document from the documents directory"""
    logger.info(f"Request to delete document: {filename}")
    
    try:
        if 'rag_handler' in globals():
            from llm.improved_rag_handler import DOCUMENTS_PATH
        else:
            DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "documents")
        
        # Security check: prevent path traversal attacks
        if '..' in filename or filename.startswith('/'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(DOCUMENTS_PATH, filename)
        logger.info(f"Filepath is: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete the file
        os.remove(file_path)
        logger.info(f"Document deleted: {filename}")
        
        # Also try to delete the corresponding text file if it exists
        text_file_path = os.path.join(DOCUMENTS_PATH, f"{os.path.splitext(filename)[0]}.txt")
        if os.path.exists(text_file_path):
            os.remove(text_file_path)
        
        logging.info(f"Document deleted: {filename}")
        return {"message": f"Document '{filename}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting enhanced backend server.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)