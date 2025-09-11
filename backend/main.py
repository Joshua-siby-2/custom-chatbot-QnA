import sys
import os
import threading
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to the sys.path to allow imports from the llm module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.rag_handler import get_qa_chain, create_vector_db

app = FastAPI()

class Question(BaseModel):
    question: str

# Global variable to track database creation status
db_creation_status = {"in_progress": False, "completed": False, "error": None}

def create_db_background():
    """Run database creation in background thread"""
    global db_creation_status
    logging.info("Starting database creation in background.")
    db_creation_status = {"in_progress": True, "completed": False, "error": None}
    try:
        create_vector_db()
        db_creation_status = {"in_progress": False, "completed": True, "error": None}
        logging.info("Database creation completed successfully.")
    except Exception as e:
        db_creation_status = {"in_progress": False, "completed": False, "error": str(e)}
        logging.error(f"Database creation failed: {e}", exc_info=True)

@app.post("/ask")
def ask(question: Question):
    """
    Receives a question, gets the answer from the RAG chain, and returns it.
    """
    logging.info(f"Received question: {question.question}")
    if db_creation_status["in_progress"]:
        logging.warning("Attempted to ask a question while database creation is in progress.")
        raise HTTPException(status_code=400, detail="Database creation in progress. Please wait.")
    
    if not db_creation_status["completed"]:
        logging.warning("Attempted to ask a question before database creation.")
        raise HTTPException(status_code=400, detail="Vector database not found. Please create it first.")
    
    logging.info("Getting QA chain.")
    qa_chain = get_qa_chain()
    if not qa_chain:
        logging.error("QA chain not found after database creation.")
        raise HTTPException(status_code=400, detail="Vector database not found. Please create it first.")
    
    try:
        logging.info("Processing question with QA chain.")
        result = qa_chain({"query": question.question})
        logging.info(f"Successfully answered question: {question.question}")
        return {"answer": result["result"]}
    except Exception as e:
        logging.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/create-db")
def create_db():
    """
    Triggers the creation of the vector database.
    """
    logging.info("Received request to create database.")
    if db_creation_status["in_progress"]:
        logging.info("Database creation already in progress.")
        return {"message": "Database creation already in progress."}
    
    if db_creation_status["completed"]:
        logging.info("Database already exists.")
        return {"message": "Database already exists."}
    
    # Start database creation in background thread
    logging.info("Starting database creation thread.")
    thread = threading.Thread(target=create_db_background)
    thread.daemon = True
    thread.start()
    
    return {"message": "Vector database creation process started."}

@app.get("/db-status")
def get_db_status():
    """Get the current status of database creation"""
    logging.info("Request for database creation status.")
    return db_creation_status

if __name__ == "__main__":
    logging.info("Starting backend server.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
