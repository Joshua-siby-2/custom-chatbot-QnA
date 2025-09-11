import os
import logging
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

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

logging.info(f"Project root: {PROJECT_ROOT}")
logging.info(f"Documents path: {DOCUMENTS_PATH}")
logging.info(f"DB path: {PERSIST_DIRECTORY}")

def create_vector_db():
    """
    Creates a ChromaDB vector store from documents in the DOCUMENTS_PATH.
    """
    logging.info(f"Checking if vector store exists at {PERSIST_DIRECTORY}")
    if os.path.exists(PERSIST_DIRECTORY):
        logging.info(f"Vector store already exists at {PERSIST_DIRECTORY}. Skipping creation.")
        return

    # Create documents directory if it doesn't exist
    logging.info(f"Checking if documents directory exists at {DOCUMENTS_PATH}")
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        logging.info(f"Created documents directory at {DOCUMENTS_PATH}")
        logging.warning("Please add some .txt files to the documents directory and run again.")
        return

    logging.info(f"Creating vector store from documents in {DOCUMENTS_PATH}...")
    
    # Check if there are any .txt files in the directory
    txt_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.txt')]
    logging.info(f"Found {len(txt_files)} .txt files: {txt_files}")
    
    if not txt_files:
        logging.warning("No .txt files found in documents directory.")
        return

    try:
        loader = DirectoryLoader(DOCUMENTS_PATH, glob="*", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents")

        if not documents:
            logging.warning("No documents could be loaded. Please check your .txt files.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(texts)} text chunks")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        vectordb.persist()
        logging.info(f"Vector store created and persisted at {PERSIST_DIRECTORY}.")
        
    except Exception as e:
        logging.error(f"Error creating vector database: {str(e)}", exc_info=True)
        raise


def get_qa_chain():
    """
    Creates a RetrievalQA chain for question answering.
    """
    # Check if vector database exists
    logging.info("Checking for existing vector database.")
    if not os.path.exists(PERSIST_DIRECTORY):
        logging.warning("Vector database not found. Please create it first.")
        return None
    
    try:
        logging.info("Loading vector database.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        retriever = vectordb.as_retriever()

        logging.info("Initializing LLM.")
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

        logging.info("Creating QA chain.")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        logging.info("QA chain created successfully.")
        return qa_chain
    except Exception as e:
        logging.error(f"Error creating QA chain: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    # Create the vector database if it doesn't exist
    create_vector_db()

    # Example usage
    qa_chain = get_qa_chain()
    if qa_chain:
        question = "What is Gemini?"
        logging.info(f"Testing QA chain with question: {question}")
        result = qa_chain({"query": question})
        print("Question:", question)
        print("Answer:", result["result"])
        logging.info(f"Test finished. Answer: {result['result']}")
