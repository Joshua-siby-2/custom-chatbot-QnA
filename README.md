
# Custom Chatbot Q&A (RAG Application)

This project is a custom chatbot that can answer questions about your documents. It uses a Retrieval-Augmented Generation (RAG) architecture with LangChain, ChromaDB, and a local LLM running with Ollama.

## Project Structure

```
custom-chatbot-qna/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── llm/
│   ├── rag_handler.py
│   └── requirements.txt
├── documents/
│   └── sample.txt
├── .gitignore
└── README.md
```

## Setup

1.  **Install Ollama:** Make sure you have Ollama installed and running. You can download it from [https://ollama.ai/](https://ollama.ai/).

2.  **Pull the LLM model:**

    ```bash
    ollama pull llama2
    ```

3.  **Install Python dependencies:**

    ```bash
    # Install backend dependencies
    pip install -r backend/requirements.txt

    # Install frontend dependencies
    pip install -r frontend/requirements.txt

    # Install llm dependencies
    pip install -r llm/requirements.txt
    ```

4.  **Add your documents:** Place the text documents you want to chat with into the `documents` directory.

## How to Run

1.  **Start the backend server:**

    ```bash
    python backend/main.py
    ```

2.  **Run the frontend application:**

    ```bash
    streamlit run frontend/app.py
    ```

3.  **Create the database:** Open the Streamlit application in your browser and click the "Create Database" button. This will create a ChromaDB vector store from your documents.

4.  **Ask questions:** Once the database is created, you can ask questions about your documents using the text input.
