import streamlit as st
import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Custom Chatbot Q&A", page_icon="🤖")

st.title("Custom Chatbot Q&A 🤖")

st.write("This is a custom chatbot that can answer questions about your documents. "
         "Upload your documents, create the database, and ask your questions.")

# Check database status
logging.info("Checking database status.")
try:
    status_response = requests.get(f"{BACKEND_URL}/db-status")
    if status_response.status_code == 200:
        db_status = status_response.json()
        logging.info(f"Database status: {db_status}")
        if db_status["in_progress"]:
            st.info("Database creation in progress...")
        elif db_status["completed"]:
            st.success("Database is ready!")
        elif db_status["error"]:
            st.error(f"Database creation failed: {db_status['error']}")
            logging.error(f"Database creation failed: {db_status['error']}")
    else:
        logging.warning(f"Failed to get database status. Status code: {status_response.status_code}")
except Exception as e:
    st.warning("Could not connect to backend to check database status.")
    logging.error("Could not connect to backend to check database status.", exc_info=True)

if st.button("Create Database"):
    logging.info("'Create Database' button clicked.")
    with st.spinner("Creating database..."):
        try:
            logging.info("Sending request to create database.")
            response = requests.post(f"{BACKEND_URL}/create-db")
            if response.status_code == 200:
                st.success(response.json()["message"])
                logging.info("Database creation request successful.")
                # Refresh the page to update status
                st.rerun()
            else:
                st.error(f"Error creating database: {response.json().get('detail', 'Unknown error')}")
                logging.error(f"Error creating database: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
            logging.error(f"Error connecting to backend: {e}", exc_info=True)

question = st.text_input("Ask your question:")

if st.button("Ask"):
    logging.info("'Ask' button clicked.")
    if question:
        logging.info(f"User asked question: {question}")
        with st.spinner("Thinking..."):
            try:
                payload = {"question": question}
                logging.info(f"Sending question to backend: {payload}")
                response = requests.post(f"{BACKEND_URL}/ask", json=payload)
                if response.status_code == 200:
                    st.write("### Answer")
                    st.write(response.json()["answer"])
                    logging.info("Successfully received answer from backend.")
                else:
                    st.error(f"Error getting answer: {response.json().get('detail', 'Unknown error')}")
                    logging.error(f"Error getting answer: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")
                logging.error(f"Error connecting to backend: {e}", exc_info=True)
    else:
        st.warning("Please enter a question.")
        logging.warning("User clicked 'Ask' without entering a question.")
