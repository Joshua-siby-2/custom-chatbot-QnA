Project Title: Custom Chatbot Q&A (RAG Application)
Issues Faced & Resolution Log
Multi-Format Document Processing Integration
Problem: Initially, the system only supported plain text (.txt) files, severely limiting its usefulness for real-world document processing needs.
Root Cause:
Backend lacked text extraction capabilities for PDF, DOCX, and other formats
Frontend file uploader was hardcoded to only accept .txt files
No proper error handling for unsupported file types
Solution:
Backend Enhancement: Implemented extract_text_from_file_async() function with support for:
PDF: Using PyPDF2 for text extraction from PDF documents
DOCX/DOC: Using python-docx for Microsoft Word documents
CSV: Using pandas for structured data extraction
Frontend Upgrade: Modified file uploader to dynamically detect supported formats from backend
Async Processing: Implemented asynchronous file processing to handle large documents without blocking

Streaming Response Implementation Challenges
Problem: The initial implementation provided only batch responses, causing long wait times for complex queries and poor user experience.
Root Cause:
Traditional request-response cycle forced users to wait for complete answer generation
No real-time feedback during LLM processing
Frontend would freeze during long operations
Solution:
Server-Sent Events (SSE): Implemented streaming endpoints using FastAPI's StreamingResponse
Chunked Processing: Modified QA chain to stream responses token-by-token
Frontend Real-time Updates: Added streaming text display with progressive rendering
Progress Indicators: Implemented visual feedback during streaming operations

Extremely Slow /ask Endpoint Performance
Problem: The /ask endpoint was taking 30+ seconds to respond to simple questions, making the API practically unusable for real-time applications. Users experienced significant delays when querying the document database.
Root Cause: Multiple performance bottlenecks were identified:
Repeated LLM Initialization: The QA chain was being recreated on every request
Inefficient Database Loading: Vector database was loaded from disk for each query
No Caching Mechanism: No component reuse between requests
Large Document Retrieval: Retrieving 5 documents instead of optimizing for speed
Synchronous Blocking Operations: All operations were synchronous, blocking the event loop
Solution: Implemented a comprehensive performance optimization strategy:
Component Caching: LLM and QA chain instances cached with 5-minute timeout
Pre-loading Strategy: Vector database loaded during application startup
Optimized Retrieval: Reduced document retrieval from 5 to 3 most relevant chunks
Efficient Source Extraction: Used set-based duplicate detection for faster processing
Async-Compatible Operations: Ensured all operations were non-blocking

Memory Leaks and Resource Exhaustion
Problem: After prolonged usage, the API would consume increasing amounts of memory, eventually leading to crashes and requiring server restarts.
Root Cause:
Unbounded Cache Growth: Cached objects were never cleared or expired
Document Loading Overhead: Large documents were loaded multiple times
Thread Pool Exhaustion: Parallel document processing didn't properly clean up resources
Solution:
Time-based Cache Expiry: Implemented 5-minute TTL for QA chain cache
Lazy Loading: Components initialized only when needed
Resource Cleanup: Proper context managers and thread pool management
Memory Monitoring: Added health checks and memory usage logging
● Database Creation Timeout Issues
Problem: Large document collections would cause database creation to timeout, leaving the system in an inconsistent state.
Root Cause:
No Progress Tracking: Users had no visibility into long-running operations
Blocking Operations: Database creation blocked the main thread
No Resume Capability: Failed operations had to start from scratch
Solution:
Background Processing: Moved database creation to separate threads
Progress Tracking: Implemented real-time progress updates via status endpoint
Incremental Updates: Only process new or modified files
Metadata Tracking: File hashing to track changes and avoid redundant processing

Unsupported File Format Errors
Problem: The system would crash when encountering unsupported file formats or corrupted documents.
Root Cause:
Insufficient Error Handling: Exceptions weren't properly caught and handled
No File Validation: Files weren't validated before processing
All-or-Nothing Processing: Single file failure would stop entire pipeline
Solution:
Graceful Error Handling: Comprehensive try-catch blocks around file processing
File Type Validation: Magic number detection and format verification
Parallel Processing with Isolation: Each file processed independently in thread pool
Error Reporting: Detailed error messages with file-specific information
File Format Support Limitations
○ Problem: Initially, the system only supported plain text files, severely limiting its usefulness.
○ Root Cause: The original implementation used a simple text loader without support for common document formats.
○ Solution:
Implemented multi-format support using specialized loaders (PDF, DOCX, CSV, TXT)
Created a modular loader system that can easily be extended
Added file type detection and appropriate loader selection
Implemented parallel processing for better performance with multiple files
Lack of File Management Capabilities
○ Problem: Users had no way to manage uploaded documents through the interface.
○ Root Cause: The initial implementation focused only on Q&A functionality.
○ Solution:
Added file upload functionality with format validation
Implemented document management features
Created database clearing and rebuilding options
Added support for incremental updates without full rebuilds

