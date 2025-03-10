from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import Document
import os
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Setup folders and environment with absolute paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGITAL_DOCS_DIR = os.path.join(BASE_DIR, "digital_docs")
SCANNED_DOCS_DIR = os.path.join(BASE_DIR, "scanned_docs")

# Ensure directories exist
os.makedirs(DIGITAL_DOCS_DIR, exist_ok=True)
os.makedirs(SCANNED_DOCS_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ----------------------------
# Define OCR Functions with improved error handling
# ----------------------------
def extract_text_from_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return f"PDF file does not exist: {pdf_path}"
            
        if os.path.getsize(pdf_path) == 0:
            logger.error(f"PDF file is empty: {pdf_path}")
            return f"PDF file is empty: {pdf_path}"
            
        logger.info(f"Converting PDF to images: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=300)
        text = ""
        for i, page in enumerate(pages):
            logger.info(f"Processing PDF page {i + 1}/{len(pages)}")
            page_text = pytesseract.image_to_string(page)
            text += page_text
            logger.info(f"Extracted {len(page_text)} characters from page {i+1}")
        
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from PDF: {pdf_path}")
        else:
            logger.info(f"Extracted total of {len(text)} characters from PDF: {pdf_path}")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return f"Error processing PDF: {str(e)}"

def extract_text_from_image(image_path):
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return f"Image file does not exist: {image_path}"
            
        if os.path.getsize(image_path) == 0:
            logger.error(f"Image file is empty: {image_path}")
            return f"Image file is empty: {image_path}"
            
        logger.info(f"Opening image for OCR: {image_path}")
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        if not text or len(text.strip()) == 0:
            logger.warning(f"No text extracted from image: {image_path}")
        else:
            logger.info(f"Extracted {len(text)} characters from image: {image_path}")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {e}")
        return f"Error processing image: {str(e)}"

def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return f"File does not exist: {file_path}"
    
    if os.path.getsize(file_path) == 0:
        logger.error(f"File is empty: {file_path}")
        return f"File is empty: {file_path}"
        
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Processing file: {file_path} with extension {file_extension}")
    
    if file_extension in ['.pdf']:
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        return extract_text_from_image(file_path)
    else:
        try:
            # Try to read as a text file
            logger.info(f"Reading as text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            if not text or len(text.strip()) == 0:
                logger.warning(f"No text content in file: {file_path}")
            else:
                logger.info(f"Read {len(text)} characters from text file: {file_path}")
                
            return text
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                logger.info(f"Retrying with different encoding: {file_path}")
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return text
            except:
                error_msg = f"Unable to read file as text: {file_extension}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Unsupported file type or error reading file: {file_extension}, {str(e)}"
            logger.error(error_msg)
            return error_msg

# ----------------------------
# Initialize LlamaIndex components (loaded only once)
# ----------------------------
system_prompt = """<|SYSTEM|># You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the documents. Your answer should be precise.
"""
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

try:
    llm = HuggingFaceLLM(
        context_window=1024,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        model_kwargs={"torch_dtype": torch.float16}
    )
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info("LLM and embedding models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    llm = None
    embed_model = None

# Global variable to hold the query engine
query_engine = None
has_documents = False

# ----------------------------
# Function to process documents and build index
# ----------------------------
def process_documents():
    global query_engine, has_documents
    
    processed_texts = []
    
    # Check if there are any documents to process using absolute paths
    digital_files = os.listdir(DIGITAL_DOCS_DIR)
    scanned_files = os.listdir(SCANNED_DOCS_DIR)
    
    logger.info(f"Found {len(digital_files)} digital files and {len(scanned_files)} scanned files")
    logger.info(f"Digital files: {digital_files}")
    logger.info(f"Scanned files: {scanned_files}")
    
    # Check if all directories exist and are accessible
    logger.info(f"Digital docs path exists: {os.path.exists(DIGITAL_DOCS_DIR)}")
    logger.info(f"Scanned docs path exists: {os.path.exists(SCANNED_DOCS_DIR)}")
    
    if not digital_files and not scanned_files:
        logger.warning("No documents found in either directory.")
        has_documents = False
        return False
    
    # Process scanned documents
    for file_name in scanned_files:
        file_path = os.path.join(SCANNED_DOCS_DIR, file_name)
        try:
            # Check if file exists and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"Processing scanned file: {file_name} (size: {os.path.getsize(file_path)} bytes)")
                text = extract_text_from_file(file_path)
                if text and len(text.strip()) > 0:
                    processed_texts.append({"id": f"scanned_{file_name}", "text": text})
                    logger.info(f"Successfully extracted text from scanned file: {file_name}")
                else:
                    logger.warning(f"No text extracted from file: {file_name}")
            else:
                logger.warning(f"File doesn't exist or is empty: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    
    # Process digital documents
    if digital_files:
        logger.info("Processing Digital Docs")
        try:
            # First try using SimpleDirectoryReader for compatible files
            try:
                logger.info(f"Attempting to use SimpleDirectoryReader for {DIGITAL_DOCS_DIR}")
                documents = SimpleDirectoryReader(DIGITAL_DOCS_DIR).load_data()
                print("Documents:", documents)
                logger.info(f"SimpleDirectoryReader loaded {len(documents)} documents")
                
                for i, doc in enumerate(documents):
                    try:
                        doc_text = doc.text_resource.text if hasattr(doc.text_resource, 'text') else str(doc)
                        print(doc_text)
                        if doc_text and len(doc_text.strip()) > 0:
                            processed_texts.append({"id": f"digital_{i}", "text": doc_text})
                            logger.info(f"Successfully processed digital doc #{i}")
                        else:
                            logger.warning(f"Empty text from SimpleDirectoryReader for doc #{i}")
                    except Exception as e:
                        logger.error(f"Error processing document from SimpleDirectoryReader: {e}")
                        
            except Exception as e:
                logger.warning(f"SimpleDirectoryReader failed: {e}, falling back to manual processing")
                
                # Fall back to manual processing
                for file_name in digital_files:
                    file_path = os.path.join(DIGITAL_DOCS_DIR, file_name)
                    try:
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            logger.info(f"Processing digital file: {file_name} (size: {os.path.getsize(file_path)} bytes)")
                            text = extract_text_from_file(file_path)
                            if text and len(text.strip()) > 0:
                                processed_texts.append({"id": f"digital_{file_name}", "text": text})
                                logger.info(f"Successfully extracted text from digital file: {file_name}")
                            else:
                                logger.warning(f"No text extracted from file: {file_name}")
                        else:
                            logger.warning(f"File doesn't exist or is empty: {file_path}")
                    except Exception as e:
                        logger.error(f"Error manually processing file {file_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing digital documents: {e}")
    
    # Check if we have any processed texts
    if not processed_texts:
        logger.warning("No documents were successfully processed.")
        has_documents = False
        return False
    
    # Log the number of documents processed
    logger.info(f"Successfully processed {len(processed_texts)} documents")
    for doc in processed_texts:
        text_preview = doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"]
        logger.info(f"Document {doc['id']}: {text_preview}")
    
    # Convert processed texts into LlamaIndex-compatible format
    formatted_documents = []
    for doc in processed_texts:
        if doc["text"] and len(doc["text"].strip()) > 0:
            formatted_documents.append(Document(text=doc["text"], doc_id=doc["id"]))
    
    if not formatted_documents:
        logger.warning("No valid documents after formatting.")
        has_documents = False
        return False
    
    try:
        # Create the index and query engine
        logger.info(f"Creating index with {len(formatted_documents)} documents")
        index = VectorStoreIndex.from_documents(
            formatted_documents,
            chunk_size=1024,
            llm=llm,
            embed_model=embed_model
        )
        query_engine = index.as_query_engine(llm=llm)
        has_documents = True
        logger.info("Documents processed and index created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        has_documents = False
        return False

# Try to process documents on startup (if any exist)
try:
    process_documents()
except Exception as e:
    logger.error(f"Error during initial document processing: {e}")

# ----------------------------
# Create Flask App to handle queries
# ----------------------------
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    global query_engine, has_documents
    
    data = request.get_json()
    user_query = data.get("query", "").strip()
    
    logger.info(f"Received query: {user_query}")
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    if not has_documents:
        logger.warning("Query received but no documents are processed")
        return jsonify({"response": "You haven't uploaded any documents yet. Please upload documents and click 'Process Documents' to proceed."}), 200
    
    try:
        logger.info("Sending query to the engine")
        response = query_engine.query(user_query)
        response_text = str(response)
        
        logger.info(f"Response received: {response_text[:100]}...")
        
        if not response_text or response_text.strip() == "":
            logger.warning("Empty response received from query engine")
            return jsonify({"response": "No relevant information found in the documents. Please try a different query or upload more relevant documents."}), 200
        
        return jsonify({"response": response_text})
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        logger.info("Processing documents request received")
        success = process_documents()
        if success:
            logger.info("Documents processed successfully")
            return jsonify({"status": "success", "message": "Documents processed successfully"})
        else:
            logger.warning("Document processing failed - no valid documents found")
            return jsonify({"status": "error", "message": "No documents found or error processing documents"}), 400
    except Exception as e:
        error_msg = f"Error during document processing: {str(e)}"
        logger.error(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/status', methods=['GET'])
def status():
    digital_count = len(os.listdir(DIGITAL_DOCS_DIR))
    scanned_count = len(os.listdir(SCANNED_DOCS_DIR))
    
    logger.info(f"Status request: has_documents={has_documents}, digital_count={digital_count}, scanned_count={scanned_count}")
    
    return jsonify({
        "has_documents": has_documents,
        "digital_count": digital_count,
        "scanned_count": scanned_count
    })

if __name__ == '__main__':
    logger.info("Starting backend service on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)