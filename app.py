import streamlit as st
import requests
import os
import time

# Set the page title and layout
st.set_page_config(page_title="RAG based Document Q&A", layout="wide")
st.title("RAG based Document Q&A")

# Define absolute paths for file storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGITAL_DOCS_DIR = os.path.join(BASE_DIR, "digital_docs")
SCANNED_DOCS_DIR = os.path.join(BASE_DIR, "scanned_docs")

# Ensure directories exist
os.makedirs(DIGITAL_DOCS_DIR, exist_ok=True)
os.makedirs(SCANNED_DOCS_DIR, exist_ok=True)

# Backend URL (Update this whenever ngrok gives a new URL)
# BACKEND_URL = "https://df98-34-82-225-135.ngrok-free.app"
BACKEND_URL = "http://localhost:5000"

# Application state initialization
if 'documents_processed' not in st.session_state:
    # Try to get initial status from backend
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.ok:
            status = response.json()
            st.session_state.documents_processed = status.get('has_documents', False)
            st.session_state.digital_count = status.get('digital_count', 0)
            st.session_state.scanned_count = status.get('scanned_count', 0)
        else:
            st.warning(f"Could not get status from backend: {response.text}")
            st.session_state.documents_processed = False
            st.session_state.digital_count = 0
            st.session_state.scanned_count = 0
    except Exception as e:
        st.warning(f"Could not connect to backend: {e}")
        st.session_state.documents_processed = False
        st.session_state.digital_count = 0
        st.session_state.scanned_count = 0

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: File Uploads
st.sidebar.header("Upload Documents")

# Display current document counts
st.sidebar.subheader("Current Documents")
st.sidebar.info(f"Digital Documents: {st.session_state.digital_count}")
st.sidebar.info(f"Scanned Documents: {st.session_state.scanned_count}")

# Create backend connection status indicator
backend_status = st.sidebar.empty()
try:
    response = requests.get(f"{BACKEND_URL}/status", timeout=3)
    if response.ok:
        backend_status.success("✅ Backend connected")
    else:
        backend_status.error("❌ Backend connection issue")
except:
    backend_status.error("❌ Backend not connected")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.subheader("Digital Documents")
    digital_files = st.file_uploader(
        "Upload Digital Docs",
        type=["png", "jpg", "jpeg", "pdf", "bmp", "tiff", "txt", "md", "html", "doc", "docx"],
        accept_multiple_files=True,
        key="digital_uploader"
    )

with col2:
    st.subheader("Scanned Documents")
    scanned_files = st.file_uploader(
        "Upload Scanned Docs",
        type=["png", "jpg", "jpeg", "pdf", "bmp", "tiff"],
        accept_multiple_files=True,
        key="scanned_uploader"
    ) 

# Function to save uploaded files locally with improved error handling
def save_uploaded_files(uploaded_files, target_dir):
    saved_count = 0
    if uploaded_files:
        for file in uploaded_files:
            try:
                file_path = os.path.join(target_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Verify file was saved correctly
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    saved_count += 1
                    st.sidebar.success(f"Saved {file.name} to {target_dir} (size: {os.path.getsize(file_path)} bytes)")
                else:
                    st.sidebar.error(f"Failed to save {file.name} properly. File empty or missing.")
            except Exception as e:
                st.sidebar.error(f"Error saving {file.name}: {e}")
    return saved_count

# Save uploaded files when button is clicked
if st.sidebar.button("Save Uploaded Files"):
    with st.spinner("Saving files..."):
        try:
            digital_saved = save_uploaded_files(digital_files, DIGITAL_DOCS_DIR)
            scanned_saved = save_uploaded_files(scanned_files, SCANNED_DOCS_DIR)
            
            if digital_saved > 0 or scanned_saved > 0:
                st.session_state.digital_count += digital_saved
                st.session_state.scanned_count += scanned_saved
                st.sidebar.success(f"Saved {digital_saved + scanned_saved} files. Click 'Process Documents' to make them searchable.")
                
                # Get updated status from backend
                try:
                    response = requests.get(f"{BACKEND_URL}/status")
                    if response.ok:
                        status = response.json()
                        st.session_state.digital_count = status.get('digital_count', st.session_state.digital_count)
                        st.session_state.scanned_count = status.get('scanned_count', st.session_state.scanned_count)
                except Exception as e:
                    st.sidebar.warning(f"Could not refresh counts from backend: {e}")
            else:
                st.sidebar.warning("No files selected for upload.")
        except Exception as e:
            st.sidebar.error(f"Error during file saving: {e}")

# Debug button to check files in directory
if st.sidebar.button("Check Files in Directory"):
    try:
        digital_files_list = os.listdir(DIGITAL_DOCS_DIR)
        scanned_files_list = os.listdir(SCANNED_DOCS_DIR)
        
        st.sidebar.write("Digital files:")
        for file in digital_files_list:
            file_path = os.path.join(DIGITAL_DOCS_DIR, file)
            st.sidebar.write(f"- {file} ({os.path.getsize(file_path)} bytes)")
            
        st.sidebar.write("Scanned files:")
        for file in scanned_files_list:
            file_path = os.path.join(SCANNED_DOCS_DIR, file)
            st.sidebar.write(f"- {file} ({os.path.getsize(file_path)} bytes)")
    except Exception as e:
        st.sidebar.error(f"Error checking files: {e}")

# Process documents button with improved error handling
process_button = st.sidebar.button("Process Documents")

if process_button:
    # Use a regular spinner for processing
    process_status = st.sidebar.empty()
    process_status.info("Starting document processing...")
    
    try:
        response = requests.post(f"{BACKEND_URL}/process", timeout=600)
        if response.ok:
            result = response.json()
            if result.get("status") == "success":
                st.session_state.documents_processed = True
                process_status.success("Documents processed successfully!")
                
                # Update status after processing
                try:
                    status_response = requests.get(f"{BACKEND_URL}/status")
                    if status_response.ok:
                        status = status_response.json()
                        st.session_state.documents_processed = status.get('has_documents', False)
                    else:
                        process_status.warning(f"Could not get updated status: {status_response.text}")
                except Exception as e:
                    process_status.warning(f"Error getting status after processing: {e}")
            else:
                error_msg = result.get("message", "Unknown error processing documents")
                process_status.error(f"Processing error: {error_msg}")
                
                # Show more detailed guidance
                if "No documents found" in error_msg:
                    st.sidebar.error("No valid documents found. Please make sure files are uploaded and saved correctly.")
                    st.sidebar.info("Try clicking 'Check Files in Directory' to verify that files exist.")
        else:
            process_status.error(f"Backend error: {response.text}")
    except requests.exceptions.ConnectionError:
        process_status.error("Connection error. Make sure the backend is running.")
    except requests.exceptions.Timeout:
        process_status.error("Request timed out. The processing might be taking too long.")
    except Exception as e:
        process_status.error(f"Error connecting to backend: {e}")

# Display document status
if st.session_state.documents_processed:
    st.sidebar.success("✅ Documents are processed and ready for querying")
else:
    st.sidebar.warning("⚠️ No documents processed yet. Upload documents and click 'Process Documents'")

# Main Panel: Q&A Section
st.header("Question & Answer")

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    st.markdown(f"**Question {i+1}:** {question}")
    st.markdown(f"**Answer {i+1}:** {answer}")
    st.markdown("---")

# Input area with improved error handling
user_query = st.text_input("Enter your question:")

if st.button("Ask") and user_query:
    query_status = st.empty()
    query_status.info("Processing your question...")
    
    try:
        # Make a POST request to the Flask backend
        response = requests.post(
            f"{BACKEND_URL}/query", 
            json={"query": user_query}, 
            timeout=300
        )
        
        if response.ok:
            data = response.json()
            answer = data.get("response", "No response received")
            
            # Add to chat history
            st.session_state.chat_history.append((user_query, answer))
            
            # Display the latest response
            query_status.success("Response received:")
            st.markdown(f"**Question:** {user_query}")
            st.markdown(f"**Answer:** {answer}")
        else:
            query_status.error(f"Backend error: {response.text}")
    except requests.exceptions.ConnectionError:
        query_status.error("Connection error. Make sure the backend is running.")
    except requests.exceptions.Timeout:
        query_status.error("Request timed out. The processing might be taking too long.")
    except Exception as e:
        query_status.error(f"Error connecting to backend: {e}")

# Add a button to clear chat history
if st.button("Clear Chat History") and st.session_state.chat_history:
    st.session_state.chat_history = []
    st.rerun()
