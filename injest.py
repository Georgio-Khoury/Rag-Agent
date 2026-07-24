import os
import shutil
import chromadb
import uuid
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Folders configuration
UNPROCESSED_FOLDER = "dataset/unprocessed"
PROCESSED_FOLDER = "dataset/processed"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def document_already_exists(collection, filename):
    """Sanity check: Check if chunks from this PDF filename already exist in ChromaDB."""
    try:
        results = collection.get(
            where={"source": filename},
            limit=1
        )
        # If the results list of IDs is not empty, the document exists
        return len(results.get("ids", [])) > 0
    except Exception:
        return False

def generate_document_summary(pdf_path):
    print(f"🤖 Generating document summary for: {os.path.basename(pdf_path)}")
    reader = PdfReader(pdf_path)
    
    sample_text = ""
    for page in reader.pages[:2]:
        text = page.extract_text()
        if text:
            sample_text += text + "\n"
            
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    prompt = f"""Provide a concise, one-to-two sentence summary explaining what this document is about. 
This summary will be used by an AI routing agent to match user queries to this document.

Document Sample:
{sample_text[:3000]}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )
    summary = response.text.strip()
    print(f"📝 Summary: {summary}")
    return summary

def extract_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap):
    print(f"📖 Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    chunks = []
    metadatas = []
    
    doc_summary = generate_document_summary(pdf_path)
    doc_name = os.path.basename(pdf_path)
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append(chunk_text)
            metadatas.append({
                "source": doc_name,
                "summary": doc_summary,
                "page": page_num + 1
            })
            
            start += (chunk_size - chunk_overlap)
            
    print(f"✅ Created {len(chunks)} chunks for {doc_name}.")
    return chunks, metadatas

def get_embeddings(texts, model_name="gemini-embedding-2"):
    print(f"🧠 Generating embeddings using {model_name}...")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model=model_name,
            contents=text
        )
        embeddings.append(response.embeddings[0].values)
        
    return embeddings

def save_to_vector_db(collection, chunks, metadatas, embeddings):
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"💾 Stored {len(chunks)} records in master collection '{COLLECTION_NAME}'.")

def process_all_documents():
    if not os.path.exists(UNPROCESSED_FOLDER):
        os.makedirs(UNPROCESSED_FOLDER)
        os.makedirs(PROCESSED_FOLDER)
        print(f"📁 Created directories. Drop your PDFs into '{UNPROCESSED_FOLDER}' and run again!")
        return

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Initialize Chroma client and collection once
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    files = [f for f in os.listdir(UNPROCESSED_FOLDER) if f.lower().endswith(".pdf")]
    
    if not files:
        print(f"⚠️ No unprocessed PDFs found in '{UNPROCESSED_FOLDER}'.")
        return

    for filename in files:
        pdf_path = os.path.join(UNPROCESSED_FOLDER, filename)
        print(f"\n-----------------------------------------")
        print(f"🚀 Checking: {filename}")
        print(f"-----------------------------------------")
        
        # Sanity Check: Ensure it doesn't already exist in the database by filename
        if document_already_exists(collection, filename):
            print(f"🛡️ [SKIP] '{filename}' is already processed and exists in ChromaDB. Moving to processed folder.")
            shutil.move(pdf_path, os.path.join(PROCESSED_FOLDER, filename))
            continue
        
        try:
            # 1. Extract and chunk
            chunks, metadatas = extract_and_chunk_pdf(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                print(f"⚠️ Skipping {filename} (empty or unreadable text).")
                continue
                
            # 2. Get embeddings
            embeddings = get_embeddings(chunks)
            
            # 3. Save to database
            save_to_vector_db(collection, chunks, metadatas, embeddings)
            
            # 4. Move to processed folder
            shutil.move(pdf_path, os.path.join(PROCESSED_FOLDER, filename))
            print(f"📦 Moved {filename} to '{PROCESSED_FOLDER}/'.")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    load_dotenv()
    process_all_documents()
    print("\n🎉 Batch ingestion workflow finished!")

