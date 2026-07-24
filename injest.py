import os
import chromadb
import uuid
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv  # <-- 1. Import dotenv

PDF_PATH = "dataset/attention.pdf"  # Make sure this matches your PDF name!
DB_PATH = "./chroma_db"
COLLECTION_NAME = "attention_is_all_you_need"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def extract_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap):
    print(f"📖 Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    chunks = []
    metadatas = []
    
    # Extract text from all pages
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        # Standard sliding window chunking loop
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append(chunk_text)
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": page_num + 1
            })
            
            # Slide window forward, keeping the overlap
            start += (chunk_size - chunk_overlap)
            
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks, metadatas

# ==========================================
# 3. Generate Embeddings using Google Gen AI
# ==========================================
def get_embeddings(texts, model_name="gemini-embedding-2"):
    print(f"🧠 Generating embeddings using {model_name}...")
    api_key = os.environ.get("GEMINI_API_KEY")
    print(api_key)
    client = genai.Client(api_key=api_key)
    
    embeddings = []
    # We embed chunks individually or in small batches to guarantee stability 
    # and handle the model's single-context input behavior cleanly.
    for text in texts:
        response = client.models.embed_content(
            model=model_name,
            contents=text
        )
        # Extract the float values of the first embedding returned
        vector = response.embeddings[0].values
        embeddings.append(vector)
        
    return embeddings

# ==========================================
# 4. Save to Chroma DB
# ==========================================
def save_to_vector_db(chunks, metadatas, embeddings):
    print(f"💾 Saving vectors to local Chroma DB at '{DB_PATH}'...")
    # Initialize Persistent Client (saves database to disk)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create or retrieve the collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Generate unique string IDs for every chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    
    # Add records to the collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"🎉 Successfully stored {len(chunks)} records in collection '{COLLECTION_NAME}'!")

# ==========================================
# Execution Flow
# ==========================================
if __name__ == "__main__":
    # Run the ingestion process
    load_dotenv()
    chunks, metadatas = extract_and_chunk_pdf(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = get_embeddings(chunks)
    save_to_vector_db(chunks, metadatas, embeddings)