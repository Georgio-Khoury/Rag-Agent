import chromadb
import pandas as pd

def dump_chroma_collection_to_txt(db_path: str, collection_name: str, output_filename: str = "chromaOut.txt"):
    # 1. Connect to your persistent local database
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        # 2. Grab the collection
        collection = client.get_collection(name=collection_name)
        
        # 3. Pull ALL records out of the collection
        all_data = collection.get(include=["documents", "metadatas", "embeddings"])
        
        # 4. Check if the collection is empty
        if not all_data["ids"]:
            status_msg = f"The collection '{collection_name}' is empty."
            print(status_msg)
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(status_msg)
            return

        # 5. Format the dictionary safely into a scannable Pandas DataFrame
        df = pd.DataFrame({
            "ID": all_data["ids"],
            "Document Text": all_data["documents"],
            "Metadata": all_data["metadatas"],
            # Fix: safe array evaluation check using vec is not None and len()
            "Embedding (First 3 Dimensions)": [
                list(vec)[:3] + ["..."] if vec is not None and len(vec) > 0 else None 
                for vec in all_data["embeddings"]
            ] if all_data.get("embeddings") is not None else "Not Retrieved"
        })
        
        # 6. Configure pandas layout for clean text dumping
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', 1000)
        
        # 7. Construct the final output string
        output_str = f"--- Total Rows Found: {len(df)} ---\n"
        output_str += df.to_string(index=False)
        
        # 8. Write directly to the txt file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(output_str)
            
        print(f"Successfully dumped collection data to {output_filename}!")

    except Exception as e:
        error_msg = f"Error accessing collection: {e}"
        print(error_msg)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(error_msg)

# Example Usage:
dump_chroma_collection_to_txt(db_path="./chroma_db", collection_name="knowledge_base")