from pypdf import PdfReader

def test_pdf_extraction(pdf_path: str):
    print(f"Opening {pdf_path}...")
    reader = PdfReader(pdf_path)
    
    # Extract structural details
    total_pages = len(reader.pages)
    print(f"Total Pages: {total_pages}\n" + "-"*30)
    
    print(type(reader.pages))
if __name__ == "__main__":
    # Change "sample.pdf" to match your downloaded file's name
    test_pdf_extraction("./dataset/attention.pdf")