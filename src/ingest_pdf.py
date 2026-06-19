"""
ingest_pdf.py - Phase 1: PDF Ingestion & Local Vector Storage

Loads a PDF, splits it into chunks, generates embeddings using a local
HuggingFace model, and persists the vector store to disk via Chroma.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "../data/document.pdf"
CHROMA_PERSIST_DIR = "../data/chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    try:
        # 1. Load the PDF document
        print(f"Loading PDF: {PDF_PATH} ...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} page(s) from the PDF.")

        # 2. Split into overlapping chunks
        print("Splitting document into chunks ...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunk(s).")

        # 3. Initialize the local embedding model
        print(f"Initializing embedding model: {EMBEDDING_MODEL} ...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 4. Build the Chroma vector store and persist to disk
        print(f"Generating embeddings and storing in '{CHROMA_PERSIST_DIR}' ...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Chroma 0.4+ auto-persists; no manual persist() needed

        # 5. Success summary
        print("\n--- Ingestion Complete ---")
        print(f"Total chunks processed & saved: {len(chunks)}")
        print(f"Vector store persisted at: {CHROMA_PERSIST_DIR}/")

    except FileNotFoundError:
        print(f"ERROR: Could not find '{PDF_PATH}'. "
              "Please place the PDF in the same directory as this script.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
