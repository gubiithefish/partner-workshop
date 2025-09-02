import os
import re
from pathlib import Path
from typing import List

from ftfy import fix_text
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from pypdf import PdfReader

# --- LlamaIndex for Chunking ---
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Milvus admin and data management ---
from pymilvus import connections, utility, DataType, FieldSchema, CollectionSchema, Collection

# =========================
# Load environment & setup
# =========================
load_dotenv()
console = Console()

# =========================
# Config
# =========================
# ---> IMPORTANT: Set the path to your PDF file here <---
PDF_FILE_PATH = Path("2024-annual-report-ceo-letter.pdf") 

# Chunking settings for the PDF
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Milvus Configuration (ensure these are in your .env file or set as environment variables)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # GRPC host
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")      # GRPC port
MILVUS_USER = os.getenv("MILVUS_USER", "ibmlhapikey") # Usually 'ibmlhapikey' for watsonx.data
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")       # Your IAM API key
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "cyber_security_col")

# =========================
# Embeddings (HF MiniLM L6 v2)
# =========================
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embedding_dim() -> int:
    """Calculates the dimension of the embedding model."""
    vec = embedding_model.get_query_embedding("probe")
    return len(vec)

def clean_text(text: str) -> str:
    """
    A robust function to clean text data.
    - Fixes Unicode and encoding errors with ftfy.
    - Replaces all ASCII control characters with a space.
    - Strips leading/trailing whitespace.
    """
    # 1. Fix text encoding, mojibake, etc.
    text = fix_text(text)
    
    # 2. Remove ASCII control characters
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    
    return text.strip()

# =========================
# Milvus: Setup and Data Ingestion
# =========================
def setup_milvus_collection():
    """
    Connects to Milvus, ensures the collection exists with the correct schema,
    and ingests data from the specified PDF if the collection is empty.
    """
    console.print(Panel.fit(f"[bold]Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...[/]"))

    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
            secure=True,
            timeout=5,
        )
        console.print("[green]Successfully connected to Milvus.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error connecting to Milvus: {e}[/bold red]")
        return

    embed_dim = get_embedding_dim()
    console.print(f"[cyan]Embedding model dimension: {embed_dim}[/]")

    # Drop collection if schema mismatch is detected (simple check for 'title' field)
    if utility.has_collection(COLLECTION_NAME):
        col_temp = Collection(COLLECTION_NAME)
        if "title" not in [f.name for f in col_temp.schema.fields]:
             console.print(
                f"[yellow]Warning: Collection '{COLLECTION_NAME}' has an outdated schema. "
                f"Dropping collection to apply the new schema...[/]"
            )
             utility.drop_collection(COLLECTION_NAME)

    # Create collection if it doesn't exist
    if not utility.has_collection(COLLECTION_NAME):
        console.print(f"[blue]Creating collection '{COLLECTION_NAME}' with a new schema...[/]")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="body", dtype=DataType.VARCHAR, max_length=6000),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
        ]
        schema = CollectionSchema(fields, description="Cyber Security PDF Documents")
        col = Collection(name=COLLECTION_NAME, schema=schema)
    else:
        col = Collection(COLLECTION_NAME)
        console.print(f"[green]Using existing collection '{COLLECTION_NAME}'.[/green]")

    # Ingest data only if the collection is empty
    if col.num_entities == 0:
        console.print(Panel.fit(f"[bold]Collection is empty. Ingesting data from '{PDF_FILE_PATH.name}'...[/]"))
        
        if not PDF_FILE_PATH.exists():
            console.print(f"[bold red]Error: PDF file not found at '{PDF_FILE_PATH}'[/bold red]")
            return

        # Read the PDF text using pypdf
        console.print(f"Reading text from '{PDF_FILE_PATH.name}' using pypdf...")
        reader = PdfReader(PDF_FILE_PATH)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        # Split the text into chunks using SentenceSplitter
        console.print("Splitting text into chunks...")
        text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        bodies = text_splitter.split_text(full_text)
        
        # Clean each text chunk
        bodies = [clean_text(text) for text in bodies]
        
        # Prepare metadata for the new schema
        pdf_filename = PDF_FILE_PATH.name
        titles: List[str] = [f"{pdf_filename} - Chunk {i+1}" for i in range(len(bodies))]
        urls: List[str] = [str(PDF_FILE_PATH.resolve()) for _ in range(len(bodies))]
        
        console.print(f"PDF split into {len(bodies)} chunks.")

        # Generate embeddings
        console.print("Generating embeddings for all text bodies...")
        embeddings = embedding_model.get_text_embedding_batch(bodies, show_progress=True)

        # Insert data for all fields into the collection
        console.print(f"Inserting {len(bodies)} records into '{COLLECTION_NAME}'...")
        col.insert([embeddings, titles, bodies, urls])
        col.flush()
        console.print(f"[green]Successfully inserted {col.num_entities} entities.[/green]")

        # Create a vector index
        console.print("Creating vector index (IVF_FLAT)...")
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        col.create_index(field_name="embedding", index_params=index_params)
        console.print("[green]Index created successfully.[/green]")
    else:
        console.print(f"Collection '{COLLECTION_NAME}' already contains {col.num_entities} entities. Skipping ingestion.")

    connections.disconnect("default")
    console.print("[bold]Milvus ingestion process complete.[/bold]")


# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # This script now only performs the setup and ingestion process.
    setup_milvus_collection()