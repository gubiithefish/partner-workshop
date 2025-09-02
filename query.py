import os
import re
from pathlib import Path
from typing import List, Dict, Any

from ftfy import fix_text
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
from pypdf import PdfReader

# --- LlamaIndex for PDF Reading and Chunking ---
from llama_index.core import SimpleDirectoryReader
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
CHUNK_SIZE = 1024
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
    # This regex finds all characters in the \x00-\x1F range and the DEL character \x7F
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

    if utility.has_collection(COLLECTION_NAME):
        # Drop collection if schema mismatch is detected (simple check)
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
        
        # MODIFIED SCHEMA: Added 'title', 'body', and 'url' fields
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
        
        console.print(f"Reading text from '{PDF_FILE_PATH.name}' using pypdf...")
        reader = PdfReader(PDF_FILE_PATH)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # --- You can add this line to debug and see the extracted text ---
        # console.print("Extracted Text Sample:", full_text[:500])

        console.print("Splitting text into chunks...")
        text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # The splitter directly takes the full text string and returns a list of chunk strings
        bodies = text_splitter.split_text(full_text)

        # Now apply your cleaning function to each chunk
        bodies = [clean_text(text) for text in bodies]

        # The rest of your data preparation remains the same
        pdf_filename = PDF_FILE_PATH.name
        titles: List[str] = [f"{pdf_filename} - Chunk {i+1}" for i in range(len(bodies))]
        urls: List[str] = [str(PDF_FILE_PATH.resolve()) for _ in range(len(bodies))]
        # ----------------------------------------------------

        console.print(f"PDF split into {len(bodies)} chunks.")

        # Generate embeddings (this part remains unchanged)
        console.print("Generating embeddings for all text bodies...")
        embeddings = embedding_model.get_text_embedding_batch(bodies, show_progress=True)

        # MODIFIED: Insert data for all fields into the collection
        console.print(f"Inserting {len(bodies)} records into '{COLLECTION_NAME}'...")
        # The order of lists must match the schema order (excluding the auto-ID 'id' field)
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

    # Load collection into memory
    console.print(f"[bold blue]Loading collection '{COLLECTION_NAME}' into memory...[/]")
    col.load()
    console.print("[green]Collection loaded and ready for queries.[/green]")

    connections.disconnect("default")
    console.print("[bold]Milvus setup complete.[/bold]")

# =========================
# NEW: Query Functions
# =========================

def connect_to_milvus():
    """Helper function to establish Milvus connection."""
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
        return True
    except Exception as e:
        console.print(f"[bold red]Error connecting to Milvus: {e}[/bold red]")
        return False

def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar documents using vector similarity.
    
    Args:
        query (str): The search query text
        top_k (int): Number of top results to return (default: 5)
    
    Returns:
        List[Dict]: List of search results with metadata
    """
    if not connect_to_milvus():
        return []
    
    try:
        # Load collection
        col = Collection(COLLECTION_NAME)
        col.load()
        
        # Generate embedding for the query
        console.print(f"[cyan]Searching for: '{query}'[/cyan]")
        query_embedding = embedding_model.get_query_embedding(query)
        
        # Define search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Perform the search
        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "body", "url"]
        )
        
        # Process results
        search_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "score": hit.score,  # Lower score = better match for L2 distance
                    "title": hit.entity.get("title"),
                    "body": hit.entity.get("body"),
                    "url": hit.entity.get("url")
                }
                search_results.append(result)
        
        connections.disconnect("default")
        return search_results
        
    except Exception as e:
        console.print(f"[bold red]Error during search: {e}[/bold red]")
        connections.disconnect("default")
        return []

def display_search_results(results: List[Dict[str, Any]], max_body_length: int = 200):
    """
    Display search results in a formatted table.
    
    Args:
        results (List[Dict]): Search results from search_documents()
        max_body_length (int): Maximum length of body text to display
    """
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Title", style="green", width=30)
    table.add_column("Content Preview", style="white", width=60)
    
    for i, result in enumerate(results, 1):
        # Truncate body text for display
        body_preview = result["body"][:max_body_length]
        if len(result["body"]) > max_body_length:
            body_preview += "..."
        
        table.add_row(
            str(i),
            f"{result['score']:.4f}",
            result["title"],
            body_preview
        )
    
    console.print(table)

def get_document_by_id(doc_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific document by its ID.
    
    Args:
        doc_id (int): The document ID to retrieve
    
    Returns:
        Dict: Document data or empty dict if not found
    """
    if not connect_to_milvus():
        return {}
    
    try:
        col = Collection(COLLECTION_NAME)
        col.load()
        
        # Query by ID
        results = col.query(
            expr=f"id == {doc_id}",
            output_fields=["title", "body", "url"]
        )
        
        connections.disconnect("default")
        
        if results:
            result = results[0]
            return {
                "id": doc_id,
                "title": result.get("title"),
                "body": result.get("body"),
                "url": result.get("url")
            }
        else:
            console.print(f"[yellow]No document found with ID: {doc_id}[/yellow]")
            return {}
            
    except Exception as e:
        console.print(f"[bold red]Error retrieving document: {e}[/bold red]")
        connections.disconnect("default")
        return {}

def interactive_search():
    """
    Interactive search interface for querying the vector database.
    """
    console.print(Panel.fit("[bold green]Interactive Search Interface[/bold green]\nType 'quit' to exit"))
    
    while True:
        try:
            query = input("\n🔍 Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query:
                console.print("[red]Please enter a valid query.[/red]")
                continue
            
            # Perform search
            results = search_documents(query, top_k=5)
            
            if results:
                display_search_results(results)
                
                # Ask if user wants to see full content of any result
                try:
                    choice = input("\n📄 Enter result number to see full content (or press Enter to continue): ").strip()
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(results):
                            doc = results[idx]
                            console.print(Panel(
                                f"[bold]{doc['title']}[/bold]\n\n{doc['body']}", 
                                title=f"Full Content (Score: {doc['score']:.4f})"
                            ))
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection.[/red]")
            else:
                console.print("[yellow]No results found for your query.[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Search interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error during search: {e}[/bold red]")

def query_with_filter(query: str, title_filter: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search documents with optional title filtering.
    
    Args:
        query (str): The search query text
        title_filter (str): Optional filter for title field (partial match)
        top_k (int): Number of results to return
    
    Returns:
        List[Dict]: Filtered search results
    """
    if not connect_to_milvus():
        return []
    
    try:
        col = Collection(COLLECTION_NAME)
        col.load()
        
        # Generate embedding for the query
        query_embedding = embedding_model.get_query_embedding(query)
        
        # Build filter expression if title filter is provided
        filter_expr = None
        if title_filter:
            # Use LIKE operator for partial matching
            filter_expr = f"title like '%{title_filter}%'"
            console.print(f"[cyan]Applying filter: {filter_expr}[/cyan]")
        
        # Define search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Perform search with optional filter
        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,  # Apply filter if provided
            output_fields=["title", "body", "url"]
        )
        
        # Process results
        search_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "title": hit.entity.get("title"),
                    "body": hit.entity.get("body"),
                    "url": hit.entity.get("url")
                }
                search_results.append(result)
        
        connections.disconnect("default")
        return search_results
        
    except Exception as e:
        console.print(f"[bold red]Error during filtered search: {e}[/bold red]")
        connections.disconnect("default")
        return []

def get_collection_stats():
    """Get basic statistics about the collection."""
    if not connect_to_milvus():
        return
    
    try:
        col = Collection(COLLECTION_NAME)
        col.load()
        
        # Get collection statistics
        num_entities = col.num_entities
        
        console.print(Panel(
            f"Collection: [bold]{COLLECTION_NAME}[/bold]\n"
            f"Total Documents: [cyan]{num_entities}[/cyan]\n"
            f"Embedding Dimension: [magenta]{get_embedding_dim()}[/magenta]",
            title="Collection Statistics"
        ))
        
        connections.disconnect("default")
        
    except Exception as e:
        console.print(f"[bold red]Error getting collection stats: {e}[/bold red]")
        connections.disconnect("default")

# =========================
# Example Usage Functions
# =========================

def example_searches():
    """Run some example searches to demonstrate functionality."""
    console.print(Panel.fit("[bold]Running Example Searches[/bold]"))
    
    example_queries = [
        "cybersecurity threats",
        "annual financial performance", 
        "risk management strategies",
        "technology investments"
    ]
    
    for query in example_queries:
        console.print(f"\n[bold blue]Example Query: '{query}'[/bold blue]")
        results = search_documents(query, top_k=3)
        display_search_results(results, max_body_length=150)

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            setup_milvus_collection()
        elif command == "search":
            interactive_search()
        elif command == "stats":
            get_collection_stats()
        elif command == "examples":
            example_searches()
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[yellow]Available commands: setup, search, stats, examples[/yellow]")
    else:
        # Default: run setup first, then interactive search
        setup_milvus_collection()
        get_collection_stats()
        interactive_search()