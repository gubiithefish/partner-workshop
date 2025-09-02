# file: delete_collection.py

import os
from dotenv import load_dotenv
from rich.console import Console

# --- Milvus admin ---
from pymilvus import connections, utility

# =========================
# Load environment & setup
# =========================
load_dotenv()
console = Console()

# =========================
# Config (from your .env file)
# =========================
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")

# =========================
# Main Deletion Logic
# =========================
def main():
    """
    Connects to Milvus and drops the specified collection.
    """
    if not all([MILVUS_HOST, MILVUS_PORT, MILVUS_PASSWORD, COLLECTION_NAME]):
        console.print("[bold red]Error: Milvus environment variables are not set in your .env file.[/]")
        return

    console.print(f"[bold]Attempting to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...[/]")
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

    # Check if the collection exists
    if utility.has_collection(COLLECTION_NAME):
        console.print(f"[yellow]Found existing collection: '[bold]{COLLECTION_NAME}[/bold]'.[/yellow]")
        
        # Drop the collection
        console.print(f"Proceeding to drop the collection...")
        utility.drop_collection(COLLECTION_NAME)
        console.print(f"[bold green]Successfully dropped collection '{COLLECTION_NAME}'.[/bold green]")
    else:
        console.print(f"[cyan]Collection '[bold]{COLLECTION_NAME}[/bold]' does not exist. Nothing to do.[/cyan]")
    
    connections.disconnect("default")

if __name__ == "__main__":
    main()