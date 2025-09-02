# ITM8 - hackathon

## Steps to connect Watsonx.data with Orchestrate

---

## Prerequisites

- Python 3.11+
- A running Milvus (GRPC) endpoint on wx.data
- An **IAM API key** with access to your Milvus instance

```bash
# (recommended) create & activate a virtual env
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

```

## Setup .env file

- Create a .env at the project root and fill the values for Milvus connection.

## Data ingestion to Milvus

```bash
python ingestion.py
```

## Query data from Milvus

```bash
python query.py
```

### Note: The above script uses the following methods since Orchestrate expects this format from Watsonx.data

- While connecting to wx.data Milvus, please ensure you use the GRPC details and set secure=True (this enables SSL authentication with the API key).
- WXO uses `collection.search()` under the hood. Therfore, data is inserted via `collection.insert()`.
- After inserting data into Milvus, it is essential to explicitly load the collection into memory using `collection.load()` since WXO expects the data to be available in memory.
