# Quick Setup Guide

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Step 1: Get Your API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Create a new API key
4. Copy your API key

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up Neo4j (Graph Database)

Neo4j is required for graph-based RAG functionality.

### Option A: Using Neo4j Desktop (Recommended for local development)

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and start a local database
3. Note your connection credentials (default: `localhost:7687`, username: `neo4j`, password: `password`)

### Option B: Using Docker

```bash
docker run -d --name neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

## Step 4: Set Your API Key and Neo4j Credentials

Create a `.env` file in the project root directory:

```
GROQ_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

Replace the following:
- `your_api_key_here` with your Groq API key from Step 1
- `your_neo4j_password` with your Neo4j database password
- Adjust `NEO4J_URI` and `NEO4J_USERNAME` if your Neo4j setup differs

## Step 5: Run the App

```bash
streamlit run app.py
## Troubleshooting

**Neo4j Connection Issues?**
- Verify Neo4j is running and accessible at the URI specified in your `.env`
- Check that your username and password are correct
- Ensure port 7687 (or your custom port) is not blocked by a firewall

**Groq API Issues?**
- Verify your API key is valid at [console.groq.com](https://console.groq.com)
- Check internet connectivity

```

The app will open in your browser at `http://localhost:8501`

---

That's it! You're ready to go. 🚀
