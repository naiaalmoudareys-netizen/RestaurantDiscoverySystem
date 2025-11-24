# Restaurant Discovery System

A complete restaurant discovery and recommendation system implementing:
- **Task 1.1**: RAG System Architecture & Implementation
- **Task 1.2**: Agentic Workflow with Multi-Agent System
- **Task 2.1**: ML Rating Prediction Model
- **Task 3.1**: REST API & Deployment

## System Overview

This system provides intelligent restaurant discovery using:
- **RAG (Retrieval-Augmented Generation)**: Semantic search with natural language understanding
- **Agentic Workflow**: Multi-agent system for conversational restaurant discovery
- **ML Model**: XGBoost-based rating prediction using restaurant features, reviews, and trends
- **REST API**: FastAPI-based API with full documentation

## Technology Stack

- **Groq**: Fast LLM inference (10x faster than OpenAI)
- **HuggingFace**: FREE embeddings (no API key needed!)
- **LangChain**: RAG framework and orchestration
- **LangGraph**: Multi-agent workflow orchestration
- **ChromaDB**: Vector database for embeddings
- **XGBoost**: ML model for rating prediction
- **FastAPI**: REST API framework
- **Python 3.8+**: Runtime environment

## Features

-  **Semantic Search**: Natural language queries with semantic understanding
-  **Hybrid Search**: Combines semantic search with metadata filtering
- **Contextual Responses**: LLM-generated natural language recommendations
-  **Production-Ready**: Persistent storage, error handling, edge cases
-  **100% Free Embeddings**: HuggingFace runs locally - no API key needed!
-  **Fast LLM**: Groq provides 10x faster inference than OpenAI

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection (for Groq API calls and initial HuggingFace model download)
- ~80MB disk space (for HuggingFace model - one-time download)
- Groq API key (free tier available at https://console.groq.com)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download HuggingFace model (~80MB) - this is a one-time download.

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
```

Or set as environment variable:

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_groq_key_here"
```

**Windows (CMD):**
```cmd
set GROQ_API_KEY=your_groq_key_here
```

**Linux/Mac:**
```bash
export GROQ_API_KEY=your_groq_key_here
```

**Note**: 
- **Embeddings**: Uses **FREE HuggingFace embeddings** (no API key needed!)
- **LLM**: Uses Groq for fast inference (API key required, but has generous free tier)

### 3. Required Data Files

Make sure you have the following data files in the project root:
- `restaurant.json` - Restaurant data (50 restaurants)
- `reviews.csv` - Restaurant reviews data
- `user_data.csv` - User profile data
- `dining_trends.csv` - Dining trends data

These files are required for:
- RAG System: `restaurant.json`
- ML Model: All CSV files + `restaurant.json`

### 4. Running the System

#### Option A: Run Individual Components

**RAG System Only (Task 1.1):**
```bash
python rag_system.py
```

**Agentic System (Task 1.2):**
```bash
python agentic_system.py
```

**ML Model (Task 2.1):**
```bash
python rating_prediction_model.py
```

#### Option B: Run Complete API Server (Recommended)

This initializes all systems and provides a REST API:

```bash
python api_server.py
```

The server will:
1. Initialize RAG System (Task 1.1)
2. Initialize Agentic System (Task 1.2)
3. Load or train ML Model (Task 2.1)
4. Start API server on http://localhost:8000

## Usage

### Basic Usage

```python
from rag_system import RestaurantRAGSystem

# Initialize system
rag = RestaurantRAGSystem(
    llm_model="llama-3.1-70b-versatile"  # Fast Groq model
)
rag.initialize("restaurant.json")

# Search for restaurants
result = rag.search("Find Italian restaurants in downtown Dubai with outdoor seating under AED 200")
print(result["answer"])
```

### Hybrid Search with Filters

```python
# Search with metadata filters
result = rag.hybrid_search(
    query="romantic restaurants",
    cuisine_filter="Italian",
    location_filter="Downtown Dubai",
    max_price="AED 200"
)
```

## Example Queries

- "Find Italian restaurants in downtown Dubai with outdoor seating under AED 200 per person"
- "Show me romantic restaurants in Palm Jumeirah"
- "What are the best rated Chinese restaurants in Al Barsha?"
- "Find vegetarian-friendly restaurants with live music"

## Architecture

The system architecture consists of four main components:

1. **RAG System (Task 1.1)**: Vector-based semantic search with ChromaDB and HuggingFace embeddings
2. **Agentic System (Task 1.2)**: Multi-agent workflow using LangGraph for conversational interactions
3. **ML Model (Task 2.1)**: XGBoost regression model for rating prediction
4. **REST API (Task 3.1)**: FastAPI server that integrates all components

All components are integrated through the API server, which initializes and manages the lifecycle of each system.

## Project Structure

```
.
├── api_server.py              # FastAPI server (Task 3.1)
├── rag_system.py              # RAG system implementation (Task 1.1)
├── agentic_system.py          # Agentic workflow system (Task 1.2)
├── rating_prediction_model.py # ML rating prediction model (Task 2.1)
├── test_all_tasks.py          # Complete system test suite
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── Data Files:
├── restaurant.json            # Restaurant data (50 restaurants)
├── reviews.csv                # Restaurant reviews data
├── user_data.csv              # User profile data
├── dining_trends.csv          # Dining trends data
│
├── Generated Directories:
├── chroma_db/                 # ChromaDB persistent storage (created on first run)
└── models/                    # Saved ML models (created on first run)
    └── rating_model.pkl       # Trained rating prediction model
```

## System Components

### Task 1.1: RAG System
- Semantic search using HuggingFace embeddings
- Hybrid search with metadata filtering
- Natural language responses using Groq LLM
- Persistent vector storage with ChromaDB

### Task 1.2: Agentic Workflow
- Multi-agent system using LangGraph
- Conversational restaurant discovery
- Context-aware follow-up queries
- Thread-based conversation management

### Task 2.1: Rating Prediction Model
- XGBoost-based regression model
- Features: restaurant attributes, review statistics, user preferences, dining trends
- Model persistence and loading
- Feature importance analysis

### Task 3.1: REST API
- FastAPI-based RESTful API
- Swagger/OpenAPI documentation
- Health monitoring and metrics
- Error handling and validation

## Performance

- **First Run**: ~60-90 seconds (downloads HuggingFace model, generates embeddings)
- **Subsequent Runs**: ~5-10 seconds (uses cached model and embeddings)
- **Query Latency**: ~1-3 seconds (embedding + retrieval + Groq generation)
- **Cost**: $0 for embeddings + minimal for LLM (Groq has generous free tier)

## Testing

### Unit Tests

**Install test dependencies:**
```bash
pip install -r requirements.txt
```

**Run all unit tests:**
```bash
# Windows (PowerShell/CMD)
python -m pytest

# Linux/Mac
pytest
```

**Run specific test file:**
```bash
# Windows
python -m pytest test_rag_system.py -v
python -m pytest test_rating_prediction_model.py -v
python -m pytest test_agentic_system.py -v

# Linux/Mac
pytest test_rag_system.py -v
pytest test_rating_prediction_model.py -v
pytest test_agentic_system.py -v
```

**Run with coverage:**
```bash
# Windows
python -m pytest --cov=. --cov-report=html

# Linux/Mac
pytest --cov=. --cov-report=html
```

**Note:** On Windows, use `python -m pytest` instead of just `pytest` to ensure pytest runs in the correct Python environment.

**Unit test files:**
- `test_rag_system.py` - Tests for RAG System (Task 1.1)
- `test_rating_prediction_model.py` - Tests for ML Model (Task 2.1)
- `test_agentic_system.py` - Tests for Agentic System (Task 1.2)
- `conftest.py` - Shared pytest fixtures and configuration

**Note:** Unit tests use mocks and don't require the API server to be running. They test individual components in isolation.

### Integration/System Tests

**Prerequisites:** API server must be running

1. Start the API server in one terminal:
```bash
python api_server.py
```

2. Run the integration test suite in another terminal:
```bash
python test_all_tasks.py
```

This will test:
-  Task 1.1: RAG System (via API)
-  Task 1.2: Agentic Workflow (via API)
-  Task 2.1: Rating Prediction Model (via API)
-  Task 3.1: REST API endpoints

The test script will provide a detailed summary of all test results.

### Test Individual Components (Manual)

**RAG System:**
```bash
python rag_system.py
```

**Agentic System:**
```bash
python agentic_system.py
```

**ML Model:**
```bash
python rating_prediction_model.py
```

## API Server (Task 3.1)

### Starting the API Server

```bash
python api_server.py
```

The server will:
- Initialize RAG System (Task 1.1)
- Initialize Agentic System (Task 1.2)
- Load and train ML Model (Task 2.1)
- Start API server on http://localhost:8000

**IMPORTANT**: Use `http://localhost:8000` or `http://127.0.0.1:8000` in your browser.
Do NOT use `http://0.0.0.0:8000` - that won't work!

**Expected Output:**
```
================================================================================
Initializing Restaurant Discovery API...
================================================================================

[1/3] Initializing RAG System...
[2/3] Initializing Agentic System...
[3/3] Loading ML Model...
================================================================================
API Server Ready!
================================================================================

API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/api/v1/health
================================================================================
```

### Testing the API

**Option A: Use the test script**
```bash
# In a new terminal
python test_all_tasks.py
```

**Option B: Use the interactive docs**
Open your browser: http://localhost:8000/docs

**Option C: Use cURL**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Find Italian restaurants"}'

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1}'
```

### API Endpoints

**POST `/api/v1/search`** - Restaurant search using RAG/Agentic system
- Request body: `{"query": "your search query", "thread_id": "optional_thread_id"}`
- Returns: Search results with restaurants and natural language answer
- Uses: Task 1.1 (RAG) and Task 1.2 (Agentic workflow)

**POST `/api/v1/predict`** - Rating prediction using ML model
- Request body: `{"restaurant_id": 1}`
- Returns: Predicted rating, confidence, and actual rating (if available)
- Uses: Task 2.1 (ML model)

**GET `/api/v1/health`** - Health check endpoint
- Returns: System status, uptime, and component health
- No authentication required

**GET `/api/v1/metrics`** - API metrics and statistics
- Returns: Request counts, error counts, average latency, uptime
- Useful for monitoring API performance

### API Documentation

- **Swagger UI (Interactive)**: http://localhost:8000/docs
  - Try endpoints directly from your browser
  - See request/response schemas
  - Test all endpoints interactively

- **ReDoc (Alternative Docs)**: http://localhost:8000/redoc
  - Alternative documentation format

- **Full API Documentation**: See `API_DOCUMENTATION.md` (if available)
  - Detailed endpoint descriptions
  - Request/response examples
  - Error codes and handling

### Troubleshooting

**Port already in use:**
```bash
# Windows PowerShell
$env:PORT=8001; python api_server.py

# Windows CMD
set PORT=8001 && python api_server.py

# Linux/Mac
PORT=8001 python api_server.py
```

**API not responding:**
- Check if server is running (look for "API Server Ready!" message)
- Verify GROQ_API_KEY is set correctly in `.env` file or environment
- Check console for error messages
- Ensure all data files exist (restaurant.json, reviews.csv, user_data.csv, dining_trends.csv)

**Slow responses:**
- First request may be slow (model initialization, ~30-60 seconds)
- Subsequent requests should be faster (~1-3 seconds)
- First API startup takes longer (initializes all systems)

**Model training on first run:**
- If `models/rating_model.pkl` doesn't exist, the system will train a new model
- This takes ~1-2 minutes on first API startup
- The model is saved for future runs

**Connection errors:**
- Ensure Groq API key is valid
- Check internet connection
- Verify firewall isn't blocking requests

**Import errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher
- Consider using a virtual environment

## Usage Examples

### Using the API

Once the API server is running, you can interact with it in several ways:

**1. Interactive API Documentation:**
- Open http://localhost:8000/docs in your browser
- Try endpoints directly from the Swagger UI

**2. Python Client:**
```python
import requests

# Search for restaurants
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={"query": "Find Italian restaurants in downtown Dubai", "thread_id": "user123"}
)
result = response.json()
print(result['answer'])

# Predict rating
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"restaurant_id": 1}
)
result = response.json()
print(f"Predicted rating: {result['predicted_rating']}/5.0")
```

**3. Command Line (cURL):**
See the "Testing the API" section above for cURL examples.

### Using Components Directly

**RAG System:**
```python
from rag_system import RestaurantRAGSystem

rag = RestaurantRAGSystem()
rag.initialize("restaurant.json")
result = rag.search("Find Italian restaurants")
print(result["answer"])
```

**Agentic System:**
```python
from rag_system import RestaurantRAGSystem
from agentic_system import RestaurantAgenticSystem

rag = RestaurantRAGSystem()
rag.initialize("restaurant.json")
agentic = RestaurantAgenticSystem(rag)

# First query
result = agentic.search("Find romantic restaurants", thread_id="user123")

# Follow-up query
result = agentic.continue_conversation("What about Italian ones?", thread_id="user123")
```

**ML Model:**
```python
from rating_prediction_model import RatingPredictionModel

model = RatingPredictionModel()
restaurants_df, reviews_df, user_df, trends_df = model.load_data()
X, y = model.create_feature_matrix(restaurants_df, reviews_df, user_df, trends_df)
results = model.train(X, y)

# Predict rating for a restaurant
prediction = model.predict(restaurant_id=1, restaurants_df=restaurants_df, 
                          reviews_df=reviews_df, user_df=user_df, trends_df=trends_df)
```

## Performance Notes

- **First Run**: ~60-90 seconds (downloads HuggingFace model, generates embeddings)
- **Subsequent Runs**: ~5-10 seconds (uses cached model and embeddings)
- **Query Latency**: ~1-3 seconds (embedding + retrieval + Groq generation)
- **API Startup**: ~30-60 seconds (initializes all systems, may train ML model)
- **Cost**: $0 for embeddings + minimal for LLM (Groq has generous free tier)

## Important Notes

- First run downloads HuggingFace model (~80MB) - this is a one-time download
- Embeddings are generated locally (no external API calls for embeddings)
- ChromaDB persists embeddings for faster subsequent runs
- ML model is saved after first training for faster subsequent API startups
- Groq provides fast inference with generous free tier
- All data files must be present in the project root directory

