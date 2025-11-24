"""
Task 3.1: API & Deployment Design
REST API for Restaurant Discovery System

Endpoints:
- POST /api/v1/search - Restaurant search using RAG/Agentic system
- POST /api/v1/predict - Rating prediction using ML model
- GET /api/v1/health - Health monitoring
- GET /api/v1/metrics - System metrics

Features:
- Request/Response schemas (Pydantic)
- Error handling
- Rate limiting
- Health monitoring
- API documentation (Swagger/OpenAPI)
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import os
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn

# Rate limiting (simple in-memory implementation)

# Import our systems
from rag_system import RestaurantRAGSystem
from agentic_system import RestaurantAgenticSystem
from rating_prediction_model import RatingPredictionModel
import pandas as pd
import json

# Rate limiting will be handled by check_rate_limit function

# Global system instances (initialized on startup)
rag_system = None
agentic_system = None
ml_model = None
restaurants_df = None
reviews_df = None
user_df = None
trends_df = None

# Metrics tracking
api_metrics = {
    "total_requests": 0,
    "search_requests": 0,
    "predict_requests": 0,
    "errors": 0,
    "avg_latency_ms": 0,
    "start_time": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize systems on startup, cleanup on shutdown."""
    global rag_system, agentic_system, ml_model
    global restaurants_df, reviews_df, user_df, trends_df
    
    print("="*80)
    print("Initializing Restaurant Discovery API...")
    print("="*80)
    
    # Initialize RAG System
    print("\n[1/3] Initializing RAG System...")
    rag_system = RestaurantRAGSystem()
    rag_system.initialize("restaurant.json")
    
    # Initialize Agentic System
    print("\n[2/3] Initializing Agentic System...")
    agentic_system = RestaurantAgenticSystem(rag_system)
    
    # Initialize ML Model
    print("\n[3/3] Loading ML Model...")
    model_path = "models/rating_model.pkl"
    
    # Try to load saved model, otherwise train new one
    if Path(model_path).exists():
        print(f"   Loading saved model from {model_path}...")
        ml_model = RatingPredictionModel.load_model(model_path)
        restaurants_df, reviews_df, user_df, trends_df = ml_model.load_data()
    else:
        print(f"   No saved model found. Training new model...")
        ml_model = RatingPredictionModel()
        restaurants_df, reviews_df, user_df, trends_df = ml_model.load_data()
        X, y = ml_model.create_feature_matrix(restaurants_df, reviews_df, user_df, trends_df)
        ml_model.train(X, y, tune_hyperparameters=False)
        # Save the trained model
        ml_model.save_model(model_path)
    
    api_metrics["start_time"] = datetime.now()
    
    print("\n" + "="*80)
    print("API Server Ready!")
    print("="*80)
    print("\n" + "="*80)
    print("API SERVER STARTED SUCCESSFULLY!")
    print("="*80)
    print("\nAccess the API at:")
    print("  - API Docs (Swagger): http://localhost:8000/docs")
    print("  - API Docs (ReDoc):   http://localhost:8000/redoc")
    print("  - Health Check:       http://localhost:8000/api/v1/health")
    print("\nIMPORTANT: Use 'localhost' or '127.0.0.1' in your browser,")
    print("NOT '0.0.0.0' (0.0.0.0 is only for server binding)")
    print("="*80)
    print("="*80)
    
    yield
    
    # Cleanup on shutdown
    print("\nShutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Restaurant Discovery API",
    description="""
     **Restaurant Discovery System API**
    
    A comprehensive REST API for intelligent restaurant discovery and rating prediction.
    
    ## Features
    
    *  **Semantic Search**: Find restaurants using natural language queries powered by RAG (Retrieval-Augmented Generation)
    *  **Intelligent Agents**: Multi-agent system for understanding queries and generating personalized recommendations
    *  **Rating Prediction**: ML-powered rating predictions based on reviews, trends, and user data
    *  **Health Monitoring**: Real-time system health and performance metrics
    
    ## Quick Start
    
    1. Use the **Search** endpoint to find restaurants by cuisine, location, price, or ambiance
    2. Use the **Predict** endpoint to get rating predictions for specific restaurants
    3. Check **Health** to verify all systems are operational
    4. View **Metrics** to see API performance statistics
    
    ## Authentication
    
    Currently, no authentication is required. Rate limiting is applied per IP address.
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@restaurant-discovery.com"
    },
    license_info={
        "name": "MIT",
    },
    tags_metadata=[
        {
            "name": "Search",
            "description": "Restaurant search endpoints using RAG and agentic systems. Find restaurants by natural language queries.",
        },
        {
            "name": "Prediction",
            "description": "ML model endpoints for rating predictions. Get predicted ratings for restaurants based on reviews and trends.",
        },
        {
            "name": "Monitoring",
            "description": "System health and performance monitoring endpoints.",
        },
        {
            "name": "General",
            "description": "General API information and root endpoints.",
        },
    ],
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiting (for production, use Redis)
from collections import defaultdict
from datetime import timedelta

rate_limit_store = defaultdict(list)
RATE_LIMIT_SEARCH = 30  # requests per minute
RATE_LIMIT_PREDICT = 60  # requests per minute

def get_remote_address(request: Request) -> str:
    """Get client IP address."""
    if request:
        return request.client.host if request.client else "unknown"
    return "unknown"

def check_rate_limit(client_id: str, limit: int) -> bool:
    """Check if client has exceeded rate limit."""
    now = datetime.now()
    # Remove old entries (older than 1 minute)
    rate_limit_store[client_id] = [
        ts for ts in rate_limit_store[client_id]
        if now - ts < timedelta(minutes=1)
    ]
    # Check if limit exceeded
    if len(rate_limit_store[client_id]) >= limit:
        return False
    # Add current request
    rate_limit_store[client_id].append(now)
    return True


# ============================================================================
# Request/Response Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """Request schema for restaurant search."""
    query: str = Field(
        ...,
        description="Natural language search query. Describe what you're looking for (cuisine, location, price, ambiance, amenities).",
        example="Find Italian restaurants in downtown Dubai with outdoor seating",
        min_length=3,
        max_length=500
    )
    thread_id: Optional[str] = Field(
        None,
        description="Optional conversation thread ID for multi-turn conversations. Use the same thread_id to maintain context across multiple queries.",
        example="user_123"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "Find Italian restaurants in downtown Dubai",
                    "thread_id": "user_123"
                },
                {
                    "query": "Show me romantic restaurants with outdoor seating",
                    "thread_id": "user_456"
                },
                {
                    "query": "What are the best rated Chinese restaurants under AED 200?",
                    "thread_id": None
                }
            ]
        }


class RestaurantInfo(BaseModel):
    """Restaurant information schema."""
    name: str = Field(..., description="Restaurant name", example="Central Italian Eatery")
    cuisine: str = Field(..., description="Type of cuisine", example="Italian")
    location: str = Field(..., description="Restaurant location", example="Sharjah")
    price_range: str = Field(..., description="Price range in AED", example="AED 100 - 150")
    rating: Optional[float] = Field(None, description="Average rating (1-5)", example=3.03, ge=1.0, le=5.0)
    amenities: Optional[str] = Field(None, description="Available amenities", example="Wheelchair Accessible")
    attributes: Optional[str] = Field(None, description="Restaurant attributes", example="Vegetarian-Focused, Casual Dining")


class SearchResponse(BaseModel):
    """Response schema for restaurant search."""
    answer: str = Field(..., description="Natural language response with recommendations")
    restaurants: List[RestaurantInfo] = Field(..., description="List of matching restaurants")
    num_results: int = Field(..., description="Number of restaurants found")
    query: str = Field(..., description="Original query")
    latency_ms: float = Field(..., description="Request processing time in milliseconds")


class PredictRequest(BaseModel):
    """Request schema for rating prediction."""
    restaurant_id: int = Field(
        ...,
        description="Unique identifier of the restaurant. Valid range: 1-50.",
        example=1,
        ge=1,
        le=50
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"restaurant_id": 1},
                {"restaurant_id": 5},
                {"restaurant_id": 10}
            ]
        }


class PredictResponse(BaseModel):
    """Response schema for rating prediction."""
    restaurant_id: int
    restaurant_name: str
    predicted_rating: float = Field(..., description="Predicted rating (1-5 scale)")
    actual_rating: Optional[float] = Field(None, description="Actual average rating from reviews (if available)")
    confidence: Optional[str] = Field(None, description="Confidence level")
    latency_ms: float = Field(..., description="Request processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    systems: Dict[str, str] = Field(..., description="Status of each system component")


class MetricsResponse(BaseModel):
    """Response schema for metrics."""
    total_requests: int
    search_requests: int
    predict_requests: int
    errors: int
    avg_latency_ms: float
    uptime_seconds: float


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"], summary="API Information", operation_id="api_root", description="Get basic API information and available endpoints. This is optional - just shows API version and endpoint list.")
async def root():
    """
    Root endpoint with API information.
    
    **Optional endpoint** - Just provides basic info about the API.
    Returns API version and list of available endpoints.
    Not required for core functionality.
    """
    return {
        "message": "Welcome to Restaurant Discovery API",
        "version": "1.0.0",
        "description": "REST API for intelligent restaurant discovery and rating prediction",
        "endpoints": {
            "search": "/api/v1/search",
            "predict": "/api/v1/predict",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search Restaurant",
    operation_id="search_restaurant",
    include_in_schema=True,
    description="""
    Search for restaurants and get personalized recommendations.
    
    **This is the main recommendation endpoint** - It provides restaurant recommendations based on your query.
    
    This endpoint uses:
    - **RAG System** (Task 1.1): Semantic search with embeddings
    - **Agentic Workflow** (Task 1.2): Multi-agent system for query understanding and filtering
    
    **Returns:** Restaurant recommendations with natural language explanations.
    
    **Example Queries:**
    - "Find Italian restaurants in downtown Dubai"
    - "Show me romantic restaurants with outdoor seating"
    - "What are the best rated Chinese restaurants under AED 200?"
    
    **Rate Limit:** 30 requests per minute per IP address
    """,
    response_description="Returns matching restaurants with natural language recommendations"
)
async def search_restaurants(
    request: SearchRequest,
    http_request: Request = None
):
    """
    Search for restaurants using RAG and agentic system.
    
    **Parameters:**
    - **query** (required): Natural language search query describing what you're looking for
    - **thread_id** (optional): Conversation thread ID for maintaining context across multiple queries
    
    **Returns:**
    - Natural language response with recommendations
    - List of matching restaurants with details
    - Number of results found
    - Request processing latency
    """
    start_time = time.time()
    api_metrics["total_requests"] += 1
    api_metrics["search_requests"] += 1
    
    try:
        # Rate limiting
        client_id = get_remote_address(http_request) if http_request else "default"
        if not check_rate_limit(client_id, RATE_LIMIT_SEARCH):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Maximum 30 requests per minute."
            )
        
        if not agentic_system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agentic system not initialized"
            )
        
        # Use thread_id if provided, otherwise use default
        thread_id = request.thread_id or "default"
        
        # Search using agentic system
        result = agentic_system.search(request.query, thread_id=thread_id)
        
        # Format response
        restaurants = []
        for r in result.get("filtered_restaurants", [])[:10]:  # Limit to top 10
            # Validate and convert rating to ensure it's a valid float or None
            rating = r.get("rating")
            if rating is not None:
                try:
                    rating = float(rating)
                    # Ensure rating is within valid range (1.0 to 5.0)
                    if not (1.0 <= rating <= 5.0):
                        rating = None
                except (ValueError, TypeError):
                    rating = None
            
            restaurants.append(RestaurantInfo(
                name=r.get("name", "Unknown"),
                cuisine=r.get("cuisine", "Unknown"),
                location=r.get("location", "Unknown"),
                price_range=r.get("price_range", "Unknown"),
                rating=rating,
                amenities=r.get("amenities"),
                attributes=r.get("attributes")
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        api_metrics["avg_latency_ms"] = (
            (api_metrics["avg_latency_ms"] * (api_metrics["total_requests"] - 1) + latency_ms) 
            / api_metrics["total_requests"]
        )
        
        return SearchResponse(
            answer=result.get("answer", "No restaurants found."),
            restaurants=restaurants,
            num_results=len(restaurants),
            query=request.query,
            latency_ms=round(latency_ms, 2)
        )
    
    except Exception as e:
        api_metrics["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict Restaurant Ratings",
    operation_id="predict_restaurant_ratings",
    include_in_schema=True,
    description="""
    Predict restaurant rating using machine learning model.
    
    This endpoint uses:
    - **ML Model** (Task 2.1): XGBoost model trained on reviews, trends, and user data
    
    The model considers:
    - Review sentiment and content
    - Dining trends and seasonality
    - User demographics and behavior
    - Restaurant features (cuisine, price, amenities)
    
    **Rate Limit:** 60 requests per minute per IP address
    """,
    response_description="Returns predicted rating with confidence level and actual rating (if available)"
)
async def predict_rating(
    request: PredictRequest,
    http_request: Request = None
):
    """
    Predict restaurant rating using ML model.
    
    **Parameters:**
    - **restaurant_id** (required): Unique identifier of the restaurant (1-50)
    
    **Returns:**
    - Predicted rating (1-5 scale)
    - Actual average rating from reviews (if available)
    - Confidence level (High/Medium/Low)
    - Request processing latency
    
    **Note:** Confidence is based on the number of reviews available for the restaurant.
    """
    start_time = time.time()
    api_metrics["total_requests"] += 1
    api_metrics["predict_requests"] += 1
    
    try:
        # Rate limiting
        client_id = get_remote_address(http_request) if http_request else "default"
        if not check_rate_limit(client_id, RATE_LIMIT_PREDICT):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Maximum 60 requests per minute."
            )
        
        if not ml_model or not ml_model.is_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not initialized or trained"
            )
        
        # Find restaurant
        restaurant = restaurants_df[restaurants_df['id'] == request.restaurant_id]
        if restaurant.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Restaurant with ID {request.restaurant_id} not found"
            )
        
        restaurant_name = restaurant.iloc[0]['name']
        
        # Get actual rating from reviews
        restaurant_reviews = reviews_df[reviews_df['restaurant_id'] == request.restaurant_id]
        actual_rating = None
        if not restaurant_reviews.empty:
            actual_rating = round(restaurant_reviews['rating'].mean(), 2)
        
        # Predict rating using the trained model
        predicted_rating = ml_model.predict(
            request.restaurant_id,
            restaurants_df,
            reviews_df,
            user_df,
            trends_df
        )
        predicted_rating = round(float(predicted_rating), 2)
        
        # Determine confidence (simplified)
        if actual_rating:
            confidence = "High" if len(restaurant_reviews) > 10 else "Medium"
        else:
            confidence = "Low"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        api_metrics["avg_latency_ms"] = (
            (api_metrics["avg_latency_ms"] * (api_metrics["total_requests"] - 1) + latency_ms) 
            / api_metrics["total_requests"]
        )
        
        return PredictResponse(
            restaurant_id=request.restaurant_id,
            restaurant_name=restaurant_name,
            predicted_rating=predicted_rating,
            actual_rating=actual_rating,
            confidence=confidence,
            latency_ms=round(latency_ms, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        api_metrics["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health Check",
    operation_id="health_check",
    include_in_schema=True,
    description="""
    Check the health status of all system components.
    
    Returns:
    - Overall service status (healthy/degraded)
    - Individual system status (RAG, Agentic, ML Model)
    - Service uptime
    - Current timestamp
    
    Use this endpoint to verify all systems are operational before making requests.
    """,
    response_description="Returns health status of all system components"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns comprehensive status of all system components including:
    - RAG System status
    - Agentic System status
    - ML Model status
    - Service uptime
    """
    uptime = 0
    if api_metrics["start_time"]:
        uptime = (datetime.now() - api_metrics["start_time"]).total_seconds()
    
    systems = {
        "rag_system": "healthy" if rag_system else "unavailable",
        "agentic_system": "healthy" if agentic_system else "unavailable",
        "ml_model": "healthy" if (ml_model and ml_model.is_trained) else "unavailable"
    }
    
    status_str = "healthy" if all(s == "healthy" for s in systems.values()) else "degraded"
    
    return HealthResponse(
        status=status_str,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2),
        systems=systems
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
    summary="Get API Metrics",
    operation_id="get_api_metrics",
    include_in_schema=True,
    description="""
    Get API performance metrics and statistics.
    
    **Optional endpoint** - For monitoring and debugging.
    
    Returns:
    - Total number of requests processed
    - Breakdown by endpoint (search, predict)
    - Error count
    - Average request latency
    - Service uptime
    
    Useful for monitoring API performance and usage patterns.
    Not required for core functionality.
    """,
    response_description="Returns API performance metrics and statistics"
)
async def get_metrics():
    """
    Get API metrics.
    
    **Optional endpoint** - For monitoring purposes.
    
    Returns comprehensive performance metrics including:
    - Request counts by type
    - Error statistics
    - Performance metrics (latency)
    - Service uptime
    """
    uptime = 0
    if api_metrics["start_time"]:
        uptime = (datetime.now() - api_metrics["start_time"]).total_seconds()
    
    return MetricsResponse(
        total_requests=api_metrics["total_requests"],
        search_requests=api_metrics["search_requests"],
        predict_requests=api_metrics["predict_requests"],
        errors=api_metrics["errors"],
        avg_latency_ms=round(api_metrics["avg_latency_ms"], 2),
        uptime_seconds=round(uptime, 2)
    )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    api_metrics["errors"] += 1
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

