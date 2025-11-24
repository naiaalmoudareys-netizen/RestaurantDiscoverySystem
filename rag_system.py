"""
RAG System for Restaurant Discovery
Task 1.1: RAG System Architecture & Implementation

Production-ready RAG system using:
- Groq: Fast LLM inference (llama-3.1-70b-versatile)
- HuggingFace: Free embeddings (sentence-transformers/all-MiniLM-L6-v2)
- LangChain: RAG orchestration
- ChromaDB: Vector database
"""

import json
import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


class RestaurantRAGSystem:
    """
    Production-ready RAG system for restaurant discovery.
    
    Architecture:
    - Vector DB: ChromaDB (persistent, efficient)
    - Embeddings: HuggingFace sentence-transformers (FREE, no API key needed)
    - LLM: Groq (llama-3.1-70b-versatile) - Fast inference
    - Framework: LangChain (orchestration and chain management)
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "llama-3.1-8b-instant",
                 persist_directory: str = "./chroma_db",
                 groq_api_key: str = None):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: HuggingFace embedding model name (free, no API key needed)
            llm_model: Groq LLM model name
            persist_directory: Directory to persist ChromaDB
            groq_api_key: Groq API key (if not provided, uses GROQ_API_KEY env var)
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory
        
        # Initialize FREE HuggingFace embeddings (no API key needed!)
        print("   Initializing FREE HuggingFace embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use CPU (can change to 'cuda' if GPU available)
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
        
        # Initialize Groq LLM for fast inference
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file or pass as parameter.")
        
        print("   Initializing Groq LLM for fast inference...")
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model=llm_model,
            temperature=0
        )
        
        # Vector store (will be initialized after data ingestion)
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
    def load_restaurants(self, json_path: str = "restaurant.json") -> List[Dict]:
        """
        Load restaurant data from JSON file.
        
        Args:
            json_path: Path to restaurant JSON file
            
        Returns:
            List of restaurant dictionaries
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            restaurants = json.load(f)
        
        print(f"[OK] Loaded {len(restaurants)} restaurants from {json_path}")
        return restaurants
    
    def create_restaurant_documents(self, restaurants: List[Dict]) -> List[Document]:
        """
        Convert restaurant data into LangChain Documents for embedding.
        
        Creates rich text representations that include all searchable information.
        
        Args:
            restaurants: List of restaurant dictionaries
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for restaurant in restaurants:
            # Create a comprehensive text representation
            text_parts = [
                f"Restaurant Name: {restaurant['name']}",
                f"Cuisine Type: {restaurant['cuisine']}",
                f"Location: {restaurant['location']}",
                f"Price Range: {restaurant['price_range']}",
                f"Description: {restaurant['description']}",
                f"Amenities: {restaurant['amenities']}",
                f"Attributes: {restaurant['attributes']}",
                f"Opening Hours: {restaurant['opening_hours']}",
                f"Rating: {restaurant['rating']}/5.0",
                f"Review Count: {restaurant['review_count']}"
            ]
            
            text = "\n".join(text_parts)
            
            # Create document with metadata for filtering
            doc = Document(
                page_content=text,
                metadata={
                    "id": restaurant['id'],
                    "name": restaurant['name'],
                    "cuisine": restaurant['cuisine'],
                    "location": restaurant['location'],
                    "price_range": restaurant['price_range'],
                    "rating": restaurant['rating'],
                    "review_count": restaurant['review_count'],
                    "amenities": restaurant['amenities'],
                    "attributes": restaurant['attributes']
                }
            )
            documents.append(doc)
        
        print(f"[OK] Created {len(documents)} documents from restaurants")
        return documents
    
    def ingest_data(self, restaurants: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Ingest restaurant data into vector database.
        
        Implements the data pipeline:
        1. Convert restaurants to documents
        2. Split into chunks (if needed)
        3. Generate embeddings using HuggingFace
        4. Store in ChromaDB
        
        Args:
            restaurants: List of restaurant dictionaries
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        # Convert to documents
        documents = self.create_restaurant_documents(restaurants)
        
        # For restaurant data, we typically don't need chunking since each restaurant
        # is already a complete unit. However, we'll use a text splitter for flexibility.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Split documents (may not split if already small enough)
        split_docs = text_splitter.split_documents(documents)
        
        print(f"[OK] Split into {len(split_docs)} document chunks")
        print("   Generating embeddings (this may take 30-60 seconds)...")
        
        # Create vector store with embeddings
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # ChromaDB 0.4+ automatically persists, no need to call persist()
        
        print(f"[OK] Ingested {len(split_docs)} documents into ChromaDB at {self.persist_directory}")
        
        # Initialize retriever with hybrid search capabilities
        # Using regular similarity search (threshold can cause issues with normalized embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Top 5 most similar results
        )
        
        print("[OK] Vector store and retriever initialized")
    
    def build_rag_chain(self):
        """
        Build the RAG chain for query answering using LangChain 1.0 LCEL.
        
        Creates a retrieval-augmented generation chain that:
        1. Retrieves relevant restaurants based on query
        2. Provides context to LLM
        3. Generates contextual, natural language responses using Groq
        """
        # Custom prompt template for restaurant recommendations
        prompt_template = """You are a helpful restaurant discovery assistant. Use the following pieces of context about restaurants to answer the user's query.

Context (restaurant information):
{context}

User Query: {question}

Instructions:
1. Analyze the user's query to understand their requirements (cuisine, location, price, amenities, etc.)
2. From the provided context, identify restaurants that match the user's criteria
3. Provide a natural, conversational response with:
   - Restaurant names that match the criteria
   - Key details (location, price range, cuisine, amenities)
   - Why each restaurant is a good match
4. If no restaurants match exactly, suggest the closest alternatives
5. Be specific about what matches and what doesn't

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain using LangChain 1.0 LCEL
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        print("[OK] RAG chain built and ready")
    
    def search_by_attribute(self, attribute_value: str) -> List[Dict[str, Any]]:
        """
        Search for restaurants by a specific attribute value (e.g., "Romantic").
        Uses ChromaDB collection to get all documents and filter by metadata.
        Case-insensitive matching - works with "Romantic", "romantic", "ROMANTIC", etc.
        
        Args:
            attribute_value: The attribute value to search for (e.g., "Romantic" or "romantic")
            
        Returns:
            List of restaurant dictionaries matching the attribute (case-insensitive)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        try:
            # Access ChromaDB collection directly to get all documents
            # ChromaDB stores documents with metadata, we can query all and filter
            collection = self.vectorstore._collection
            
            # Get all documents from the collection
            results = collection.get(include=["metadatas", "documents"])
            
            matching_restaurants = []
            seen_names = set()
            
            # Iterate through all documents
            if results and "metadatas" in results:
                for i, metadata in enumerate(results["metadatas"]):
                    attributes = metadata.get("attributes", "")
                    # Check if the attribute value is in the attributes string (case-insensitive)
                    if attributes and attribute_value.lower() in str(attributes).lower():
                        name = metadata.get("name", "Unknown")
                        if name not in seen_names:
                            seen_names.add(name)
                            # Get rating and ensure it's a valid float or None
                            rating = metadata.get("rating")
                            if rating is not None:
                                try:
                                    rating = float(rating)
                                    # Validate rating is within valid range
                                    if not (1.0 <= rating <= 5.0):
                                        rating = None
                                except (ValueError, TypeError):
                                    rating = None
                            
                            matching_restaurants.append({
                                "name": name,
                                "cuisine": metadata.get("cuisine", "Unknown"),
                                "location": metadata.get("location", "Unknown"),
                                "price_range": metadata.get("price_range", "Unknown"),
                                "rating": rating,
                                "amenities": metadata.get("amenities", "Unknown"),
                                "attributes": metadata.get("attributes", "Unknown")
                            })
            
            return matching_restaurants
        except Exception as e:
            print(f"[WARNING] Attribute search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Search for restaurants using RAG.
        
        Implements semantic retrieval with contextual LLM responses.
        
        Args:
            query: Natural language query (e.g., "Find Italian restaurants in downtown Dubai with outdoor seating under AED 200")
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("RAG chain not built. Call build_rag_chain() first.")
        
        try:
            # Get source documents first for metadata (LangChain 1.0 uses invoke)
            source_docs = self.retriever.invoke(query)
            
            # Execute RAG query
            answer = self.qa_chain.invoke(query)
            
            # Deduplicate restaurants by name to avoid showing same restaurant multiple times
            seen_names = set()
            unique_restaurants = []
            for doc in source_docs:
                name = doc.metadata.get("name", "Unknown")
                if name not in seen_names:
                    seen_names.add(name)
                    # Get rating and ensure it's a valid float or None
                    rating = doc.metadata.get("rating")
                    if rating is not None:
                        try:
                            rating = float(rating)
                            # Validate rating is within valid range
                            if not (1.0 <= rating <= 5.0):
                                rating = None
                        except (ValueError, TypeError):
                            rating = None
                    
                    unique_restaurants.append({
                        "name": name,
                        "cuisine": doc.metadata.get("cuisine", "Unknown"),
                        "location": doc.metadata.get("location", "Unknown"),
                        "price_range": doc.metadata.get("price_range", "Unknown"),
                        "rating": rating,
                        "amenities": doc.metadata.get("amenities", "Unknown"),
                        "attributes": doc.metadata.get("attributes", "Unknown")  # Include attributes for ambiance filtering
                    })
            
            return {
                "answer": answer,
                "source_restaurants": unique_restaurants,
                "num_sources": len(unique_restaurants)
            }
        except Exception as e:
            # Edge case handling - print full error for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"\n[ERROR] Search failed: {str(e)}")
            print(f"[DEBUG] Full error:\n{error_details}\n")
            
            return {
                "answer": f"I encountered an error while searching: {str(e)}. Please try rephrasing your query.",
                "source_restaurants": [],
                "num_sources": 0,
                "error": str(e),
                "error_details": error_details
            }
    
    def hybrid_search(self, query: str, 
                     cuisine_filter: Optional[str] = None,
                     location_filter: Optional[str] = None,
                     max_price: Optional[str] = None) -> Dict[str, Any]:
        """
        Hybrid search combining semantic search with metadata filtering.
        
        This provides better results by:
        1. Using semantic search for natural language understanding
        2. Applying metadata filters for precise matching
        
        Args:
            query: Natural language query
            cuisine_filter: Filter by cuisine type
            location_filter: Filter by location
            max_price: Maximum price range (e.g., "AED 200")
            
        Returns:
            Dictionary with filtered results
        """
        # First, get semantic search results
        base_results = self.search(query)
        
        # Apply metadata filters if provided
        filtered_restaurants = base_results["source_restaurants"]
        
        if cuisine_filter:
            filtered_restaurants = [
                r for r in filtered_restaurants 
                if cuisine_filter.lower() in r.get("cuisine", "").lower()
            ]
        
        if location_filter:
            filtered_restaurants = [
                r for r in filtered_restaurants 
                if location_filter.lower() in r.get("location", "").lower()
            ]
        
        if max_price:
            # Parse price range and filter
            filtered_restaurants = [
                r for r in filtered_restaurants
                if self._price_within_range(r.get("price_range", ""), max_price)
            ]
        
        return {
            "answer": base_results["answer"],
            "source_restaurants": filtered_restaurants,
            "num_sources": len(filtered_restaurants),
            "filters_applied": {
                "cuisine": cuisine_filter,
                "location": location_filter,
                "max_price": max_price
            }
        }
    
    def _price_within_range(self, price_range: str, max_price: str) -> bool:
        """
        Check if price range is within maximum price.
        
        Args:
            price_range: Restaurant price range (e.g., "AED 150 - 200")
            max_price: Maximum price (e.g., "AED 200")
            
        Returns:
            True if price is within range
        """
        try:
            # Extract numeric values
            max_price_num = int(max_price.replace("AED", "").strip())
            
            # Parse price range (e.g., "AED 150 - 200" -> 200)
            if "-" in price_range:
                upper = int(price_range.split("-")[-1].replace("AED", "").replace("+", "").strip())
                return upper <= max_price_num
            return True
        except (ValueError, AttributeError, TypeError):
            # If parsing fails, include the restaurant
            return True
    
    def initialize(self, json_path: str = "restaurant.json"):
        """
        Complete initialization: load data, ingest, and build chain.
        
        Args:
            json_path: Path to restaurant JSON file
        """
        print("Initializing RAG System...")
        print("   Using Groq for fast LLM inference")
        print("   Using FREE HuggingFace embeddings (no API key needed!)")
        print()
        
        # Load restaurants
        restaurants = self.load_restaurants(json_path)
        
        # Ingest data
        self.ingest_data(restaurants)
        
        # Build RAG chain
        self.build_rag_chain()
        
        print()
        print("[OK] RAG System fully initialized and ready!")


def main():
    """Example usage of the RAG system."""
    
    # Check for Groq API key
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        print("⚠️  Warning: GROQ_API_KEY not found in environment variables")
        print("Please set your Groq API key in a .env file or environment variable")
        print("Note: No API key needed for embeddings (using free HuggingFace)")
        return
    
    # Initialize system
    rag_system = RestaurantRAGSystem(
        llm_model="llama-3.1-8b-instant"  # Fast Groq model (current)
    )
    rag_system.initialize("restaurant.json")
    
    # Example queries
    test_queries = [
        "Find Italian restaurants in downtown Dubai with outdoor seating under AED 200 per person",
        "Show me romantic restaurants in Palm Jumeirah",
        "What are the best rated Chinese restaurants in Al Barsha?",
        "Find vegetarian-friendly restaurants with live music"
    ]
    
    print("\n" + "="*80)
    print("Testing RAG System with Sample Queries")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        result = rag_system.search(query)
        print(f"Answer: {result['answer']}")
        print(f"Found {result['num_sources']} matching restaurants")
        print("\n")


if __name__ == "__main__":
    main()

