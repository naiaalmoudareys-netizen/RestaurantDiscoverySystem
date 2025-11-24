"""
Unit Tests for RAG System (Task 1.1)
Tests individual functions and methods in isolation using mocks.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from langchain_core.documents import Document

from rag_system import RestaurantRAGSystem


class TestRestaurantRAGSystem:
    """Unit tests for RestaurantRAGSystem class."""
    
    @pytest.fixture
    def mock_groq_api_key(self):
        """Mock Groq API key for testing."""
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key_123'}):
            yield 'test_key_123'
    
    @pytest.fixture
    def sample_restaurants(self):
        """Sample restaurant data for testing."""
        return [
            {
                "id": 1,
                "name": "Test Italian Restaurant",
                "cuisine": "Italian",
                "location": "Downtown Dubai",
                "price_range": "AED 100-200",
                "description": "A cozy Italian restaurant",
                "amenities": "WiFi, Outdoor Seating",
                "attributes": "Romantic, Family-friendly",
                "opening_hours": "10:00-22:00",
                "rating": 4.5,
                "review_count": 150
            },
            {
                "id": 2,
                "name": "Test Chinese Restaurant",
                "cuisine": "Chinese",
                "location": "Palm Jumeirah",
                "price_range": "AED 50-100",
                "description": "Authentic Chinese cuisine",
                "amenities": "Parking, WiFi",
                "attributes": "Casual",
                "opening_hours": "11:00-23:00",
                "rating": 4.2,
                "review_count": 200
            }
        ]
    
    @pytest.fixture
    def rag_system(self, mock_groq_api_key):
        """Create a RAG system instance with mocked dependencies."""
        with patch('rag_system.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('rag_system.ChatGroq') as mock_llm:
            
            mock_embeddings.return_value = Mock()
            mock_llm.return_value = Mock()
            
            system = RestaurantRAGSystem(
                groq_api_key=mock_groq_api_key,
                persist_directory="./test_chroma_db"
            )
            yield system
    
    def test_init_with_api_key(self, mock_groq_api_key):
        """Test RAG system initialization with API key."""
        with patch('rag_system.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('rag_system.ChatGroq') as mock_llm:
            
            mock_embeddings.return_value = Mock()
            mock_llm.return_value = Mock()
            
            system = RestaurantRAGSystem(groq_api_key=mock_groq_api_key)
            
            assert system.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert system.llm_model == "llama-3.1-8b-instant"
            assert system.vectorstore is None
            mock_embeddings.assert_called_once()
            mock_llm.assert_called_once()
    
    def test_init_without_api_key(self):
        """Test RAG system initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('rag_system.HuggingFaceEmbeddings') as mock_embeddings:
                mock_embeddings.return_value = Mock()
                
                with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
                    RestaurantRAGSystem()
    
    def test_load_restaurants(self, rag_system, sample_restaurants, tmp_path):
        """Test loading restaurants from JSON file."""
        # Create temporary JSON file
        json_file = tmp_path / "restaurants.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_restaurants, f)
        
        # Test loading
        restaurants = rag_system.load_restaurants(str(json_file))
        
        assert len(restaurants) == 2
        assert restaurants[0]['name'] == "Test Italian Restaurant"
        assert restaurants[1]['name'] == "Test Chinese Restaurant"
    
    def test_create_restaurant_documents(self, rag_system, sample_restaurants):
        """Test creating documents from restaurant data."""
        documents = rag_system.create_restaurant_documents(sample_restaurants)
        
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert "Test Italian Restaurant" in documents[0].page_content
        assert documents[0].metadata['id'] == 1
        assert documents[0].metadata['cuisine'] == "Italian"
        assert documents[0].metadata['location'] == "Downtown Dubai"
    
    def test_create_restaurant_documents_metadata(self, rag_system, sample_restaurants):
        """Test that document metadata contains all required fields."""
        documents = rag_system.create_restaurant_documents(sample_restaurants)
        
        required_metadata = ['id', 'name', 'cuisine', 'location', 'price_range', 
                             'rating', 'review_count', 'amenities', 'attributes']
        
        for doc in documents:
            for field in required_metadata:
                assert field in doc.metadata, f"Missing metadata field: {field}"
    
    def test_price_within_range(self, rag_system):
        """Test price range comparison logic."""
        # Test cases: (restaurant_price, max_price, expected)
        # Logic: Returns True if restaurant's upper price <= max_price
        # "AED 100-200" means restaurant prices go up to 200
        test_cases = [
            ("AED 100-200", "AED 200", True),   # 200 <= 200
            ("AED 100-200", "AED 150", False),  # 200 > 150, so False
            ("AED 100-200", "AED 250", True),   # 200 <= 250
            ("AED 100-200", "AED 50", False),   # 200 > 50, so False
            ("AED 50-100", "AED 200", True),    # 100 <= 200
            ("AED 200-300", "AED 150", False),  # 300 > 150, so False
        ]
        
        for restaurant_price, max_price, expected in test_cases:
            result = rag_system._price_within_range(restaurant_price, max_price)
            assert result == expected, \
                f"Failed for {restaurant_price} vs {max_price}: expected {expected}, got {result}"
    
    def test_search_by_attribute(self, rag_system, sample_restaurants):
        """Test searching restaurants by attribute."""
        # Mock vectorstore and collection
        mock_vectorstore = Mock()
        mock_collection = Mock()
        
        # Mock collection.get() to return proper structure
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "id": 1,
                    "name": "Test Italian Restaurant",
                    "cuisine": "Italian",
                    "location": "Downtown Dubai",
                    "price_range": "AED 100-200",
                    "rating": 4.5,
                    "amenities": "WiFi, Parking",
                    "attributes": "Romantic, Family-friendly"
                }
            ],
            "documents": ["Test content"]
        }
        
        # Set up the mock structure
        mock_vectorstore._collection = mock_collection
        rag_system.vectorstore = mock_vectorstore
        
        # Test search - search for "Romantic" in attributes
        results = rag_system.search_by_attribute("Romantic")
        
        assert len(results) == 1
        assert results[0]['name'] == "Test Italian Restaurant"
        mock_collection.get.assert_called_once()
    
    @patch('rag_system.Chroma')
    def test_ingest_data(self, mock_chroma, rag_system, sample_restaurants):
        """Test data ingestion into vector store."""
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        rag_system.ingest_data(sample_restaurants)
        
        assert rag_system.vectorstore is not None
        mock_chroma.from_documents.assert_called_once()
    
    def test_initialize(self, rag_system, sample_restaurants, tmp_path):
        """Test full system initialization."""
        json_file = tmp_path / "restaurants.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_restaurants, f)
        
        with patch.object(rag_system, 'load_restaurants', return_value=sample_restaurants), \
             patch.object(rag_system, 'ingest_data') as mock_ingest, \
             patch.object(rag_system, 'build_rag_chain') as mock_build:
            
            rag_system.initialize(str(json_file))
            
            mock_ingest.assert_called_once_with(sample_restaurants)
            mock_build.assert_called_once()


class TestRAGSystemEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def mock_groq_api_key(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key_123'}):
            yield 'test_key_123'
    
    @pytest.fixture
    def rag_system(self, mock_groq_api_key):
        with patch('rag_system.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('rag_system.ChatGroq') as mock_llm:
            mock_embeddings.return_value = Mock()
            mock_llm.return_value = Mock()
            system = RestaurantRAGSystem(groq_api_key=mock_groq_api_key)
            yield system
    
    def test_load_restaurants_file_not_found(self, rag_system):
        """Test loading restaurants with non-existent file."""
        with pytest.raises(FileNotFoundError):
            rag_system.load_restaurants("non_existent_file.json")
    
    def test_load_restaurants_invalid_json(self, rag_system, tmp_path):
        """Test loading restaurants with invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            rag_system.load_restaurants(str(json_file))
    
    def test_create_restaurant_documents_empty_list(self, rag_system):
        """Test creating documents from empty restaurant list."""
        documents = rag_system.create_restaurant_documents([])
        assert len(documents) == 0
    
    def test_price_within_range_invalid_format(self, rag_system):
        """Test price range comparison with invalid format."""
        # Should handle gracefully or return False
        result = rag_system._price_within_range("Invalid", "AED 200")
        # The method should handle this - check actual implementation behavior
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

