"""
Unit Tests for Agentic System (Task 1.2)
Tests individual functions and methods in isolation using mocks.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agentic_system import (
    RestaurantAgenticSystem, 
    QueryUnderstandingAgent, 
    ResponseGenerationAgent,
    RestaurantRetrievalAgent,
    AgentState
)


class TestQueryUnderstandingAgent:
    """Unit tests for QueryUnderstandingAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock = Mock()
        mock.invoke.return_value.content = '{"cuisine": "Italian", "location": "Downtown Dubai", "price_range": null, "ambiance": null, "amenities": null}'
        return mock
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create agent instance with mocked LLM."""
        return QueryUnderstandingAgent(mock_llm)
    
    def test_init(self, agent, mock_llm):
        """Test agent initialization."""
        assert agent.llm is not None
        assert agent.llm == mock_llm
        assert agent.prompt is not None
    
    def test_extract_entities(self, agent, mock_llm):
        """Test entity extraction from state."""
        state: AgentState = {
            "query": "Find Italian restaurants in downtown Dubai",
            "messages": [],
            "extracted_entities": {},
            "retrieved_restaurants": [],
            "filtered_restaurants": [],
            "final_response": "",
            "conversation_turn": 1
        }
        
        result_state = agent.extract_entities(state)
        
        assert "extracted_entities" in result_state
        entities = result_state["extracted_entities"]
        assert isinstance(entities, dict)
        assert "cuisine" in entities
        mock_llm.invoke.assert_called()
    
    def test_extract_entities_with_json_parsing(self, agent):
        """Test entity extraction handles JSON parsing."""
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = '```json\n{"cuisine": "Italian"}\n```'
        agent.llm = mock_llm
        
        state: AgentState = {
            "query": "Find Italian restaurants",
            "messages": [],
            "extracted_entities": {},
            "retrieved_restaurants": [],
            "filtered_restaurants": [],
            "final_response": "",
            "conversation_turn": 1
        }
        
        result_state = agent.extract_entities(state)
        assert "extracted_entities" in result_state


class TestRestaurantRetrievalAgent:
    """Unit tests for RestaurantRetrievalAgent."""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system."""
        mock = Mock()
        mock.search.return_value = {
            "answer": "Test answer",
            "sources": [
                Document(
                    page_content="Restaurant info",
                    metadata={"id": 1, "name": "Test Restaurant", "cuisine": "Italian"}
                )
            ],
            "num_sources": 1
        }
        return mock
    
    @pytest.fixture
    def agent(self, mock_rag_system):
        """Create agent instance."""
        return RestaurantRetrievalAgent(mock_rag_system)
    
    def test_init(self, agent, mock_rag_system):
        """Test agent initialization."""
        assert agent.rag_system == mock_rag_system
    
    def test_retrieve_and_filter(self, agent, mock_rag_system):
        """Test restaurant retrieval and filtering."""
        state: AgentState = {
            "query": "Find Italian restaurants",
            "messages": [],
            "extracted_entities": {"cuisine": "Italian", "location": None},
            "retrieved_restaurants": [],
            "filtered_restaurants": [],
            "final_response": "",
            "conversation_turn": 1
        }
        
        result_state = agent.retrieve_and_filter(state)
        
        assert "retrieved_restaurants" in result_state
        assert "filtered_restaurants" in result_state
        mock_rag_system.search.assert_called()


class TestResponseGenerationAgent:
    """Unit tests for ResponseGenerationAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock = Mock()
        mock.invoke.return_value.content = "Here are some great Italian restaurants..."
        return mock
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create agent instance with mocked LLM."""
        return ResponseGenerationAgent(mock_llm)
    
    def test_init(self, agent, mock_llm):
        """Test agent initialization."""
        assert agent.llm is not None
        assert agent.llm == mock_llm
        assert agent.prompt is not None
    
    def test_generate_response(self, agent, mock_llm):
        """Test response generation from state."""
        state: AgentState = {
            "query": "Find Italian restaurants",
            "messages": [],
            "extracted_entities": {"cuisine": "Italian"},
            "retrieved_restaurants": [
                {"name": "Test Restaurant", "cuisine": "Italian"}
            ],
            "filtered_restaurants": [
                {"name": "Test Restaurant", "cuisine": "Italian"}
            ],
            "final_response": "",
            "conversation_turn": 1
        }
        
        result_state = agent.generate_response(state)
        
        assert "final_response" in result_state
        assert len(result_state["final_response"]) > 0
        mock_llm.invoke.assert_called()


class TestRestaurantAgenticSystem:
    """Unit tests for RestaurantAgenticSystem."""
    
    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system."""
        mock = Mock()
        mock.search.return_value = {
            'answer': 'Test answer',
            'sources': [Document(page_content="Test", metadata={"name": "Test Restaurant"})],
            'num_sources': 1
        }
        return mock
    
    @pytest.fixture
    def agentic_system(self, mock_rag_system):
        """Create agentic system with mocked RAG."""
        with patch('agentic_system.ChatGroq') as mock_chat_groq:
            mock_chat_groq.return_value = Mock()
            system = RestaurantAgenticSystem(mock_rag_system)
            return system
    
    def test_init(self, agentic_system, mock_rag_system):
        """Test system initialization."""
        assert agentic_system.rag_system == mock_rag_system
        assert agentic_system.query_agent is not None
        assert agentic_system.retrieval_agent is not None
        assert agentic_system.response_agent is not None
        assert agentic_system.app is not None  # Compiled graph
        assert agentic_system.memory is not None  # Memory checkpointer
    
    def test_search_new_conversation(self, agentic_system, mock_rag_system):
        """Test search with new conversation thread."""
        query = "Find Italian restaurants"
        thread_id = "test_thread_1"
        
        # Mock the app (compiled graph) execution
        with patch.object(agentic_system, 'app') as mock_app:
            mock_app.invoke.return_value = {
                "final_response": "Here are Italian restaurants...",
                "extracted_entities": {"cuisine": "Italian"},
                "retrieved_restaurants": [{"name": "Test Restaurant"}],
                "filtered_restaurants": [{"name": "Test Restaurant"}],
                "conversation_turn": 1
            }
            
            result = agentic_system.search(query, thread_id=thread_id)
            
            assert 'answer' in result
            assert 'filtered_restaurants' in result
            assert result['answer'] == "Here are Italian restaurants..."
            mock_app.invoke.assert_called_once()
    
    def test_continue_conversation(self, agentic_system):
        """Test continuing an existing conversation."""
        thread_id = "test_thread_1"
        
        # Mock the app (compiled graph) execution
        with patch.object(agentic_system, 'app') as mock_app:
            mock_app.invoke.return_value = {
                "final_response": "Here are downtown Italian restaurants...",
                "extracted_entities": {"cuisine": "Italian", "location": "downtown"},
                "retrieved_restaurants": [{"name": "Test Restaurant"}],
                "filtered_restaurants": [{"name": "Test Restaurant"}],
                "conversation_turn": 2
            }
            
            result = agentic_system.continue_conversation("What about downtown?", thread_id=thread_id)
            
            assert 'answer' in result
            assert result['answer'] == "Here are downtown Italian restaurants..."
            mock_app.invoke.assert_called_once()


class TestAgenticSystemEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def mock_rag_system(self):
        mock = Mock()
        return mock
    
    @pytest.fixture
    def agentic_system(self, mock_rag_system):
        with patch('agentic_system.ChatGroq') as mock_chat_groq:
            mock_chat_groq.return_value = Mock()
            system = RestaurantAgenticSystem(mock_rag_system)
            return system
    
    def test_search_with_empty_query(self, agentic_system):
        """Test search with empty query."""
        with patch.object(agentic_system, 'app') as mock_app:
            mock_app.invoke.return_value = {
                "final_response": "Please provide a query.",
                "extracted_entities": {},
                "retrieved_restaurants": [],
                "filtered_restaurants": [],
                "conversation_turn": 1
            }
            
            result = agentic_system.search("", thread_id="test")
            
            assert 'answer' in result
            assert result['answer'] == "Please provide a query."
    
    def test_continue_conversation_without_existing_thread(self, agentic_system):
        """Test continuing conversation without existing thread."""
        # Should create new conversation or handle gracefully (LangGraph memory handles this)
        with patch.object(agentic_system, 'app') as mock_app:
            mock_app.invoke.return_value = {
                "final_response": "Response",
                "extracted_entities": {},
                "retrieved_restaurants": [],
                "filtered_restaurants": [],
                "conversation_turn": 1
            }
            
            result = agentic_system.continue_conversation("Query", thread_id="new_thread")
            
            assert 'answer' in result
            assert result['answer'] == "Response"
            mock_app.invoke.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

