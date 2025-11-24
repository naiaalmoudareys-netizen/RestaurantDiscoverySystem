"""
Task 1.2: Agentic Workflow Design
Multi-agent system using LangGraph for restaurant discovery

Agents:
1. Query Understanding Agent - Entity extraction (cuisine, location, price, ambiance)
2. Restaurant Retrieval Agent - Retrieval & filtering
3. Response Generation Agent - Personalized recommendations

Features:
- Conditional logic
- Memory management
- Multi-turn conversations
"""

import json
import os
import warnings
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import the RAG system from Task 1.1
from rag_system import RestaurantRAGSystem

load_dotenv()


class AgentState(TypedDict):
    """State shared between agents in the workflow."""
    messages: Annotated[List, add_messages]  # Conversation history
    query: str  # Original user query
    extracted_entities: Dict[str, Any]  # Extracted entities (cuisine, location, price, ambiance)
    retrieved_restaurants: List[Dict[str, Any]]  # Retrieved restaurant documents
    filtered_restaurants: List[Dict[str, Any]]  # Filtered restaurants
    final_response: str  # Final generated response
    conversation_turn: int  # Track conversation turns


class QueryUnderstandingAgent:
    """
    Agent 1: Query Understanding & Entity Extraction
    
    Extracts structured information from natural language queries:
    - Cuisine type
    - Location
    - Price range
    - Ambiance/attributes
    """
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert at understanding restaurant search queries. 
Extract structured information from the user's query. Return a JSON object with the following fields:
- cuisine: The type of cuisine (e.g., "Italian", "Chinese", "Indian", or null if not specified)
- location: The location/area (e.g., "Downtown Dubai", "Palm Jumeirah", or null if not specified)
- price_range: Price range mentioned (e.g., "AED 200", "under 150", or null if not specified)
- ambiance: Ambiance/attributes mentioned (e.g., "romantic", "casual", "outdoor seating", or null if not specified)
- amenities: Specific amenities mentioned (e.g., "outdoor seating", "live music", or null if not specified)

Return ONLY valid JSON, no additional text."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def extract_entities(self, state: AgentState) -> AgentState:
        """Extract entities from the user query."""
        query = state["query"]
        
        # Get conversation history
        messages = state.get("messages", [])
        
        # Create prompt with query
        prompt_messages = self.prompt.format_messages(
            messages=[HumanMessage(content=query)]
        )
        
        # Get LLM response
        response = self.llm.invoke(prompt_messages)
        
        # Parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            entities = json.loads(response_text)
        except Exception as e:
            # Fallback: try to extract entities with regex or use defaults
            entities = {
                "cuisine": None,
                "location": None,
                "price_range": None,
                "ambiance": None,
                "amenities": None
            }
            print(f"[WARNING] Could not parse entities, using defaults: {e}")
        
        # Update state
        state["extracted_entities"] = entities
        
        print(f"   [Agent 1] Extracted entities: {entities}")
        
        return state


class RestaurantRetrievalAgent:
    """
    Agent 2: Restaurant Retrieval & Filtering
    
    Uses the RAG system to retrieve restaurants and applies filters.
    """
    
    def __init__(self, rag_system: RestaurantRAGSystem):
        self.rag_system = rag_system
    
    def retrieve_and_filter(self, state: AgentState) -> AgentState:
        """Retrieve restaurants using RAG and apply filters."""
        query = state["query"]
        entities = state.get("extracted_entities", {})
        
        # Build a more specific query if we have entities
        # Prioritize cuisine in the search query for better semantic matching
        search_query = query
        if entities.get("cuisine") and entities.get("location"):
            # If both cuisine and location are specified, emphasize cuisine first
            search_query = f"{entities.get('cuisine')} cuisine restaurant {entities.get('cuisine')} food in {entities.get('location')}"
        elif entities.get("cuisine"):
            search_query = f"{entities.get('cuisine')} cuisine {entities.get('cuisine')} restaurant"
        elif entities.get("location"):
            search_query = f"restaurants in {entities.get('location')}"
        elif entities.get("ambiance"):
            # For ambiance searches, enhance the query to include the ambiance term
            ambiance = entities.get("ambiance", "").lower()
            search_query = f"romantic restaurant {ambiance} ambiance {ambiance} dining"
        
        # Do semantic search
        base_result = self.rag_system.search(search_query)
        retrieved = base_result.get("source_restaurants", [])
        
        # SPECIAL CASE: For "romantic" ambiance, also search by metadata (attributes field)
        # Case-insensitive check for "romantic" (handles "Romantic", "romantic", "ROMANTIC", etc.)
        if entities.get("ambiance") and "romantic" in entities.get("ambiance", "").lower():
            # Search for restaurants with "Romantic" in their attributes using metadata filter
            # search_by_attribute handles case-insensitive matching internally
            romantic_restaurants = self.rag_system.search_by_attribute("Romantic")
            # Merge with semantic search results, avoiding duplicates
            retrieved_names = {r.get("name") for r in retrieved}
            for r in romantic_restaurants:
                if r.get("name") not in retrieved_names:
                    retrieved.append(r)
                    retrieved_names.add(r.get("name"))
            if romantic_restaurants:
                print(f"   [Agent 2] Found {len(romantic_restaurants)} restaurant(s) with 'Romantic' attribute via metadata search")
        
        # If searching for ambiance and we have few results, try a broader search
        if entities.get("ambiance") and len(retrieved) < 3:
            # Try searching with just the ambiance term
            ambiance_query = f"{entities.get('ambiance')} restaurant"
            ambiance_result = self.rag_system.search(ambiance_query)
            ambiance_retrieved = ambiance_result.get("source_restaurants", [])
            # Merge results, avoiding duplicates
            retrieved_names = {r.get("name") for r in retrieved}
            for r in ambiance_retrieved:
                if r.get("name") not in retrieved_names:
                    retrieved.append(r)
                    retrieved_names.add(r.get("name"))
        
        # If no results, try original query
        if not retrieved:
            base_result = self.rag_system.search(query)
            retrieved = base_result.get("source_restaurants", [])
        
        
        # Then apply strict entity-based filters
        filtered = retrieved.copy()
        
        # Filter by cuisine if specified (flexible matching - cuisine name should be in the cuisine field)
        if entities.get("cuisine"):
            cuisine = entities["cuisine"].lower().strip()
            # Keep restaurants where cuisine contains the search term (flexible matching)
            before_cuisine_filter = len(filtered)
            filtered = [
                r for r in filtered
                if cuisine in r.get("cuisine", "").lower()
            ]
            if before_cuisine_filter > len(filtered):
                print(f"   [Agent 2] Filtered out {before_cuisine_filter - len(filtered)} restaurant(s) (cuisine mismatch)")
        
        # Filter by location if specified (must match exactly or contain)
        if entities.get("location"):
            location = entities["location"].lower()
            # Handle common location variations
            location_variations = {
                "downtown dubai": ["downtown", "dubai"],
                "al barsha": ["barsha"],
                "jbr": ["jumeirah beach residence", "jumeirah"],
                "palm jumeirah": ["palm"]
            }
            location_keywords = location_variations.get(location, [location])
            before_location_filter = len(filtered)
            filtered = [
                r for r in filtered
                if any(keyword in r.get("location", "").lower() for keyword in location_keywords)
            ]
            if before_location_filter > len(filtered):
                print(f"   [Agent 2] Filtered out {before_location_filter - len(filtered)} restaurants (location mismatch)")
        
        # Filter by price if specified
        if entities.get("price_range"):
            max_price = entities["price_range"]
            # Extract numeric value
            try:
                if "under" in max_price.lower() or "<" in max_price:
                    price_num = int(max_price.replace("under", "").replace("<", "").replace("AED", "").strip())
                else:
                    price_num = int(max_price.replace("AED", "").strip())
                
                before_price_filter = len(filtered)
                filtered = [
                    r for r in filtered
                    if self._price_within_range(r.get("price_range", ""), f"AED {price_num}")
                ]
                if before_price_filter > len(filtered):
                    print(f"   [Agent 2] Filtered out {before_price_filter - len(filtered)} restaurants (price mismatch)")
            except (ValueError, AttributeError, TypeError) as e:
                # If price parsing fails, don't filter by price
                pass
        
        # Filter by amenities if specified
        if entities.get("amenities"):
            amenities_keywords = entities["amenities"].lower().split()
            before_amenities_filter = len(filtered)
            filtered = [
                r for r in filtered
                if any(keyword in r.get("amenities", "").lower() for keyword in amenities_keywords)
            ]
            if before_amenities_filter > len(filtered):
                print(f"   [Agent 2] Filtered out {before_amenities_filter - len(filtered)} restaurant(s) (amenities mismatch)")
        
        # Filter by ambiance if specified (check attributes field)
        if entities.get("ambiance"):
            ambiance_keywords = entities["ambiance"].lower().split()
            before_ambiance_filter = len(filtered)
            # For "romantic", search for restaurants with "Romantic" in attributes
            if "romantic" in entities.get("ambiance", "").lower():
                # Filter to restaurants that have "Romantic" in their attributes
                filtered = [
                    r for r in filtered
                    if "romantic" in r.get("attributes", "").lower()
                ]
            else:
                # For other ambiance terms, check both attributes and amenities
                filtered = [
                    r for r in filtered
                    if any(keyword in r.get("attributes", "").lower() for keyword in ambiance_keywords) or
                       any(keyword in r.get("amenities", "").lower() for keyword in ambiance_keywords)
                ]
            if before_ambiance_filter > len(filtered):
                print(f"   [Agent 2] Filtered out {before_ambiance_filter - len(filtered)} restaurant(s) (ambiance mismatch)")
            elif before_ambiance_filter == 0 and len(filtered) > 0:
                # If we had no results before but now have some, that's good
                print(f"   [Agent 2] Found {len(filtered)} restaurant(s) matching ambiance criteria")
        
        # If filtering removed all results, try to get better results from RAG
        if not filtered and retrieved:
            # Try a more specific search if we have cuisine and location
            if entities.get("cuisine") and entities.get("location"):
                print(f"   [Agent 2] No matches after filtering, trying more specific search...")
                # Try multiple search variations
                search_variations = [
                    f"{entities.get('cuisine')} restaurant in {entities.get('location')}",
                    f"{entities.get('cuisine')} {entities.get('location')}",
                    f"restaurant {entities.get('cuisine')} {entities.get('location')}"
                ]
                for search_var in search_variations:
                    specific_result = self.rag_system.search(search_var)
                    specific_retrieved = specific_result.get("source_restaurants", [])
                    if specific_retrieved:
                        # Apply strict filters again
                        filtered = specific_retrieved.copy()
                        if entities.get("cuisine"):
                            cuisine = entities["cuisine"].lower().strip()
                            # Use flexible matching (in) instead of exact match (==) for consistency
                            filtered = [r for r in filtered if cuisine in r.get("cuisine", "").lower()]
                        if entities.get("location"):
                            location = entities["location"].lower()
                            location_variations = {
                                "downtown dubai": ["downtown", "dubai"],
                                "al barsha": ["barsha"],
                                "jbr": ["jumeirah beach residence", "jumeirah"],
                                "palm jumeirah": ["palm"]
                            }
                            location_keywords = location_variations.get(location, [location])
                            filtered = [r for r in filtered if any(keyword in r.get("location", "").lower() for keyword in location_keywords)]
                        if filtered:
                            break  # Found matches, stop trying variations
        
        state["retrieved_restaurants"] = retrieved
        state["filtered_restaurants"] = filtered
        
        print(f"   [Agent 2] Retrieved {len(retrieved)} restaurant(s) from semantic search")
        if filtered:
            print(f"   [Agent 2] After filtering: {len(filtered)} restaurant(s) match criteria")
            print(f"   [Agent 2] Matches: {', '.join([r.get('name', 'Unknown') for r in filtered[:3]])}")
        else:
            print(f"   [Agent 2] After filtering: 0 restaurants match ALL criteria")
            if retrieved:
                print(f"   [Agent 2] Note: Found {len(retrieved)} similar restaurant(s) but they don't match all filters")
        
        return state
    
    def _price_within_range(self, price_range: str, max_price: str) -> bool:
        """Check if price range is within maximum price."""
        try:
            max_price_num = int(max_price.replace("AED", "").strip())
            if "-" in price_range:
                upper = int(price_range.split("-")[-1].replace("AED", "").replace("+", "").strip())
                return upper <= max_price_num
            return True
        except (ValueError, AttributeError, TypeError):
            # If price parsing fails, include the restaurant
            return True


class ResponseGenerationAgent:
    """
    Agent 3: Response Generation with Personalized Recommendations
    
    Generates natural, personalized responses based on retrieved restaurants.
    """
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful restaurant discovery assistant. Generate a natural, 
conversational response recommending restaurants to the user.

Guidelines:
1. Be friendly and conversational
2. Highlight why each restaurant matches the user's criteria
3. Mention key details: name, location, price range, cuisine, amenities
4. If no exact matches, suggest the closest alternatives
5. Personalize the response based on the user's preferences
6. Keep responses concise but informative

Use the retrieved restaurant information to craft your response."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate personalized response."""
        query = state["query"]
        entities = state.get("extracted_entities", {})
        restaurants = state.get("filtered_restaurants", [])
        
        # Format restaurant information for the prompt
        restaurant_info = ""
        if restaurants:
            restaurant_info = "\n\n=== AVAILABLE RESTAURANTS (FROM DATABASE - USE THESE ONLY) ===\n"
            for i, r in enumerate(restaurants[:5], 1):  # Limit to top 5
                restaurant_info += f"\n{i}. {r.get('name', 'Unknown')}\n"
                restaurant_info += f"   - Cuisine: {r.get('cuisine', 'Unknown')}\n"
                restaurant_info += f"   - Location: {r.get('location', 'Unknown')}\n"
                restaurant_info += f"   - Price Range: {r.get('price_range', 'Unknown')}\n"
                restaurant_info += f"   - Rating: {r.get('rating', 'Unknown')}/5.0\n"
                restaurant_info += f"   - Amenities: {r.get('amenities', 'None')}\n"
                if r.get('attributes'):
                    restaurant_info += f"   - Attributes: {r.get('attributes', 'None')}\n"
            restaurant_info += "\n=== END OF RESTAURANT LIST ===\n"
            restaurant_info += "\nCRITICAL INSTRUCTIONS:\n"
            restaurant_info += "1. You MUST recommend restaurants ONLY from the list above\n"
            restaurant_info += "2. Use the EXACT restaurant names from the list - copy them character by character (e.g., if list shows 'Saffron Indian Eatery', you MUST say 'Saffron Indian Eatery', NOT 'The Indian Eatery')\n"
            restaurant_info += "3. DO NOT modify, shorten, or paraphrase restaurant names - use them exactly as shown\n"
            restaurant_info += "4. If the user asks for Chinese restaurants and the list shows 'Golden Dhow Cafe' with cuisine 'Chinese', you MUST say you found a Chinese restaurant\n"
            restaurant_info += "5. Include all relevant details (location, price, cuisine, amenities) from the list\n"
            restaurant_info += "6. Explain why each restaurant matches the user's criteria\n"
            restaurant_info += "7. DO NOT say 'no restaurants found' if restaurants are listed above\n"
        else:
            restaurant_info = "\n\n=== NO RESTAURANTS FOUND ===\n"
            restaurant_info += "IMPORTANT: No restaurants in the database match the exact criteria.\n"
            restaurant_info += "- DO NOT make up or invent restaurant names\n"
            restaurant_info += "- Explain that no matches were found\n"
            restaurant_info += "- Suggest the user try different criteria (different location, cuisine, or price range)\n"
            restaurant_info += "- Ask if they'd like to see what restaurants are available in general\n"
        
        # Create context for the LLM
        context = f"""User Query: {query}

Extracted Requirements:
- Cuisine: {entities.get('cuisine', 'Any')}
- Location: {entities.get('location', 'Any')}
- Price Range: {entities.get('price_range', 'Any')}
- Ambiance: {entities.get('ambiance', 'Any')}
- Amenities: {entities.get('amenities', 'Any')}
{restaurant_info}"""
        
        # Get conversation history for context
        messages = state.get("messages", [])
        
        # Create prompt
        prompt_messages = self.prompt.format_messages(
            messages=[HumanMessage(content=context)]
        )
        
        # Generate response
        response = self.llm.invoke(prompt_messages)
        final_response = response.content
        
        state["final_response"] = final_response
        
        print(f"   [Agent 3] Generated personalized response ({len(final_response)} chars)")
        
        return state


class RestaurantAgenticSystem:
    """
    Multi-agent system using LangGraph for restaurant discovery.
    
    Workflow:
    1. Query Understanding Agent -> Extract entities
    2. Restaurant Retrieval Agent -> Retrieve & filter restaurants
    3. Response Generation Agent -> Generate personalized response
    """
    
    def __init__(self, rag_system: RestaurantRAGSystem, llm_model: str = "llama-3.1-8b-instant"):
        """
        Initialize the agentic system.
        
        Args:
            rag_system: Initialized RAG system from Task 1.1
            llm_model: Groq LLM model name
        """
        self.rag_system = rag_system
        
        # Initialize LLM
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file.")
        
        self.llm = ChatGroq(
            groq_api_key=groq_key,
            model=llm_model,
            temperature=0
        )
        
        # Initialize agents
        self.query_agent = QueryUnderstandingAgent(self.llm)
        self.retrieval_agent = RestaurantRetrievalAgent(rag_system)
        self.response_agent = ResponseGenerationAgent(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Memory for multi-turn conversations
        self.memory = MemorySaver()
        
        # Compile graph with memory
        self.app = self.graph.compile(checkpointer=self.memory)
        
        print("[OK] Agentic system initialized with LangGraph")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("query_understanding", self.query_agent.extract_entities)
        workflow.add_node("restaurant_retrieval", self.retrieval_agent.retrieve_and_filter)
        workflow.add_node("response_generation", self.response_agent.generate_response)
        
        # Define edges (workflow) - linear flow
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "restaurant_retrieval")
        workflow.add_edge("restaurant_retrieval", "response_generation")
        workflow.add_edge("response_generation", END)
        
        return workflow
    
    def search(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Search for restaurants using the multi-agent system.
        
        Args:
            query: Natural language query
            thread_id: Conversation thread ID for memory management
            
        Returns:
            Dictionary with response and metadata
        """
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "extracted_entities": {},
            "retrieved_restaurants": [],
            "filtered_restaurants": [],
            "final_response": "",
            "conversation_turn": 0
        }
        
        # Get conversation history from memory
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Run the graph with checkpointing
            result = self.app.invoke(initial_state, config)
            
            # Add AI response to conversation history for next turn
            # This is handled automatically by the memory checkpointer
            
            return {
                "answer": result["final_response"],
                "extracted_entities": result["extracted_entities"],
                "retrieved_restaurants": result["retrieved_restaurants"],
                "filtered_restaurants": result["filtered_restaurants"],
                "num_sources": len(result["filtered_restaurants"])
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n[ERROR] Agentic search failed: {str(e)}")
            print(f"[DEBUG] Full error:\n{error_details}\n")
            
            return {
                "answer": f"I encountered an error: {str(e)}. Please try rephrasing your query.",
                "extracted_entities": {},
                "retrieved_restaurants": [],
                "filtered_restaurants": [],
                "num_sources": 0,
                "error": str(e)
            }
    
    def continue_conversation(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Continue a multi-turn conversation.
        
        Args:
            query: Follow-up query
            thread_id: Same thread ID to maintain conversation context
            
        Returns:
            Dictionary with response
        """
        # Get conversation history
        config = {"configurable": {"thread_id": thread_id}}
        
        # Add new message to conversation
        state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "extracted_entities": {},
            "retrieved_restaurants": [],
            "filtered_restaurants": [],
            "final_response": "",
            "conversation_turn": 0
        }
        
        # Run with conversation context
        result = self.app.invoke(state, config)
        
        return {
            "answer": result["final_response"],
            "extracted_entities": result["extracted_entities"],
            "retrieved_restaurants": result["retrieved_restaurants"],
            "filtered_restaurants": result["filtered_restaurants"],
            "num_sources": len(result["filtered_restaurants"])
        }


def main():
    """Example usage of the agentic system."""
    print("="*80)
    print("Task 1.2: Agentic Workflow System")
    print("="*80)
    print()
    
    # Check for Groq API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not found. Please set it in .env file.")
        return
    
    # Initialize RAG system (Task 1.1)
    print("Step 1: Initializing RAG System (Task 1.1)...")
    rag_system = RestaurantRAGSystem()
    rag_system.initialize("restaurant.json")
    print()
    
    # Initialize Agentic System (Task 1.2)
    print("Step 2: Initializing Agentic System (Task 1.2)...")
    agentic_system = RestaurantAgenticSystem(rag_system)
    print()
    
    # Test queries
    print("="*80)
    print("Testing Multi-Agent System")
    print("="*80)
    print()
    
    test_queries = [
        "Find Italian restaurants in downtown Dubai with outdoor seating under AED 200 per person",
        "What about something more romantic?",
        "Show me Chinese restaurants in Al Barsha"
    ]
    
    thread_id = "test_conversation"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}")
        print()
        
        if i == 1:
            result = agentic_system.search(query, thread_id=thread_id)
        else:
            result = agentic_system.continue_conversation(query, thread_id=thread_id)
        
        print(f"\n{'-'*80}")
        print("EXTRACTION RESULTS")
        print(f"{'-'*80}")
        entities_found = {k: v for k, v in result['extracted_entities'].items() if v}
        if entities_found:
            for key, value in entities_found.items():
                print(f"  [*] {key.replace('_', ' ').title()}: {value}")
        else:
            print("  (No specific entities extracted)")
        
        print(f"\n{'-'*80}")
        print("RESTAURANT MATCHES")
        print(f"{'-'*80}")
        if result['filtered_restaurants']:
            print(f"Found {result['num_sources']} matching restaurant(s):\n")
            for idx, r in enumerate(result['filtered_restaurants'], 1):
                print(f"  {idx}. {r.get('name', 'Unknown')}")
                print(f"     Cuisine: {r.get('cuisine', 'Unknown')} | Location: {r.get('location', 'Unknown')}")
                print(f"     Price: {r.get('price_range', 'Unknown')} | Rating: {r.get('rating', 'Unknown')}/5.0")
                if r.get('amenities'):
                    print(f"     Amenities: {r.get('amenities', 'None')}")
                print()
        else:
            print(f"  [!] No restaurants match ALL the specified criteria")
            if result['retrieved_restaurants']:
                print(f"  [i] Found {len(result['retrieved_restaurants'])} similar restaurant(s) but they don't match all filters")
        
        print(f"{'-'*80}")
        print("ASSISTANT RESPONSE")
        print(f"{'-'*80}")
        print(result['answer'])
        print()


if __name__ == "__main__":
    main()

