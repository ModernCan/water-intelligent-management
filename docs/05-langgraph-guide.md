# LangGraph Implementation Guide: Orchestrating Workflows for Water Networks

This guide will walk you through implementing LangGraph to orchestrate complex, stateful workflows for your Water Network Management System. Building on the previous Neo4j graph database and GraphRAG capabilities, we'll now add the crucial orchestration layer that will maintain context and coordinate multi-step reasoning processes.

## 1. Understanding LangGraph

### What is LangGraph?

LangGraph is a framework for building stateful, graph-based workflows with Large Language Models (LLMs). It allows you to:

1. **Define Reasoning Steps**: Break complex tasks into discrete steps
2. **Maintain State**: Preserve context across multiple interactions
3. **Orchestrate Workflows**: Control the flow of information through different processing nodes
4. **Implement Conditional Logic**: Create branches based on analysis results
5. **Handle Cycles**: Support iterative refinement

Unlike simple prompt-response patterns, LangGraph creates persistent reasoning graphs where information flows through multiple processing stages, with state maintained throughout the interaction.

### Key Concepts in LangGraph

- **State**: A dictionary-like object that holds the current context of a workflow
- **Nodes**: Functions that process state and return updates
- **Edges**: Connections that define how information flows between nodes
- **Conditions**: Logic that determines which path to take in the graph
- **Cycles**: Loops in the graph that allow for iterative refinement

### Why LangGraph for Water Networks?

Water network operations involve complex, multi-step reasoning processes that benefit from LangGraph:

1. **Maintenance Planning**: Requires analysis of component history, risk assessment, and resource allocation
2. **Emergency Response**: Needs rapid assessment of failure impacts, isolation strategies, and repair planning
3. **Capital Planning**: Demands analysis of aging infrastructure, performance data, and budget constraints
4. **Operational Optimization**: Requires simulation of different operational scenarios and parameter adjustments
5. **Knowledge Management**: Benefits from connecting user queries to different information retrieval and analysis methods

## 2. Setting Up Your LangGraph Environment

### Prerequisites

- Python 3.9+ environment
- Neo4j database with water network model (from previous guides)
- GraphRAG implementation (from previous guide)

### Installation

```bash
# Install LangGraph and supporting libraries
pip install langgraph
pip install langchain langchain-openai langchain-community
pip install neo4j pandas numpy matplotlib

# Install any other dependencies you need
pip install python-dotenv pydantic typing-extensions
```

### Basic Project Structure

Organize your project with this structure:

```
water_network_management/
├── config/
│   └── settings.py           # Configuration variables
├── database/
│   └── neo4j_connector.py    # Neo4j connection functions
├── retrieval/
│   ├── graph_retrieval.py    # Graph-based retrieval functions
│   └── vector_retrieval.py   # Vector-based retrieval functions
├── orchestration/
│   ├── states.py             # State definitions for workflows
│   ├── nodes.py              # Node functions for graph processing
│   └── workflows.py          # Workflow graph definitions
├── utils/
│   ├── formatters.py         # Output formatting helpers
│   └── validators.py         # Input validation functions
└── main.py                   # Application entry point
```

## 3. Designing Workflow States for Water Network Management

The first step in implementing LangGraph is to define the states that will be tracked throughout your workflows.

### Core State Definition

```python
# In orchestration/states.py
from typing import TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field

class WaterNetworkState(TypedDict):
    """Base state for water network workflows"""
    query: str  # Original user query
    parsed_query: Dict[str, Any]  # Structured representation of the query
    retrieval_results: Dict[str, Any]  # Results from retrieval systems
    context: str  # Synthesized context for LLM
    response: Optional[str]  # Generated response
    conversation_history: List[Dict[str, str]]  # History of interactions
    current_workflow: str  # Identifier for the active workflow
    error: Optional[str]  # Any error information

class MaintenanceState(WaterNetworkState):
    """State for maintenance planning workflows"""
    components_to_maintain: List[Dict[str, Any]]  # Components needing maintenance
    maintenance_history: Dict[str, List[Dict[str, Any]]]  # Historical maintenance records
    risk_assessment: Dict[str, float]  # Risk scores for components
    recommended_actions: List[Dict[str, Any]]  # Recommended maintenance actions
    resource_requirements: Dict[str, Any]  # Required resources for maintenance

class EmergencyResponseState(WaterNetworkState):
    """State for emergency response workflows"""
    affected_component: Dict[str, Any]  # The component that failed
    isolation_strategy: List[Dict[str, Any]]  # Valves to close for isolation
    affected_areas: List[Dict[str, Any]]  # Areas impacted by the failure
    customer_impact: Dict[str, Any]  # Details on customer impact
    restoration_plan: List[Dict[str, Any]]  # Steps to restore service
    estimated_time: str  # Estimated time to resolution
```

### Utility Functions for State Management

```python
# In orchestration/states.py
def initialize_state(query: str, workflow_type: str) -> WaterNetworkState:
    """Initialize a new state for a workflow"""
    base_state = WaterNetworkState(
        query=query,
        parsed_query={},
        retrieval_results={},
        context="",
        response=None,
        conversation_history=[],
        current_workflow=workflow_type,
        error=None
    )
    
    if workflow_type == "maintenance":
        return MaintenanceState(
            **base_state,
            components_to_maintain=[],
            maintenance_history={},
            risk_assessment={},
            recommended_actions=[],
            resource_requirements={}
        )
    elif workflow_type == "emergency":
        return EmergencyResponseState(
            **base_state,
            affected_component={},
            isolation_strategy=[],
            affected_areas=[],
            customer_impact={},
            restoration_plan=[],
            estimated_time=""
        )
    else:
        return base_state

def update_state(current_state: WaterNetworkState, updates: Dict[str, Any]) -> WaterNetworkState:
    """Update state with new values"""
    new_state = current_state.copy()
    for key, value in updates.items():
        if key in new_state:
            new_state[key] = value
    return new_state
```

## 4. Implementing Core Node Functions

Next, define the node functions that will process state and generate updates. Let's focus on some of the most important nodes:

### Query Understanding Node

```python
# In orchestration/nodes.py
from .states import WaterNetworkState, update_state
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

# Initialize LLM
llm = ChatOpenAI(temperature=0)

def parse_query_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Parse the user query to understand intent and extract entities"""
    query = state["query"]
    
    # Define a prompt for query understanding
    prompt_template = """
    You are an AI assistant for a water utility company. Parse the following query about a water network and identify:
    1. The query intent
    2. Any component types mentioned
    3. Specific component IDs mentioned
    4. Any attributes or properties referred to
    5. Any relationships or connections of interest
    6. Date ranges or time periods
    7. Geographic or zone information
    
    Query: {query}
    
    Provide your analysis in JSON format with the following structure:
    {{
        "intent": "one of [component_info, flow_path, maintenance_history, failure_impact, zone_analysis, component_comparison, sensor_data, isolation_strategy]",
        "component_types": ["list of component types mentioned"],
        "component_ids": ["list of specific component IDs mentioned"],
        "attributes": ["list of attributes mentioned"],
        "relationships": ["list of relationships mentioned"],
        "time_period": "any time period mentioned",
        "location": "any geographic area or zone mentioned"
    }}
    
    JSON response:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query"]
    )
    
    # Generate the response
    response = llm.invoke(prompt.format(query=query))
    
    # Parse JSON response
    try:
        parsed_response = json.loads(response.content)
    except json.JSONDecodeError:
        # If JSON parsing fails, provide a basic fallback
        parsed_response = {
            "intent": "component_info",
            "component_types": [],
            "component_ids": [],
            "attributes": [],
            "relationships": [],
            "time_period": "",
            "location": ""
        }
    
    # Update state
    return {"parsed_query": parsed_response}
```

### Retrieval Orchestration Node

```python
# In orchestration/nodes.py
from retrieval.graph_retrieval import WaterNetworkTraversal
from retrieval.vector_retrieval import VectorRetriever
from typing import Dict, Any, List, Tuple

# Initialize retrieval components (assuming you've implemented these based on previous guides)
graph_traversal = WaterNetworkTraversal()
vector_retriever = VectorRetriever()

def retrieval_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Coordinate retrieval from different sources based on query understanding"""
    parsed_query = state["parsed_query"]
    
    # Determine retrieval strategies based on intent
    strategies = determine_retrieval_strategies(parsed_query)
    
    # Execute retrieval strategies
    retrieval_results = {}
    for strategy_type, params in strategies:
        result = execute_retrieval_strategy(strategy_type, params)
        retrieval_results[strategy_type] = result
    
    return {"retrieval_results": retrieval_results}

def determine_retrieval_strategies(parsed_query: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Determine which retrieval strategies to use based on query"""
    intent = parsed_query.get("intent", "component_info")
    component_ids = parsed_query.get("component_ids", [])
    component_types = parsed_query.get("component_types", [])
    
    strategies = []
    
    # Different retrieval strategies based on intent
    if intent == "maintenance_history":
        if component_ids:
            strategies.append(("maintenance_history", {"component_ids": component_ids}))
        strategies.append(("vector", {"query_terms": ["maintenance", "history"] + component_types}))
    
    elif intent == "flow_path":
        if len(component_ids) >= 2:
            strategies.append(("path_finding", {"source_id": component_ids[0], "target_id": component_ids[1]}))
        elif len(component_ids) == 1:
            strategies.append(("connected_components", {"component_id": component_ids[0]}))
    
    elif intent == "failure_impact":
        if component_ids:
            strategies.append(("failure_impact", {"component_id": component_ids[0]}))
    
    # Add a default vector retrieval for all queries
    query_terms = component_types + component_ids
    if intent:
        query_terms.append(intent.replace("_", " "))
    strategies.append(("vector", {"query_terms": query_terms}))
    
    return strategies

def execute_retrieval_strategy(strategy_type: str, params: Dict[str, Any]) -> Any:
    """Execute a specific retrieval strategy"""
    if strategy_type == "maintenance_history":
        return graph_traversal.get_maintenance_history(**params)
    
    elif strategy_type == "path_finding":
        return graph_traversal.find_paths(**params)
    
    elif strategy_type == "connected_components":
        return graph_traversal.get_connected_components(**params)
    
    elif strategy_type == "failure_impact":
        return graph_traversal.analyze_failure_impact(**params)
    
    elif strategy_type == "vector":
        return vector_retriever.retrieve(**params)
    
    else:
        return None
```

### Context Synthesis Node

```python
# In orchestration/nodes.py
from utils.formatters import format_maintenance_history, format_path, format_component_details

def context_synthesis_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Synthesize retrieved information into a coherent context for the LLM"""
    retrieval_results = state["retrieval_results"]
    parsed_query = state["parsed_query"]
    
    context_parts = []
    
    # Add information about what was asked
    intent = parsed_query.get("intent", "information")
    context_parts.append(f"Query intent: {intent.replace('_', ' ')}")
    
    # Process each result based on its type
    for result_type, result_data in retrieval_results.items():
        if result_type == "maintenance_history":
            context_parts.append(format_maintenance_history(result_data))
        
        elif result_type == "path_finding":
            context_parts.append(format_path(result_data))
        
        elif result_type == "connected_components":
            context_parts.append(format_component_details(result_data))
        
        elif result_type == "failure_impact":
            context_parts.append(format_failure_impact(result_data))
        
        elif result_type == "vector":
            context_parts.append(format_vector_results(result_data))
    
    # Combine all context parts
    context = "\n\n".join(context_parts)
    
    return {"context": context}
```

### Response Generation Node

```python
# In orchestration/nodes.py
from langchain.prompts import PromptTemplate

def response_generation_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Generate a response based on the synthesized context"""
    query = state["query"]
    context = state["context"]
    intent = state["parsed_query"].get("intent", "component_info")
    
    # Select prompt template based on intent
    if intent == "maintenance_history":
        template = """
        You are an AI assistant for a water utility company. A user is asking about maintenance history.
        Use the following retrieved information to answer their question.
        
        Retrieved information:
        {context}
        
        User question: {query}
        
        Provide a summary of the maintenance history that addresses:
        - When maintenance was performed
        - What issues were found
        - What repairs were made
        - Any patterns or recurring issues
        - Recommendations for future maintenance if applicable
        
        Answer:
        """
    elif intent == "flow_path":
        template = """
        You are an AI assistant for a water utility company. A user is asking about water flow paths.
        Use the following retrieved information to answer their question.
        
        Retrieved information:
        {context}
        
        User question: {query}
        
        Provide a clear description of the flow path(s) identified. Explain:
        - The sequence of components water flows through
        - Any control points (valves, pumps) along the path
        - Direction of flow
        - Any alternative paths if available
        
        Answer:
        """
    else:
        # Generic template for other intents
        template = """
        You are an AI assistant for a water utility company. Use the following information to answer the question.
        
        Retrieved information:
        {context}
        
        User question: {query}
        
        Provide a helpful, accurate response based on the retrieved information.
        
        Answer:
        """
    
    # Create prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "query"]
    )
    
    # Generate response
    response = llm.invoke(prompt.format(context=context, query=query))
    
    return {"response": response.content}
```

### History Update Node

```python
# In orchestration/nodes.py
def history_update_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Update conversation history with the latest interaction"""
    conversation_history = state.get("conversation_history", [])
    
    # Add current interaction to history
    conversation_history.append({
        "query": state["query"],
        "response": state["response"]
    })
    
    return {"conversation_history": conversation_history}
```

## 5. Building Complete Workflows with LangGraph

Now let's put everything together to create complete workflows for different water network management scenarios.

### Basic Query-Response Workflow

```python
# In orchestration/workflows.py
from langgraph.graph import StateGraph
from .states import WaterNetworkState, initialize_state
from .nodes import (
    parse_query_node,
    retrieval_node,
    context_synthesis_node,
    response_generation_node,
    history_update_node
)

def create_basic_workflow():
    """Create a basic query-response workflow for water network queries"""
    # Initialize the workflow
    workflow = StateGraph(WaterNetworkState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("context_synthesis", context_synthesis_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("update_history", history_update_node)
    
    # Define edges
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieval")
    workflow.add_edge("retrieval", "context_synthesis")
    workflow.add_edge("context_synthesis", "response_generation")
    workflow.add_edge("response_generation", "update_history")
    
    # Set the exit point
    workflow.set_finish_point("update_history")
    
    # Compile the workflow
    return workflow.compile()
```

### Maintenance Planning Workflow

```python
# In orchestration/workflows.py
from .nodes import (
    identify_maintenance_candidates_node,
    assess_risk_node,
    prioritize_maintenance_node,
    generate_maintenance_plan_node
)

def create_maintenance_workflow():
    """Create a workflow for maintenance planning"""
    # Initialize the workflow
    workflow = StateGraph(WaterNetworkState)
    
    # Add basic nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("context_synthesis", context_synthesis_node)
    
    # Add maintenance-specific nodes
    workflow.add_node("identify_candidates", identify_maintenance_candidates_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("prioritize", prioritize_maintenance_node)
    workflow.add_node("generate_plan", generate_maintenance_plan_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("update_history", history_update_node)
    
    # Define edges for the basic flow
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieval")
    workflow.add_edge("retrieval", "context_synthesis")
    workflow.add_edge("context_synthesis", "identify_candidates")
    
    # Define edges for the maintenance-specific flow
    workflow.add_edge("identify_candidates", "assess_risk")
    workflow.add_edge("assess_risk", "prioritize")
    workflow.add_edge("prioritize", "generate_plan")
    workflow.add_edge("generate_plan", "response_generation")
    workflow.add_edge("response_generation", "update_history")
    
    # Set the exit point
    workflow.set_finish_point("update_history")
    
    # Compile the workflow
    return workflow.compile()
```

### Emergency Response Workflow

```python
# In orchestration/workflows.py
from .nodes import (
    analyze_failure_node,
    determine_isolation_strategy_node,
    assess_impact_node,
    create_restoration_plan_node
)

def create_emergency_workflow():
    """Create a workflow for emergency response"""
    # Initialize the workflow
    workflow = StateGraph(WaterNetworkState)
    
    # Add basic nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("context_synthesis", context_synthesis_node)
    
    # Add emergency-specific nodes
    workflow.add_node("analyze_failure", analyze_failure_node)
    workflow.add_node("determine_isolation", determine_isolation_strategy_node)
    workflow.add_node("assess_impact", assess_impact_node)
    workflow.add_node("create_restoration_plan", create_restoration_plan_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("update_history", history_update_node)
    
    # Define edges for the basic flow
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieval")
    workflow.add_edge("retrieval", "context_synthesis")
    workflow.add_edge("context_synthesis", "analyze_failure")
    
    # Define edges for the emergency-specific flow
    workflow.add_edge("analyze_failure", "determine_isolation")
    workflow.add_edge("determine_isolation", "assess_impact")
    workflow.add_edge("assess_impact", "create_restoration_plan")
    workflow.add_edge("create_restoration_plan", "response_generation")
    workflow.add_edge("response_generation", "update_history")
    
    # Set the exit point
    workflow.set_finish_point("update_history")
    
    # Compile the workflow
    return workflow.compile()
```

## 6. Implementing Conditional Logic and Error Handling

LangGraph supports conditional branching and error handling, which are crucial for robust workflows:

### Conditional Routing Based on Intent

```python
# In orchestration/workflows.py
def route_by_intent(state: WaterNetworkState):
    """Route to different nodes based on query intent"""
    intent = state["parsed_query"].get("intent", "")
    
    if intent == "maintenance_history" or intent == "maintenance_planning":
        return "maintenance_flow"
    elif intent == "failure_impact" or intent == "emergency_response":
        return "emergency_flow"
    else:
        return "basic_flow"

def create_intelligent_router_workflow():
    """Create a workflow that routes queries to specialized workflows"""
    # Initialize the workflow
    workflow = StateGraph(WaterNetworkState)
    
    # Add routing nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("basic_flow", create_basic_subgraph())
    workflow.add_node("maintenance_flow", create_maintenance_subgraph())
    workflow.add_node("emergency_flow", create_emergency_subgraph())
    workflow.add_node("update_history", history_update_node)
    
    # Set entry point
    workflow.set_entry_point("parse_query")
    
    # Add conditional edge based on intent
    workflow.add_conditional_edges(
        "parse_query",
        route_by_intent,
        {
            "basic_flow": "basic_flow",
            "maintenance_flow": "maintenance_flow",
            "emergency_flow": "emergency_flow"
        }
    )
    
    # Connect all flows to history update
    workflow.add_edge("basic_flow", "update_history")
    workflow.add_edge("maintenance_flow", "update_history")
    workflow.add_edge("emergency_flow", "update_history")
    
    # Set the exit point
    workflow.set_finish_point("update_history")
    
    # Compile the workflow
    return workflow.compile()
```

### Error Handling with Try-Except Patterns

```python
# In orchestration/nodes.py
def safe_retrieval_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Retrieval node with error handling"""
    try:
        # Attempt normal retrieval
        return retrieval_node(state)
    except Exception as e:
        # Log the error
        error_message = f"Retrieval error: {str(e)}"
        print(error_message)
        
        # Return error state
        return {
            "retrieval_results": {},
            "error": error_message
        }

# In orchestration/workflows.py
def check_for_errors(state: WaterNetworkState):
    """Check if there are errors in the state"""
    if state.get("error"):
        return "error_handling"
    else:
        return "normal_flow"

def create_workflow_with_error_handling():
    """Create a workflow with error handling"""
    # Initialize the workflow
    workflow = StateGraph(WaterNetworkState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieval", safe_retrieval_node)
    workflow.add_node("error_handling", error_handling_node)
    workflow.add_node("context_synthesis", context_synthesis_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("update_history", history_update_node)
    
    # Define basic edges
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieval")
    
    # Add conditional edge for error handling
    workflow.add_conditional_edges(
        "retrieval",
        check_for_errors,
        {
            "error_handling": "error_handling",
            "normal_flow": "context_synthesis"
        }
    )
    
    # Connect error handling back to the main flow
    workflow.add_edge("error_handling", "response_generation")
    
    # Complete the workflow
    workflow.add_edge("context_synthesis", "response_generation")
    workflow.add_edge("response_generation", "update_history")
    
    # Set the exit point
    workflow.set_finish_point("update_history")
    
    # Compile the workflow
    return workflow.compile()
```

## 7. Implementing Iterative Refinement with Cycles

LangGraph supports cycles in the graph, enabling iterative refinement of responses:

```python
# In orchestration/nodes.py
def should_refine_further(state: WaterNetworkState) -> str:
    """Determine if further refinement is needed"""
    # Check if we've hit the maximum number of refinements
    refinement_count = state.get("refinement_count", 0)
    if refinement_count >= 3:
        return "complete"
    
    # Check if the response quality is sufficient
    response = state.get("response", "")
    if not response:
        return "refine"
    
    # Evaluate response quality
    evaluation_prompt = f"""
    Evaluate the quality of this response to the query.
    
    Query: {state['query']}
    Response: {response}
    
    Rate the response completeness on a scale of 1-5:
    """
    
    evaluation = llm.invoke(evaluation_prompt)
    try:
        score = int(evaluation.content.strip())
        if score < 4:
            return "refine"
        else:
            return "complete"
    except:
        return "complete"

def refine_response_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Refine the response based on the current state"""
    query = state["query"]
    current_response = state.get("response", "")
    context = state["context"]
    refinement_count = state.get("refinement_count", 0)
    
    refinement_prompt = f"""
    You are an AI assistant for a water utility company.
    
    Original Query: {query}
    
    Your previous response was:
    {current_response}
    
    Based on this additional context:
    {context}
    
    Provide an improved, more complete response. Address any gaps in the previous response.
    
    Improved response:
    """
    
    refined_response = llm.invoke(refinement_prompt)
    
    return {
        "response": refined_response.content,
        "refinement_count": refinement_count + 1
    }

# In orchestration/workflows.py
def create_iterative_workflow():
    """Create a workflow with iterative refinement"""
    # Initialize the workflow
    workflow = StateGraph(WaterNetworkState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("context_synthesis", context_synthesis_node)
    workflow.add_node("generate_response", response_generation_node)
    workflow.add_node("refine_response", refine_response_node)
    workflow.add_node("update_history", history_update_node)
    
    # Define basic edges
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieval")
    workflow.add_edge("retrieval", "context_synthesis")
    workflow.add_edge("context_synthesis", "generate_response")
    
    # Add conditional edge for refinement
    workflow.add_conditional_edges(
        "generate_response",
        should_refine_further,
        {
            "refine": "refine_response",
            "complete": "update_history"
        }
    )
    
    # Create refinement cycle
    workflow.add_conditional_edges(
        "refine_response",
        should_refine_further,
        {
            "refine": "refine_response",
            "complete": "update_history"
        }
    )
    
    # Set the exit point
    workflow.set_finish_point("update_history")
    
    # Compile the workflow
    return workflow.compile()
```

## 8. Main Application Integration

Now let's integrate these workflows into your main application:

```python
# In main.py
from orchestration.workflows import (
    create_basic_workflow,
    create_maintenance_workflow,
    create_emergency_workflow,
    create_intelligent_router_workflow
)
from orchestration.states import initialize_state

# Initialize workflows
basic_workflow = create_basic_workflow()
maintenance_workflow = create_maintenance_workflow()
emergency_workflow = create_emergency_workflow()
router_workflow = create_intelligent_router_workflow()

def process_query(query, workflow_type=None):
    """Process a user query using the appropriate workflow"""
    if workflow_type:
        # Use specified workflow
        if workflow_type == "maintenance":
            workflow = maintenance_workflow
        elif workflow_type == "emergency":
            workflow = emergency_workflow
        else:
            workflow = basic_workflow
    else:
        # Use intelligent router
        workflow = router_workflow
    
    # Initialize state
    state = initialize_state(query, workflow_type or "auto")
    
    # Run the workflow
    result = workflow.invoke(state)
    
    # Return the final response
    return {
        "response": result["response"],
        "workflow_used": result["current_workflow"],
        "history": result["conversation_history"]
    }

# Example usage
if __name__ == "__main__":
    # Test with a basic query
    query = "What is the status of valve VLV001?"
    result = process_query(query)
    print(f"Response: {result['response']}")
    
    # Test with a maintenance query
    maintenance_query = "What is the maintenance history of the North Hill Tank?"
    result = process_query(maintenance_query, "maintenance")
    print(f"Response: {result['response']}")
    
    # Test with an emergency query
    emergency_query = "What would happen if pipe PIP002 breaks?"
    result = process_query(emergency_query, "emergency")
    print(f"Response: {result['response']}")
```

## 9. Practical Learning Exercise: Building a Water Network Assistant

Let's build a simple but practical water network assistant that demonstrates the principles we've learned. This exercise will help consolidate your understanding of LangGraph for water network management.

### Exercise: Emergency Response Workflow

Create a focused emergency response workflow that:

1. Takes a component ID and failure type as input
2. Determines which valves need to be closed to isolate the component
3. Identifies affected customers and zones
4. Estimates service restoration time
5. Generates a step-by-step emergency response plan

**Step 1: Define the State**

```python
class EmergencyState(TypedDict):
    component_id: str
    failure_type: str
    isolation_valves: List[str]
    affected_zones: List[str]
    affected_customers: Dict[str, int]  # Customer type to count
    estimated_duration: str
    response_plan: List[str]
    completed: bool
```

**Step 2: Define the Nodes**

Create these nodes:
- `identify_isolation_valves`: Find valves needed to isolate the component
- `determine_affected_areas`: Identify zones and customers affected
- `estimate_repair_time`: Estimate repair duration based on failure type
- `generate_response_plan`: Create a step-by-step emergency plan

**Step 3: Create the Graph**

Connect the nodes in a logical sequence, potentially with a refinement cycle.

**Step 4: Test with Example Scenarios**

Test with examples like:
- "Pipe break on main transmission line PIP001"
- "Pump failure at High Service Pump Station PS001"
- "Valve malfunction at pressure reducing valve PRV001"

## 10. Model Context Protocol Integration Preview

In the next guide, we'll cover the Model Context Protocol (MCP) for standardized context handling. Here's a preview of what it will enable:

```python
# Preview of MCP integration
from mcp import MCPContext, MCPRequest, MCPResponse

def mcp_enhanced_node(state: WaterNetworkState) -> Dict[str, Any]:
    """Process state using MCP for standardized context handling"""
    # Create MCP context from state
    mcp_context = MCPContext(
        conversation_history=state["conversation_history"],
        current_query=state["query"],
        retrieved_information=state["context"],
        current_workflow=state["current_workflow"]
    )
    
    # Create MCP request
    request = MCPRequest(
        query=state["query"],
        context=mcp_context,
        task="water_network_analysis"
    )
    
    # Process with MCP-aware LLM
    response = mcp_llm.process(request)
    
    # Update state with MCP response
    return {
        "response": response.content,
        "mcp_context": response.updated_context
    }
```

## 11. Resources and References

### LangGraph Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Python SDK](https://python.langchain.com/docs/get_started/introduction)

### Workflow Design
- [LLM Workflow Design Patterns](https://eugeneyan.com/writing/llm-patterns/)
- [State Management in LLM Applications](https://www.promptingguide.ai/techniques/state)

### Water Network Resources
- [Water Distribution Network Management](https://www.epa.gov/water-research/water-distribution-system-analysis)
- [Emergency Response Planning for Water Systems](https://www.epa.gov/waterutilityresponse)

This guide has provided you with a comprehensive introduction to LangGraph for orchestrating workflows in your Water Network Management System. In the next guide, we'll explore the Model Context Protocol (MCP) for standardized context handling across system components.
