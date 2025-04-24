## Mini-Project 5: LangGraph Workflow Orchestrator

### Overview
In this mini-project, you'll build a simple workflow orchestrator using LangGraph to handle multi-step water network analysis tasks. This tool will allow you to create stateful workflows where the LLM can maintain context and execute complex analysis sequences.

### Learning Objectives
- Understand and implement stateful LLM workflows with LangGraph
- Create a graph-based orchestration system for analysis tasks
- Implement basic workflow persistence
- Practice building conversational agents with memory

### Dependencies
- **Phase 3 Content**: Complete at least Weeks 1-3 of the Orchestration Layer Phase
- **Skills Required**: Python, LangGraph, Basic LLM integration
- **Previous Mini-Projects**: Mini-Project 4 (Multi-Hop Graph Reasoning) helpful but not required

### Estimated Time: 1 week

### Project Steps

#### Step 1: Setup Project Structure
1. Create a new Python project with the following structure:
```
workflow-orchestrator/
├── orchestrator/
│   ├── __init__.py
│   ├── states.py
│   ├── nodes.py
│   ├── workflows.py
│   ├── persistence.py
│   └── app.py
├── tests/
│   ├── __init__.py
│   └── test_workflows.py
├── README.md
└── requirements.txt
```

2. Install required packages:
```
pip install langgraph langchain-community langchain-openai neo4j pydantic fastapi uvicorn
```

#### Step 2: Define State and Nodes
1. Create the state definition in `states.py`:

```python
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class Message(BaseModel):
    """Message in a conversation"""
    role: str
    content: str

class WaterAnalysisState(TypedDict):
    """State for water network analysis workflows"""
    # Conversation
    messages: List[Message]
    
    # Session information
    session_id: str
    created_at: str
    
    # Context tracking
    active_components: List[str]
    active_zones: List[str]
    
    # Analysis state
    current_stage: str
    analysis_type: Optional[str]
    analysis_results: Optional[Dict[str, Any]]
    
    # Error handling
    error: Optional[str]
    
def create_initial_state() -> WaterAnalysisState:
    """Create a new initial state"""
    return {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "active_components": [],
        "active_zones": [],
        "current_stage": "start",
        "analysis_type": None,
        "analysis_results": None,
        "error": None
    }
```

2. Implement the workflow nodes in `nodes.py`:

```python
from typing import Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
import json
from .states import WaterAnalysisState, Message

# Initialize LLM
llm = ChatOpenAI(temperature=0.1)

# Neo4j connection
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "password"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def input_parser(state: WaterAnalysisState) -> Dict[str, Any]:
    """Parse user input to determine analysis type and extract entities"""
    # Get the last user message
    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    if not user_messages:
        return {"current_stage": "waiting_for_input", "error": "No user message found"}
    
    user_message = user_messages[-1]["content"]
    
    prompt = PromptTemplate(
        template="""
        You are an assistant analyzing water network queries. Parse the following query:
        
        Query: {query}
        
        Extract the following information in JSON format:
        1. Analysis Type: The type of analysis requested (maintenance, leak_detection, valve_inspection, flow_analysis, water_quality, pressure_monitoring, component_status)
        2. Component IDs: Any specific component IDs mentioned (e.g., VLV001, PIP042)
        3. Zone IDs: Any specific zone IDs mentioned (e.g., ZONE1, PRESSURE_ZONE_B)
        4. Time Frames: Any time periods mentioned (e.g., "last week", "past month")
        5. Metrics: Any specific metrics mentioned (e.g., pressure, flow rate, water quality)
        
        Return a JSON object with these fields.
        """,
        input_variables=["query"]
    )
    
    # Get parsing result from LLM
    result = llm.invoke(prompt.format(query=user_message))
    
    try:
        # Parse the JSON response
        parsed_data = json.loads(result.content)
        
        # Update the state with extracted information
        new_state = {
            "current_stage": "analysis_planning",
            "analysis_type": parsed_data.get("Analysis Type"),
        }
        
        # Track mentioned components
        component_ids = parsed_data.get("Component IDs", [])
        if component_ids:
            new_state["active_components"] = component_ids
        
        # Track mentioned zones
        zone_ids = parsed_data.get("Zone IDs", [])
        if zone_ids:
            new_state["active_zones"] = zone_ids
        
        return new_state
    
    except json.JSONDecodeError:
        return {
            "current_stage": "error",
            "error": "Failed to parse analysis requirements"
        }

def analysis_planner(state: WaterAnalysisState) -> Dict[str, Any]:
    """Plan the analysis steps based on the analysis type"""
    analysis_type = state["analysis_type"]
    
    if not analysis_type:
        return {
            "current_stage": "error",
            "error": "Analysis type not determined"
        }
    
    # Create an assistant message explaining the plan
    plan_prompt = PromptTemplate(
        template="""
        You are a water network analysis assistant. Based on the user's request for {analysis_type} analysis, 
        create a brief step-by-step plan for conducting this analysis.
        
        Include what data needs to be collected and what insights will be provided.
        
        Keep your response under 200 words and be specific about the steps.
        """,
        input_variables=["analysis_type"]
    )
    
    plan_result = llm.invoke(plan_prompt.format(analysis_type=analysis_type))
    
    # Add the plan to the messages
    messages = state["messages"] + [{"role": "assistant", "content": plan_result.content}]
    
    return {
        "messages": messages,
        "current_stage": "data_collection"
    }

def data_collector(state: WaterAnalysisState) -> Dict[str, Any]:
    """Collect relevant data from Neo4j based on the analysis type"""
    analysis_type = state["analysis_type"]
    active_components = state["active_components"]
    active_zones = state["active_zones"]
    
    results = {}
    
    try:
        with driver.session() as session:
            # If we have specific components to analyze
            if active_components:
                components_data = []
                for component_id in active_components:
                    # Get component details
                    component_result = session.run("""
                    MATCH (c {id: $component_id})
                    RETURN c, labels(c) as labels
                    """, component_id=component_id)
                    
                    record = component_result.single()
                    if record:
                        component = dict(record["c"])
                        component["type"] = record["labels"][0] if record["labels"] else "Unknown"
                        components_data.append(component)
                
                results["components"] = components_data
            
            # If we have specific zones to analyze
            if active_zones:
                zones_data = []
                for zone_id in active_zones:
                    # Get zone details
                    zone_result = session.run("""
                    MATCH (z {id: $zone_id})
                    OPTIONAL MATCH (c)-[:PART_OF]->(z)
                    WITH z, count(c) as component_count
                    RETURN z, component_count
                    """, zone_id=zone_id)
                    
                    record = zone_result.single()
                    if record:
                        zone = dict(record["z"])
                        zone["component_count"] = record["component_count"]
                        zones_data.append(zone)
                
                results["zones"] = zones_data
            
            # Collect data specific to the analysis type
            if analysis_type == "maintenance":
                # Get recent maintenance records
                maintenance_result = session.run("""
                MATCH (c)-[:HAS_MAINTENANCE]->(m)
                WHERE c.id IN $component_ids OR EXISTS {
                    MATCH (c)-[:PART_OF]->(z) WHERE z.id IN $zone_ids
                }
                RETURN c.id as component_id, m
                ORDER BY m.date DESC
                LIMIT 10
                """, component_ids=active_components, zone_ids=active_zones)
                
                maintenance_data = []
                for record in maintenance_result:
                    maintenance = dict(record["m"])
                    maintenance["component_id"] = record["component_id"]
                    maintenance_data.append(maintenance)
                
                results["maintenance_records"] = maintenance_data
            
            elif analysis_type == "valve_inspection":
                # Get valve data
                valve_result = session.run("""
                MATCH (v:Valve)
                WHERE v.id IN $component_ids OR EXISTS {
                    MATCH (v)-[:PART_OF]->(z) WHERE z.id IN $zone_ids
                }
                RETURN v.id as id, v.status as status, v.installDate as installDate,
                       v.lastInspection as lastInspection
                """, component_ids=active_components, zone_ids=active_zones)
                
                valve_data = [dict(record) for record in valve_result]
                results["valve_data"] = valve_data
        
        return {
            "current_stage": "analysis_execution",
            "analysis_results": results
        }
    
    except Exception as e:
        return {
            "current_stage": "error",
            "error": f"Data collection failed: {str(e)}"
        }

def analysis_executor(state: WaterAnalysisState) -> Dict[str, Any]:
    """Execute the analysis based on collected data"""
    analysis_type = state["analysis_type"]
    analysis_results = state["analysis_results"]
    
    if not analysis_results:
        return {
            "current_stage": "error",
            "error": "No data available for analysis"
        }
    
    # Prepare the prompt for analysis
    analysis_prompt = PromptTemplate(
        template="""
        You are a water network analysis expert. Based on the following data, provide insights for {analysis_type} analysis.
        
        Data: {data}
        
        Provide a concise analysis with key findings and recommendations. Include:
        1. Summary of what you found
        2. Key issues identified (if any)
        3. Recommended actions
        
        Format your response in a clear, professional manner suitable for water utility staff.
        """,
        input_variables=["analysis_type", "data"]
    )
    
    # Get analysis from LLM
    analysis_result = llm.invoke(analysis_prompt.format(
        analysis_type=analysis_type, 
        data=json.dumps(analysis_results, indent=2)
    ))
    
    # Add the analysis to the messages
    messages = state["messages"] + [{"role": "assistant", "content": analysis_result.content}]
    
    return {
        "messages": messages,
        "current_stage": "complete"
    }

def error_handler(state: WaterAnalysisState) -> Dict[str, Any]:
    """Handle errors in the workflow"""
    error = state["error"]
    
    # Create an error message for the user
    error_prompt = PromptTemplate(
        template="""
        You are a helpful assistant. There was an error during the water network analysis:
        
        Error: {error}
        
        Please provide a helpful, user-friendly message explaining the issue and suggesting what the user could do differently.
        Keep your response concise and constructive.
        """,
        input_variables=["error"]
    )
    
    error_result = llm.invoke(error_prompt.format(error=error))
    
    # Add the error message to the messages
    messages = state["messages"] + [{"role": "assistant", "content": error_result.content}]
    
    return {
        "messages": messages,
        "current_stage": "waiting_for_input"
    }
```

#### Step 3: Create Workflows
1. Implement the main workflow in `workflows.py`:

```python
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph
from .states import WaterAnalysisState, create_initial_state
from .nodes import input_parser, analysis_planner, data_collector, analysis_executor, error_handler

def route_based_on_stage(state: WaterAnalysisState) -> Literal["input_parser", "analysis_planner", "data_collector", "analysis_executor", "error_handler", "end"]:
    """Route to the next node based on the current stage"""
    current_stage = state["current_stage"]
    
    if current_stage == "start":
        return "input_parser"
    elif current_stage == "analysis_planning":
        return "analysis_planner"
    elif current_stage == "data_collection":
        return "data_collector"
    elif current_stage == "analysis_execution":
        return "analysis_executor"
    elif current_stage == "error":
        return "error_handler"
    elif current_stage == "complete":
        return "end"
    else:
        return "error_handler"

def create_workflow():
    """Create the water analysis workflow"""
    # Initialize the workflow with the state type
    workflow = StateGraph(WaterAnalysisState)
    
    # Add nodes
    workflow.add_node("input_parser", input_parser)
    workflow.add_node("analysis_planner", analysis_planner)
    workflow.add_node("data_collector", data_collector)
    workflow.add_node("analysis_executor", analysis_executor)
    workflow.add_node("error_handler", error_handler)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "",  # This empty string refers to the root/entry point
        route_based_on_stage,
        {
            "input_parser": "input_parser",
            "analysis_planner": "analysis_planner",
            "data_collector": "data_collector",
            "analysis_executor": "analysis_executor",
            "error_handler": "error_handler",
            "end": "__end__"
        }
    )
    
    # Add regular edges
    workflow.add_edge("input_parser", "")
    workflow.add_edge("analysis_planner", "")
    workflow.add_edge("data_collector", "")
    workflow.add_edge("analysis_executor", "")
    workflow.add_edge("error_handler", "")
    
    # Compile the workflow
    return workflow.compile()
```

#### Step 4: Implement Persistence
1. Create a simple persistence module in `persistence.py`:

```python
import json
import os
from typing import Dict, Any, Optional
from .states import WaterAnalysisState

class WorkflowStorage:
    """Simple storage for workflow states"""
    
    def __init__(self, storage_dir="./storage"):
        """Initialize the storage"""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_state(self, state: WaterAnalysisState) -> bool:
        """Save a workflow state to disk"""
        session_id = state["session_id"]
        
        try:
            # Convert the state to JSON
            state_json = json.dumps(state, indent=2)
            
            # Write to file
            with open(f"{self.storage_dir}/{session_id}.json", "w") as f:
                f.write(state_json)
            
            return True
        
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, session_id: str) -> Optional[WaterAnalysisState]:
        """Load a workflow state from disk"""
        file_path = f"{self.storage_dir}/{session_id}.json"
        
        if not os.path.exists(file_path):
            return None
        
        try:
            # Read from file
            with open(file_path, "r") as f:
                state_json = f.read()
            
            # Parse JSON
            state = json.loads(state_json)
            
            return state
        
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    
    def list_sessions(self) -> list:
        """List all available session IDs"""
        session_files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
        return [f.replace(".json", "") for f in session_files]
```

#### Step 5: Create Application Interface
1. Implement the application in `app.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from .states import create_initial_state, WaterAnalysisState, Message
from .workflows import create_workflow
from .persistence import WorkflowStorage

app = FastAPI(title="Water Network Workflow Orchestrator")
workflow = create_workflow()
storage = WorkflowStorage()

class UserMessage(BaseModel):
    """User message input"""
    content: str

class SessionResponse(BaseModel):
    """Response containing session information"""
    session_id: str
    messages: List[Dict[str, str]]
    current_stage: str
    analysis_type: Optional[str] = None

@app.post("/sessions", response_model=SessionResponse)
def create_session():
    """Create a new workflow session"""
    # Create initial state
    state = create_initial_state()
    
    # Save state
    storage.save_state(state)
    
    # Return session info
    return {
        "session_id": state["session_id"],
        "messages": state["messages"],
        "current_stage": state["current_stage"],
        "analysis_type": state["analysis_type"]
    }

@app.post("/sessions/{session_id}/messages", response_model=SessionResponse)
def add_message(session_id: str, message: UserMessage):
    """Add a message to an existing session and process it"""
    # Load existing state
    state = storage.load_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add user message to state
    state["messages"].append({"role": "user", "content": message.content})
    
    # Set stage to start to trigger workflow
    state["current_stage"] = "start"
    
    # Run the workflow
    new_state = workflow.invoke(state)
    
    # Save updated state
    storage.save_state(new_state)
    
    # Return updated session info
    return {
        "session_id": new_state["session_id"],
        "messages": new_state["messages"],
        "current_stage": new_state["current_stage"],
        "analysis_type": new_state["analysis_type"]
    }

@app.get("/sessions", response_model=List[str])
def list_sessions():
    """List all available sessions"""
    return storage.list_sessions()

@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    """Get information about a specific session"""
    # Load state
    state = storage.load_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return session info
    return {
        "session_id": state["session_id"],
        "messages": state["messages"],
        "current_stage": state["current_stage"],
        "analysis_type": state["analysis_type"]
    }

if __name__ == "__main__":
    uvicorn.run("orchestrator.app:app", host="0.0.0.0", port=8000, reload=True)
```

#### Step 6: Test the Workflow
1. Create a test file in `tests/test_workflows.py`:

```python
import pytest
from orchestrator.states import create_initial_state
from orchestrator.workflows import create_workflow
from unittest.mock import patch, MagicMock

@pytest.fixture
def workflow():
    """Create a workflow for testing"""
    return create_workflow()

@pytest.fixture
def initial_state():
    """Create an initial state for testing"""
    state = create_initial_state()
    # Add a test message
    state["messages"] = [{"role": "user", "content": "Perform valve inspection for VLV001"}]
    return state

def test_workflow_initialization(workflow, initial_state):
    """Test that the workflow initializes correctly"""
    assert workflow is not None
    assert initial_state["current_stage"] == "start"
    assert initial_state["session_id"] is not None

@patch("orchestrator.nodes.llm")
def test_input_parsing(mock_llm, workflow, initial_state):
    """Test the input parsing stage"""
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.content = '{"Analysis Type": "valve_inspection", "Component IDs": ["VLV001"], "Zone IDs": [], "Time Frames": [], "Metrics": []}'
    mock_llm.invoke.return_value = mock_response
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Check that the state was updated correctly
    assert result["analysis_type"] == "valve_inspection"
    assert "VLV001" in result["active_components"]
    assert result["current_stage"] == "complete" or result["current_stage"] == "analysis_planning"

@patch("orchestrator.nodes.driver")
@patch("orchestrator.nodes.llm")
def test_valve_inspection_workflow(mock_llm, mock_driver, workflow, initial_state):
    """Test the complete valve inspection workflow"""
    # Mock LLM responses
    mock_parse_response = MagicMock()
    mock_parse_response.content = '{"Analysis Type": "valve_inspection", "Component IDs": ["VLV001"], "Zone IDs": [], "Time Frames": [], "Metrics": []}'
    
    mock_plan_response = MagicMock()
    mock_plan_response.content = "Here's the plan for valve inspection..."
    
    mock_analysis_response = MagicMock()
    mock_analysis_response.content = "Based on the inspection data, valve VLV001 is in good condition..."
    
    mock_llm.invoke.side_effect = [mock_parse_response, mock_plan_response, mock_analysis_response]
    
    # Mock Neo4j session
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    # Mock Neo4j results
    mock_valve_result = MagicMock()
    mock_valve_data = [
        {"id": "VLV001", "status": "Open", "installDate": "2015-03-15", "lastInspection": "2022-05-10"}
    ]
    mock_valve_result.__iter__.return_value = mock_valve_data
    mock_session.run.return_value = mock_valve_result
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Check that the workflow completed successfully
    assert result["current_stage"] == "complete"
    assert len(result["messages"]) >= 3  # Initial + at least 2 assistant messages
    
    # Get the last assistant message
    assistant_messages = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(assistant_messages) >= 1
    assert "valve" in assistant_messages[-1]["content"].lower()
```

#### Step 7: Run and Test
1. Run the FastAPI application:
```bash
python -m orchestrator.app
```

2. Use an API client (like Postman or cURL) to test the API:

```bash
# Create a new session
curl -X POST http://localhost:8000/sessions

# Add a message to the session
curl -X POST http://localhost:8000/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Perform valve inspection for VLV001"}'

# Get session information
curl http://localhost:8000/sessions/{session_id}
```

### Deliverables
1. A functional LangGraph workflow for water network analysis
2. A simple API for workflow interaction and persistence
3. Support for different analysis types:
   - Valve inspection
   - Maintenance analysis
   - Other analysis types based on your implementation
4. Tests for the workflow functionality

### Extensions
1. Add a web interface to interact with the workflows
2. Implement more specialized analysis types
3. Add workflow visualization capabilities
4. Enhance the persistence with database storage
5. Implement parallel workflows for different analysis branches
6. Add authentication for API endpoints

### Relation to Main Project
This mini-project directly implements the LangGraph orchestration concepts from Phase 3 of the main project. The workflow orchestrator you build will demonstrate how to create stateful, graph-based workflows for complex water network analysis tasks. These capabilities are central to the orchestration layer in your Water Network Intelligence System, allowing for more sophisticated multi-step analysis processes.
