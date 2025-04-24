# Model Context Protocol (MCP) Integration Guide for Water Networks

This guide will introduce you to the Model Context Protocol (MCP) and demonstrate how to integrate it into your Water Network Management System. MCP provides a standardized approach for maintaining context across system components, which is crucial for building a comprehensive water network intelligence system.

## 1. Understanding the Model Context Protocol

### What is the Model Context Protocol?

The Model Context Protocol (MCP) is a standardized approach for:

1. **Packaging Context**: Structuring and organizing contextual information
2. **Maintaining State**: Preserving context across different components and interactions
3. **Transferring Knowledge**: Moving information between different parts of your system
4. **Standardizing Interactions**: Creating consistent interfaces between components

It serves as a "contract" that defines how context should be formatted, processed, and exchanged throughout your system.

### Core Concepts of MCP

- **Context Objects**: Standardized containers for contextual information
- **Context Providers**: Components that generate or supply context
- **Context Consumers**: Components that use or process context
- **Context Operations**: Standard methods for manipulating context
- **Context Protocols**: Rules for how context flows through the system

### Why MCP for Water Networks?

Water network management involves complex, multifaceted operations that benefit from standardized context handling:

1. **Cross-Component Awareness**: Components need awareness of the broader network state
2. **Context Preservation**: Operations often span multiple sessions and interactions
3. **Domain-Specific Context**: Water networks have unique contextual elements (pressure zones, flow paths, etc.)
4. **Multi-User Collaboration**: Different stakeholders may interact with the same underlying context
5. **Context Evolution**: Network understanding evolves over time through operations and analysis

## 2. Designing MCP Schema for Water Networks

The first step in implementing MCP is to design a schema that captures the specific context needs of water network management.

### Core MCP Context Object

```python
# mcp/schema.py
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class WaterNetworkContext(BaseModel):
    """Base context object for water network operations"""
    
    # Core metadata
    context_id: str = Field(..., description="Unique identifier for this context")
    version: str = Field("1.0", description="Schema version")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # User and session information
    user_id: Optional[str] = Field(None, description="ID of the user associated with this context")
    session_id: Optional[str] = Field(None, description="ID of the session")
    
    # Conversation context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, 
                                                      description="History of previous interactions")
    current_query: Optional[str] = Field(None, description="The current user query")
    
    # Network awareness
    active_components: List[str] = Field(default_factory=list, 
                                       description="Component IDs currently in focus")
    active_zones: List[str] = Field(default_factory=list, 
                                  description="Zone IDs currently in focus")
    
    # Operational awareness
    current_workflow: Optional[str] = Field(None, description="The active workflow type")
    workflow_state: Dict[str, Any] = Field(default_factory=dict, 
                                         description="Current state of the workflow")
    
    # Temporal awareness
    reference_time: Optional[datetime] = Field(None, 
                                            description="Point in time being referenced")
    time_horizon: Optional[str] = Field(None, 
                                     description="Time span being considered (e.g., 'next 24h')")
    
    # Knowledge context
    retrieved_information: Dict[str, Any] = Field(default_factory=dict, 
                                               description="Information retrieved from knowledge sources")
    
    def update(self, **kwargs):
        """Update context fields and set updated_at timestamp"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.now()
        return self
```

### Domain-Specific Context Extensions

#### Maintenance Context

```python
# mcp/schema.py
class MaintenanceContext(WaterNetworkContext):
    """Context specific to maintenance operations"""
    
    # Maintenance-specific context
    maintenance_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, 
                                                              description="Maintenance history by component")
    components_to_maintain: List[Dict[str, Any]] = Field(default_factory=list, 
                                                      description="Components scheduled for maintenance")
    maintenance_priorities: Dict[str, float] = Field(default_factory=dict, 
                                                  description="Priority scores for components")
    available_resources: Dict[str, Any] = Field(default_factory=dict, 
                                             description="Available maintenance resources")
    maintenance_schedule: List[Dict[str, Any]] = Field(default_factory=list, 
                                                   description="Scheduled maintenance activities")
```

#### Emergency Response Context

```python
# mcp/schema.py
class EmergencyResponseContext(WaterNetworkContext):
    """Context specific to emergency response operations"""
    
    # Emergency-specific context
    incident_details: Dict[str, Any] = Field(default_factory=dict, 
                                          description="Details about the current incident")
    affected_components: List[Dict[str, Any]] = Field(default_factory=list, 
                                                   description="Components affected by the incident")
    isolation_strategy: List[Dict[str, Any]] = Field(default_factory=list, 
                                                  description="Valves to close for isolation")
    affected_customers: Dict[str, int] = Field(default_factory=dict, 
                                            description="Affected customers by type")
    restoration_plan: List[Dict[str, Any]] = Field(default_factory=list, 
                                                description="Steps to restore service")
    estimated_resolution_time: Optional[datetime] = Field(None, 
                                                      description="Estimated time to resolution")
```

### MCP Operations

```python
# mcp/operations.py
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from .schema import WaterNetworkContext, MaintenanceContext, EmergencyResponseContext

def create_context(context_type: str = "base", **kwargs) -> WaterNetworkContext:
    """Create a new context object of the specified type"""
    context_id = kwargs.pop("context_id", str(uuid.uuid4()))
    
    if context_type == "maintenance":
        return MaintenanceContext(context_id=context_id, **kwargs)
    elif context_type == "emergency":
        return EmergencyResponseContext(context_id=context_id, **kwargs)
    else:
        return WaterNetworkContext(context_id=context_id, **kwargs)

def merge_contexts(primary: WaterNetworkContext, secondary: WaterNetworkContext) -> WaterNetworkContext:
    """Merge two context objects, prioritizing primary where conflicts exist"""
    # Create a copy of the primary context
    merged_data = primary.dict()
    
    # Update with non-conflicting data from secondary
    secondary_data = secondary.dict()
    for key, value in secondary_data.items():
        if key not in ["context_id", "created_at"]:
            if isinstance(value, dict) and isinstance(merged_data.get(key), dict):
                # Deep merge dictionaries
                merged_data[key] = {**merged_data[key], **value}
            elif isinstance(value, list) and isinstance(merged_data.get(key), list):
                # Combine lists without duplicates
                merged_data[key] = list(set(merged_data[key] + value))
            elif merged_data.get(key) is None or merged_data.get(key) == []:
                # Take secondary value if primary is empty
                merged_data[key] = value
    
    # Update timestamp
    merged_data["updated_at"] = datetime.now()
    
    # Create appropriate context type
    if isinstance(primary, MaintenanceContext):
        return MaintenanceContext(**merged_data)
    elif isinstance(primary, EmergencyResponseContext):
        return EmergencyResponseContext(**merged_data)
    else:
        return WaterNetworkContext(**merged_data)

def extract_subcontext(context: WaterNetworkContext, keys: List[str]) -> Dict[str, Any]:
    """Extract a subset of the context with only specified keys"""
    return {k: getattr(context, k) for k in keys if hasattr(context, k)}

def create_context_diff(original: WaterNetworkContext, updated: WaterNetworkContext) -> Dict[str, Any]:
    """Create a diff showing what changed between two context versions"""
    original_dict = original.dict()
    updated_dict = updated.dict()
    
    diff = {}
    for key in original_dict:
        if key not in ["updated_at"]:
            if original_dict[key] != updated_dict[key]:
                diff[key] = {
                    "before": original_dict[key],
                    "after": updated_dict[key]
                }
    
    return diff
```

## 3. Building MCP Request and Response Objects

The next step is to create standardized request and response objects that incorporate context.

### MCP Request Object

```python
# mcp/protocol.py
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from .schema import WaterNetworkContext

class MCPRequest(BaseModel):
    """Standard request format for MCP-enabled operations"""
    
    # Core request fields
    request_id: str = Field(..., description="Unique identifier for this request")
    operation: str = Field(..., description="Operation being requested")
    
    # Context information
    context: WaterNetworkContext = Field(..., description="Context for this request")
    
    # Operation-specific parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, 
                                    description="Parameters for the operation")
    
    # Source tracking
    source_component: str = Field(..., description="Component making the request")
    target_component: str = Field(..., description="Component that should handle the request")
    
    # Control fields
    priority: int = Field(1, description="Request priority (1-5, higher is more important)")
    timeout_ms: int = Field(30000, description="Timeout in milliseconds")
```

### MCP Response Object

```python
# mcp/protocol.py
class MCPResponse(BaseModel):
    """Standard response format for MCP-enabled operations"""
    
    # Core response fields
    response_id: str = Field(..., description="Unique identifier for this response")
    request_id: str = Field(..., description="ID of the request this responds to")
    status: str = Field(..., description="Status of the operation (success, error, partial)")
    
    # Context information
    original_context: WaterNetworkContext = Field(..., description="Original request context")
    updated_context: WaterNetworkContext = Field(..., description="Updated context after operation")
    
    # Response content
    content: Any = Field(None, description="Operation result content")
    content_type: str = Field("text", description="Type of content returned")
    
    # Error information
    error: Optional[str] = Field(None, description="Error message if status is error")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    # Performance metrics
    processing_time_ms: int = Field(0, description="Time taken to process in milliseconds")
```

### MCP Message Handler

```python
# mcp/handler.py
import uuid
import time
from typing import Dict, Any, Callable, Optional
from .protocol import MCPRequest, MCPResponse
from .schema import WaterNetworkContext
from .operations import create_context_diff

class MCPHandler:
    """Handler for processing MCP requests and generating responses"""
    
    def __init__(self):
        """Initialize the handler with registered operation handlers"""
        self.operation_handlers = {}
    
    def register_operation(self, operation: str, handler: Callable):
        """Register a handler function for a specific operation"""
        self.operation_handlers[operation] = handler
    
    def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request and generate a response"""
        start_time = time.time()
        
        # Initialize response
        response = MCPResponse(
            response_id=str(uuid.uuid4()),
            request_id=request.request_id,
            status="error",  # Default to error until successful processing
            original_context=request.context,
            updated_context=request.context  # Start with original context
        )
        
        try:
            # Check if operation is registered
            if request.operation not in self.operation_handlers:
                response.error = f"Unknown operation: {request.operation}"
                return response
            
            # Get the appropriate handler
            handler = self.operation_handlers[request.operation]
            
            # Execute the operation
            result, updated_context = handler(request.context, request.parameters)
            
            # Update response
            response.status = "success"
            response.content = result
            response.updated_context = updated_context
            
            # Add content type based on result
            if isinstance(result, str):
                response.content_type = "text"
            elif isinstance(result, dict) or isinstance(result, list):
                response.content_type = "json"
            else:
                response.content_type = "object"
            
            # Create context diff for debugging
            context_diff = create_context_diff(request.context, updated_context)
            if context_diff:
                print(f"Context changes: {context_diff}")
                
        except Exception as e:
            # Handle errors
            response.error = str(e)
        
        # Calculate processing time
        end_time = time.time()
        response.processing_time_ms = int((end_time - start_time) * 1000)
        
        return response
```

## 4. Integrating MCP with Neo4j Graph Database

Let's integrate MCP with your Neo4j graph database to maintain context about the water network structure.

### Component Context Provider

```python
# mcp/providers/neo4j_provider.py
from typing import Dict, List, Any, Tuple
from neo4j import GraphDatabase
from ..schema import WaterNetworkContext

class Neo4jContextProvider:
    """Provider that retrieves context from Neo4j database"""
    
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def get_component_context(self, component_id: str, context: WaterNetworkContext) -> WaterNetworkContext:
        """Enrich context with information about a specific component"""
        with self.driver.session() as session:
            # Get component details
            result = session.run("""
            MATCH (c {id: $component_id})
            RETURN c, labels(c) as labels
            """, component_id=component_id)
            
            record = result.single()
            if not record:
                # Component not found
                return context
            
            component = record["c"]
            labels = record["labels"]
            
            # Add component to active components if not already there
            if component_id not in context.active_components:
                context.active_components.append(component_id)
            
            # Store component details in retrieved information
            if "components" not in context.retrieved_information:
                context.retrieved_information["components"] = {}
            
            context.retrieved_information["components"][component_id] = {
                "properties": dict(component),
                "type": labels[0] if labels else "Unknown"
            }
            
            # If this is a valve, get its status for operational awareness
            if "Valve" in labels:
                if "operational_status" not in context.retrieved_information:
                    context.retrieved_information["operational_status"] = {}
                
                context.retrieved_information["operational_status"][component_id] = component.get("operationalStatus", "Unknown")
            
            # Get zone information if applicable
            zone_result = session.run("""
            MATCH (c {id: $component_id})-[:PART_OF]->(z)
            RETURN z.id as zone_id, z.name as zone_name
            """, component_id=component_id)
            
            for zone_record in zone_result:
                zone_id = zone_record["zone_id"]
                if zone_id not in context.active_zones:
                    context.active_zones.append(zone_id)
                
                if "zones" not in context.retrieved_information:
                    context.retrieved_information["zones"] = {}
                
                context.retrieved_information["zones"][zone_id] = {
                    "name": zone_record["zone_name"]
                }
            
            return context
    
    def get_connected_components(self, component_id: str, context: WaterNetworkContext) -> WaterNetworkContext:
        """Enrich context with information about connected components"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})-[r]-(connected)
            RETURN connected.id as connected_id, type(r) as relationship_type, connected, labels(connected) as labels
            """, component_id=component_id)
            
            if "connections" not in context.retrieved_information:
                context.retrieved_information["connections"] = {}
            
            context.retrieved_information["connections"][component_id] = []
            
            for record in result:
                connected_id = record["connected_id"]
                connected_component = record["connected"]
                relationship_type = record["relationship_type"]
                labels = record["labels"]
                
                # Add to active components
                if connected_id not in context.active_components:
                    context.active_components.append(connected_id)
                
                # Add connection information
                context.retrieved_information["connections"][component_id].append({
                    "id": connected_id,
                    "type": labels[0] if labels else "Unknown",
                    "relationship": relationship_type
                })
                
                # Store basic information about the connected component
                if "components" not in context.retrieved_information:
                    context.retrieved_information["components"] = {}
                
                context.retrieved_information["components"][connected_id] = {
                    "properties": dict(connected_component),
                    "type": labels[0] if labels else "Unknown"
                }
            
            return context
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
```

### Zone Context Provider

```python
# mcp/providers/neo4j_provider.py
def get_zone_context(self, zone_id: str, context: WaterNetworkContext) -> WaterNetworkContext:
    """Enrich context with information about a specific zone"""
    with self.driver.session() as session:
        # Get zone details
        zone_result = session.run("""
        MATCH (z {id: $zone_id})
        RETURN z
        """, zone_id=zone_id)
        
        zone_record = zone_result.single()
        if not zone_record:
            # Zone not found
            return context
        
        zone = zone_record["z"]
        
        # Add zone to active zones if not already there
        if zone_id not in context.active_zones:
            context.active_zones.append(zone_id)
        
        # Store zone details in retrieved information
        if "zones" not in context.retrieved_information:
            context.retrieved_information["zones"] = {}
        
        context.retrieved_information["zones"][zone_id] = dict(zone)
        
        # Get components in this zone
        components_result = session.run("""
        MATCH (c)-[:PART_OF]->(z {id: $zone_id})
        RETURN c.id as component_id, labels(c) as labels, c
        LIMIT 100
        """, zone_id=zone_id)
        
        if "zone_components" not in context.retrieved_information:
            context.retrieved_information["zone_components"] = {}
        
        context.retrieved_information["zone_components"][zone_id] = []
        
        for comp_record in components_result:
            component_id = comp_record["component_id"]
            component = comp_record["c"]
            labels = comp_record["labels"]
            
            # Add to active components
            if component_id not in context.active_components:
                context.active_components.append(component_id)
            
            # Add to zone components
            context.retrieved_information["zone_components"][zone_id].append({
                "id": component_id,
                "type": labels[0] if labels else "Unknown"
            })
            
            # Store basic information about the component
            if "components" not in context.retrieved_information:
                context.retrieved_information["components"] = {}
            
            context.retrieved_information["components"][component_id] = {
                "properties": dict(component),
                "type": labels[0] if labels else "Unknown"
            }
        
        return context
```

## 5. Integrating MCP with GraphRAG and LangGraph

Now let's integrate MCP with the GraphRAG and LangGraph components we've built in previous guides.

### MCP-Aware GraphRAG

```python
# mcp/consumers/graphrag_consumer.py
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from ..schema import WaterNetworkContext
from ..protocol import MCPRequest, MCPResponse

class MCPGraphRAG:
    """GraphRAG component that is MCP-aware"""
    
    def __init__(self, chroma_client, neo4j_driver, embedding_model):
        """Initialize with required components"""
        self.chroma_client = chroma_client
        self.neo4j_driver = neo4j_driver
        self.embedding_model = embedding_model
        
        # Get collections
        self.component_collection = chroma_client.get_collection("water_network_components")
        self.relationship_collection = chroma_client.get_collection("water_network_relationships")
    
    def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request for retrieval"""
        # Extract relevant context
        context = request.context
        query = context.current_query or ""
        parameters = request.parameters
        
        # Determine retrieval strategy based on parameters
        strategy = parameters.get("strategy", "hybrid")
        
        if strategy == "vector":
            # Vector-only retrieval
            result = self.vector_retrieval(query, context)
        elif strategy == "graph":
            # Graph-only retrieval
            result = self.graph_retrieval(query, context)
        else:
            # Hybrid retrieval (default)
            result = self.hybrid_retrieval(query, context)
        
        # Update context with retrieval results
        updated_context = context.copy()
        updated_context.retrieved_information["rag_results"] = result
        
        # Create response
        response = MCPResponse(
            response_id=str(uuid.uuid4()),
            request_id=request.request_id,
            status="success",
            original_context=context,
            updated_context=updated_context,
            content=result,
            content_type="json"
        )
        
        return response
    
    def vector_retrieval(self, query: str, context: WaterNetworkContext) -> Dict[str, Any]:
        """Perform vector-based retrieval"""
        # Use active components and zones to enhance query
        enhanced_query = query
        
        if context.active_components:
            component_str = " ".join(context.active_components[:5])  # Limit to 5 to avoid overfitting
            enhanced_query += f" {component_str}"
        
        if context.active_zones:
            zone_str = " ".join(context.active_zones[:3])
            enhanced_query += f" {zone_str}"
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode(enhanced_query).tolist()
        
        # Search components
        component_results = self.component_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Search relationships
        relationship_results = self.relationship_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        return {
            "components": component_results,
            "relationships": relationship_results,
            "query_method": "vector"
        }
    
    def graph_retrieval(self, query: str, context: WaterNetworkContext) -> Dict[str, Any]:
        """Perform graph-based retrieval"""
        results = {}
        
        # Use active components for graph traversal
        for component_id in context.active_components[:3]:  # Limit to top 3 for performance
            # Get connected components
            with self.neo4j_driver.session() as session:
                connected_result = session.run("""
                MATCH (c {id: $component_id})-[r]-(connected)
                RETURN connected.id as id, type(r) as relationship, labels(connected) as labels
                LIMIT 10
                """, component_id=component_id)
                
                results[component_id] = {
                    "connected": [dict(record) for record in connected_result]
                }
                
                # Get maintenance history if available
                maintenance_result = session.run("""
                MATCH (c {id: $component_id})-[:HAS_MAINTENANCE]->(m)
                RETURN m
                ORDER BY m.date DESC
                LIMIT 5
                """, component_id=component_id)
                
                results[component_id]["maintenance"] = [dict(record["m"]) for record in maintenance_result]
        
        return {
            "graph_results": results,
            "query_method": "graph"
        }
    
    def hybrid_retrieval(self, query: str, context: WaterNetworkContext) -> Dict[str, Any]:
        """Perform hybrid retrieval combining vector and graph approaches"""
        # Get results from both methods
        vector_results = self.vector_retrieval(query, context)
        graph_results = self.graph_retrieval(query, context)
        
        # Combine results
        hybrid_results = {
            "vector": vector_results,
            "graph": graph_results,
            "query_method": "hybrid"
        }
        
        # Add additional context-aware retrieval
        if context.current_workflow == "maintenance":
            # For maintenance workflows, add additional maintenance-focused retrieval
            with self.neo4j_driver.session() as session:
                # Find components with recent maintenance issues
                maintenance_issues = session.run("""
                MATCH (c)-[:HAS_MAINTENANCE]->(m)
                WHERE m.findings CONTAINS 'issue' OR m.findings CONTAINS 'problem'
                RETURN c.id as component_id, m.date as date, m.findings as findings
                ORDER BY m.date DESC
                LIMIT 10
                """)
                
                hybrid_results["maintenance_issues"] = [dict(record) for record in maintenance_issues]
        
        elif context.current_workflow == "emergency":
            # For emergency workflows, add critical component identification
            with self.neo4j_driver.session() as session:
                # Find potential critical components
                critical_components = session.run("""
                MATCH (c)-[r]-(connected)
                WITH c, count(r) as connections
                WHERE connections > 3
                RETURN c.id as component_id, labels(c) as type, connections
                ORDER BY connections DESC
                LIMIT 10
                """)
                
                hybrid_results["critical_components"] = [dict(record) for record in critical_components]
        
        return hybrid_results
```

### MCP-Aware LangGraph Nodes

```python
# mcp/consumers/langgraph_consumer.py
from typing import Dict, Any
from langgraph.graph import StateGraph
from ..schema import WaterNetworkContext
from ..protocol import MCPRequest, MCPResponse
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(temperature=0)

def mcp_query_understanding_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node that parses queries with MCP awareness"""
    # Extract query from state
    query = state.get("query", "")
    
    # Create or get MCP context
    if "mcp_context" in state:
        context = state["mcp_context"]
    else:
        context = create_context(context_type="base")
    
    # Update context with current query
    context.update(current_query=query)
    
    # Create request to parse the query
    request = MCPRequest(
        request_id=str(uuid.uuid4()),
        operation="parse_query",
        context=context,
        parameters={},
        source_component="langgraph",
        target_component="query_parser"
    )
    
    # Use handler to process request
    handler = MCPHandler()
    handler.register_operation("parse_query", parse_query_operation)
    response = handler.process_request(request)
    
    # Extract parsed query from response
    parsed_query = response.content
    
    # Update state with MCP context and parsed query
    return {
        "parsed_query": parsed_query,
        "mcp_context": response.updated_context
    }

def mcp_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node that performs retrieval with MCP awareness"""
    # Get MCP context and parsed query
    context = state["mcp_context"]
    parsed_query = state["parsed_query"]
    
    # Create request for retrieval
    request = MCPRequest(
        request_id=str(uuid.uuid4()),
        operation="retrieve_information",
        context=context,
        parameters={
            "strategy": "hybrid",
            "query_understanding": parsed_query
        },
        source_component="langgraph",
        target_component="graphrag"
    )
    
    # Use MCPGraphRAG to process request
    graphrag = MCPGraphRAG(chroma_client, neo4j_driver, embedding_model)
    response = graphrag.process_request(request)
    
    # Update state with retrieval results and updated context
    return {
        "retrieval_results": response.content,
        "mcp_context": response.updated_context
    }

def mcp_response_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node that generates responses with MCP awareness"""
    # Get MCP context
    context = state["mcp_context"]
    retrieval_results = state["retrieval_results"]
    
    # Create a prompt that includes context information
    prompt = f"""
    You are an AI assistant for a water utility company. Use the following information to answer the question.
    
    User query: {context.current_query}
    
    Active components: {', '.join(context.active_components)}
    Active zones: {', '.join(context.active_zones)}
    Current workflow: {context.current_workflow or 'General query'}
    
    Retrieved information:
    {retrieval_results}
    
    Previous conversation:
    {' | '.join([f"User: {exchange['query']} | Assistant: {exchange['response']}" for exchange in context.conversation_history[-3:]])}
    
    Provide a helpful, accurate response based on the retrieved information and context.
    
    Answer:
    """
    
    # Generate response
    response_text = llm.invoke(prompt).content
    
    # Create response
    request = MCPRequest(
        request_id=str(uuid.uuid4()),
        operation="generate_response",
        context=context,
        parameters={
            "prompt": prompt
        },
        source_component="langgraph",
        target_component="llm"
    )
    
    # Use handler to process request
    handler = MCPHandler()
    handler.register_operation("generate_response", generate_response_operation)
    response = handler.process_request(request)
    
    # Update conversation history in context
    updated_context = response.updated_context
    
    # Update state with response and updated context
    return {
        "response": response.content,
        "mcp_context": updated_context
    }
```

### Building MCP-Aware LangGraph Workflow

```python
# mcp/consumers/langgraph_consumer.py
def create_mcp_workflow():
    """Create a LangGraph workflow that uses MCP for context management"""
    # Define the state type including MCP context
    class MCPAwareState(TypedDict):
        query: str
        parsed_query: Dict[str, Any]
        retrieval_results: Dict[str, Any]
        response: str
        mcp_context: WaterNetworkContext
    
    # Initialize the workflow
    workflow = StateGraph(MCPAwareState)
    
    # Add nodes
    workflow.add_node("query_understanding", mcp_query_understanding_node)
    workflow.add_node("retrieval", mcp_retrieval_node)
    workflow.add_node("response_generation", mcp_response_generation_node)
    
    # Define edges
    workflow.set_entry_point("query_understanding")
    workflow.add_edge("query_understanding", "retrieval")
    workflow.add_edge("retrieval", "response_generation")
    
    # Set the exit point
    workflow.set_finish_point("response_generation")
    
    # Compile the workflow
    return workflow.compile()
```

## 6. MCP Operations Implementation

Let's implement some common MCP operations for water network management:

```python
# mcp/operations/query_operations.py
from typing import Dict, Any, Tuple
from ..schema import WaterNetworkContext
import json
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(temperature=0)

def parse_query_operation(context: WaterNetworkContext, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], WaterNetworkContext]:
    """Operation to parse a query and extract structured information"""
    query = context.current_query
    if not query:
        return {"error": "No query provided"}, context
    
    # Define a prompt for query understanding
    prompt = f"""
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
    
    # Generate the response
    response = llm.invoke(prompt)
    
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
    
    # Update context with workflow information based on intent
    updated_context = context.copy()
    intent = parsed_response.get("intent", "")
    
    if intent in ["maintenance_history", "maintenance_planning"]:
        updated_context.current_workflow = "maintenance"
    elif intent in ["failure_impact", "emergency_response", "isolation_strategy"]:
        updated_context.current_workflow = "emergency"
    else:
        updated_context.current_workflow = "general"
    
    # Add mentioned components to active components
    for component_id in parsed_response.get("component_ids", []):
        if component_id not in updated_context.active_components:
            updated_context.active_components.append(component_id)
    
    # Add mentioned zones to active zones
    location = parsed_response.get("location", "")
    if location and location not in updated_context.active_zones:
        updated_context.active_zones.append(location)
    
    return parsed_response, updated_context
```

```python
# mcp/operations/response_operations.py
from typing import Dict, Any, Tuple
from ..schema import WaterNetworkContext

def generate_response_operation(context: WaterNetworkContext, parameters: Dict[str, Any]) -> Tuple[str, WaterNetworkContext]:
    """Operation to generate a response based on context"""
    prompt = parameters.get("prompt", "")
    if not prompt:
        return "No prompt provided for response generation", context
    
    # Generate response using LLM
    response = llm.invoke(prompt)
    response_text = response.content
    
    # Update context with conversation history
    updated_context = context.copy()
    
    # Add the current exchange to conversation history
    updated_context.conversation_history.append({
        "query": context.current_query,
        "response": response_text
    })
    
    return response_text, updated_context
```

```python
# mcp/operations/maintenance_operations.py
from typing import Dict, Any, Tuple, List
from ..schema import WaterNetworkContext, MaintenanceContext
from datetime import datetime, timedelta

def identify_maintenance_candidates(context: WaterNetworkContext, parameters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], WaterNetworkContext]:
    """Operation to identify components that need maintenance"""
    # Ensure we have a maintenance context
    if not isinstance(context, MaintenanceContext):
        # Convert to maintenance context
        maintenance_context = MaintenanceContext(**context.dict())
    else:
        maintenance_context = context
    
    # Get parameters
    component_types = parameters.get("component_types", ["Valve", "Pump", "Meter"])
    age_threshold_years = parameters.get("age_threshold_years", 10)
    
    # Get current date for age calculation
    current_date = context.reference_time or datetime.now()
    threshold_date = current_date - timedelta(days=365 * age_threshold_years)
    threshold_date_str = threshold_date.strftime("%Y-%m-%d")
    
    # Query Neo4j for old components
    with neo4j_driver.session() as session:
        result = session.run("""
        MATCH (c)
        WHERE any(label IN labels(c) WHERE label IN $component_types)
        AND c.installDate < $threshold_date
        RETURN c.id as id, labels(c) as type, c.installDate as installDate, 
               c.material as material, c.operationalStatus as status
        """, component_types=component_types, threshold_date=threshold_date_str)
        
        candidates = [dict(record) for record in result]
        
        # Query for components with recent issues
        issues_result = session.run("""
        MATCH (c)-[:HAS_MAINTENANCE]->(m)
        WHERE m.findings CONTAINS 'issue' OR m.findings CONTAINS 'problem'
        AND m.date > $recent_date
        RETURN c.id as id, labels(c) as type, m.date as maintenanceDate, 
               m.findings as findings
        """, recent_date=(current_date - timedelta(days=365)).strftime("%Y-%m-%d"))
        
        for record in issues_result:
            component_id = record["id"]
            # Check if already in candidates
            if not any(c["id"] == component_id for c in candidates):
                candidates.append(dict(record))
    
    # Update maintenance context
    maintenance_context.components_to_maintain = candidates
    
    # Get maintenance history for candidates
    maintenance_history = {}
    for candidate in candidates:
        component_id = candidate["id"]
        with neo4j_driver.session() as session:
            history_result = session.run("""
            MATCH (c {id: $component_id})-[:HAS_MAINTENANCE]->(m)
            RETURN m
            ORDER BY m.date DESC
            """, component_id=component_id)
            
            maintenance_history[component_id] = [dict(record["m"]) for record in history_result]
    
    # Update maintenance history in context
    maintenance_context.maintenance_history = maintenance_history
    
    return candidates, maintenance_context
```

## 7. Implementing a Complete MCP-Enabled Water Network Assistant

Let's put everything together to create a complete MCP-enabled water network assistant:

```python
# water_network_assistant.py
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import Dict, Any

from mcp.schema import WaterNetworkContext, MaintenanceContext, EmergencyResponseContext
from mcp.protocol import MCPRequest, MCPResponse
from mcp.handler import MCPHandler
from mcp.providers.neo4j_provider import Neo4jContextProvider
from mcp.consumers.graphrag_consumer import MCPGraphRAG
from mcp.consumers.langgraph_consumer import create_mcp_workflow
from mcp.operations.query_operations import parse_query_operation
from mcp.operations.response_operations import generate_response_operation
from mcp.operations.maintenance_operations import identify_maintenance_candidates

# Load environment variables
load_dotenv()

# Initialize connections
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Neo4j connection
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Initialize embedding model
model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(model_name)

# Initialize vector database
chroma_client = chromadb.PersistentClient(os.getenv("VECTOR_DB_PATH", "./vector_db"))

# Create MCP components
neo4j_provider = Neo4jContextProvider(neo4j_uri, neo4j_user, neo4j_password)
graphrag = MCPGraphRAG(chroma_client, neo4j_driver, embedding_model)
mcp_workflow = create_mcp_workflow()

# Create MCP handler
mcp_handler = MCPHandler()
mcp_handler.register_operation("parse_query", parse_query_operation)
mcp_handler.register_operation("generate_response", generate_response_operation)
mcp_handler.register_operation("identify_maintenance_candidates", identify_maintenance_candidates)

class WaterNetworkAssistant:
    """MCP-enabled assistant for water network management"""
    
    def __init__(self):
        """Initialize the assistant"""
        self.sessions = {}  # Store session contexts
    
    def create_session(self) -> str:
        """Create a new session and return the session ID"""
        session_id = str(uuid.uuid4())
        context = WaterNetworkContext(
            context_id=str(uuid.uuid4()),
            session_id=session_id,
            conversation_history=[]
        )
        self.sessions[session_id] = context
        return session_id
    
    def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process a user query and return a response"""
        # Get or create session context
        if session_id not in self.sessions:
            session_id = self.create_session()
        
        context = self.sessions[session_id]
        
        # Update context with current query
        context.update(current_query=query)
        
        # Create request for query processing
        request = MCPRequest(
            request_id=str(uuid.uuid4()),
            operation="process_query",
            context=context,
            parameters={},
            source_component="assistant",
            target_component="workflow"
        )
        
        # Use workflow to process the request
        workflow_input = {
            "query": query,
            "mcp_context": context
        }
        
        workflow_result = mcp_workflow.invoke(workflow_input)
        
        # Extract response and updated context
        response = workflow_result["response"]
        updated_context = workflow_result["mcp_context"]
        
        # Save updated context to session
        self.sessions[session_id] = updated_context
        
        # Return response with session info
        return {
            "session_id": session_id,
            "response": response,
            "active_components": updated_context.active_components,
            "active_zones": updated_context.active_zones,
            "current_workflow": updated_context.current_workflow
        }
    
    def get_maintenance_recommendations(self, session_id: str) -> Dict[str, Any]:
        """Get maintenance recommendations based on session context"""
        # Get session context
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        context = self.sessions[session_id]
        
        # Ensure we have a maintenance context
        if not isinstance(context, MaintenanceContext):
            # Convert to maintenance context
            maintenance_context = MaintenanceContext(**context.dict())
            self.sessions[session_id] = maintenance_context
            context = maintenance_context
        
        # Create request for maintenance candidate identification
        request = MCPRequest(
            request_id=str(uuid.uuid4()),
            operation="identify_maintenance_candidates",
            context=context,
            parameters={
                "component_types": ["Valve", "Pump", "Meter"],
                "age_threshold_years": 10
            },
            source_component="assistant",
            target_component="maintenance_analyzer"
        )
        
        # Process the request
        response = mcp_handler.process_request(request)
        
        # Update session context
        self.sessions[session_id] = response.updated_context
        
        # Return maintenance recommendations
        return {
            "session_id": session_id,
            "maintenance_candidates": response.content,
            "maintenance_history": response.updated_context.maintenance_history if isinstance(response.updated_context, MaintenanceContext) else {}
        }
    
    def close(self):
        """Close connections"""
        neo4j_driver.close()
```

## 8. Practical Example: Using MCP in a Water Network Application

Let's demonstrate how to use our MCP-enabled water network assistant in a simple application:

```python
# app.py
from water_network_assistant import WaterNetworkAssistant
import json

# Initialize the assistant
assistant = WaterNetworkAssistant()

def main():
    """Simple command-line interface for the water network assistant"""
    print("Water Network Management Assistant")
    print("Type 'exit' to quit, 'maintenance' for maintenance recommendations")
    
    # Create a new session
    session_id = assistant.create_session()
    print(f"Started new session: {session_id}")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'maintenance':
            # Get maintenance recommendations
            result = assistant.get_maintenance_recommendations(session_id)
            
            print("\nAssistant: Here are the maintenance recommendations:")
            candidates = result.get("maintenance_candidates", [])
            
            if candidates:
                for i, candidate in enumerate(candidates[:5], 1):
                    component_type = candidate.get("type", ["Component"])[0]
                    component_id = candidate.get("id", "Unknown")
                    install_date = candidate.get("installDate", "Unknown")
                    
                    print(f"{i}. {component_type} {component_id} (Installed: {install_date})")
                    
                    # Show maintenance history if available
                    history = result.get("maintenance_history", {}).get(component_id, [])
                    if history:
                        print(f"   Last maintenance: {history[0].get('date', 'Unknown')}")
                        print(f"   Findings: {history[0].get('findings', 'None')}")
                
                if len(candidates) > 5:
                    print(f"...and {len(candidates) - 5} more candidates.")
            else:
                print("No maintenance candidates identified at this time.")
            
            continue
        
        # Process regular query
        result = assistant.process_query(session_id, user_input)
        
        # Display response
        print(f"\nAssistant: {result['response']}")
        
        # Show context information
        print("\nContext:")
        print(f"- Active components: {', '.join(result['active_components'][:3])}")
        if len(result['active_components']) > 3:
            print(f"  ...and {len(result['active_components']) - 3} more")
        
        print(f"- Active zones: {', '.join(result['active_zones'])}")
        print(f"- Current workflow: {result['current_workflow']}")

    # Clean up
    assistant.close()
    print("Session ended.")

if __name__ == "__main__":
    main()
```

## 9. Learning Exercise: Implementing MCP for Emergency Response

Let's implement a practical learning exercise focused on emergency response with MCP.

### Exercise: MCP-Enabled Emergency Response Workflow

In this exercise, you'll implement an MCP-enabled emergency response workflow that:

1. Takes information about a component failure
2. Maintains context about the affected area and components
3. Generates an isolation strategy
4. Tracks the emergency response process through multiple interactions

**Step 1: Define the Context Schema**

First, create an enhanced `EmergencyResponseContext` with fields specific to emergency management.

**Step 2: Implement Context Providers**

Create context providers that:
- Retrieve isolation valve information
- Identify affected customers
- Estimate repair times based on component types

**Step 3: Implement Operations**

Create operations for:
- `analyze_failure`: Analyze the impact of a component failure
- `identify_isolation_strategy`: Find valves to close for isolation
- `estimate_restoration_time`: Predict repair duration

**Step 4: Create the MCP-Enabled Workflow**

Implement a LangGraph workflow that uses these operations with MCP context.

**Step 5: Test with a Scenario**

Test with a scenario like "Pipe PIP002 has a major leak" and observe how context is maintained throughout multiple interactions.

## 10. Resources and References

### MCP Documentation
- [Introduction to MCP (Model Context Protocol)](https://github.com/anthropics/anthropic-cookbook/blob/main/mcp/examples/intro.ipynb)
- [MCP for Technical Documentation](https://github.com/anthropics/anthropic-cookbook/blob/main/mcp/examples/technical_documentation.ipynb)

### State Management
- [LLM State Management Patterns](https://www.promptingguide.ai/techniques/state)
- [Context Management for LLMs](https://eugeneyan.com/writing/llm-context/)

### Water Network Resources
- [Water Distribution Network Management](https://www.epa.gov/water-research/water-distribution-system-analysis)
- [Emergency Response Planning for Water Systems](https://www.epa.gov/waterutilityresponse)

This guide has provided you with a comprehensive introduction to the Model Context Protocol (MCP) and how to integrate it into your Water Network Management System. With MCP, you can maintain consistent context across different components, leading to more coherent and effective water network operations.

In the next guide, we'll explore advanced analytics and simulation capabilities to complete the intelligence layer (Phase 4) of your water network management system.
