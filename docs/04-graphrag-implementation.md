# GraphRAG Implementation Guide: Intelligent Retrieval for Water Networks

This guide will walk you through implementing GraphRAG (Graph Retrieval-Augmented Generation) for your water network management system. You'll learn how to combine graph database traversal with vector embeddings to create a powerful, context-aware information retrieval system that enhances LLM interactions.

## 1. Understanding GraphRAG

### What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation is a technique that enhances Large Language Models (LLMs) by providing them with relevant information retrieved from a knowledge base. Traditional RAG follows this process:

1. **Query Understanding**: Process the user's query
2. **Retrieval**: Find relevant information in a knowledge base
3. **Augmentation**: Add the retrieved information to the prompt
4. **Generation**: Have the LLM generate a response using the augmented context

Traditional RAG typically uses vector embeddings and similarity search to find relevant information.

### What is GraphRAG?

GraphRAG extends traditional RAG by incorporating graph traversal techniques:

1. **Query Understanding**: Process the user's query
2. **Dual Retrieval**:
   - **Vector Retrieval**: Find relevant information using embeddings similarity
   - **Graph Traversal**: Navigate through the graph to find connected information
3. **Context Synthesis**: Combine information from both retrieval methods
4. **Augmentation**: Add the synthesized context to the prompt
5. **Generation**: Have the LLM generate a response

### Why GraphRAG for Water Networks?

Water networks are inherently graph-structured and benefit from GraphRAG for several reasons:

1. **Relationship Awareness**: Understands how components are connected
2. **Path-based Analysis**: Can follow flow paths or isolation boundaries
3. **Hierarchical Navigation**: Can move between network levels (zones, areas, etc.)
4. **Multi-hop Reasoning**: Can make inferences across multiple connected components
5. **Contextual Understanding**: Maintains awareness of the network topology

## 2. GraphRAG Architecture for Water Networks

Let's design a GraphRAG architecture specifically for water network intelligence:

### High-Level Architecture

```
┌─────────────────┐      ┌───────────────────────┐      ┌─────────────────┐
│                 │      │                       │      │                 │
│  User Query     │──────▶  Query Understanding  │──────▶  Intent Router  │
│                 │      │                       │      │                 │
└─────────────────┘      └───────────────────────┘      └────────┬────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐      ┌───────────────────────┐      ┌─────────────────┐
│                 │      │                       │      │                 │
│  Generation     │◀─────┤  Context Synthesis    │◀─────┤  Retrieval      │
│                 │      │                       │      │                 │
└─────────────────┘      └───────────────────────┘      └─────────────────┘
        │                                                        │
        │                                                        │
        ▼                                                        ▼
┌─────────────────┐                                     ┌─────────────────┐
│                 │                                     │                 │
│  Response       │                                     │  Neo4j Database │
│                 │                                     │                 │
└─────────────────┘                                     └─────────────────┘
```

### Component Details

1. **Query Understanding Module**
   - Parses natural language queries
   - Identifies entities (e.g., valves, zones, pipes)
   - Determines query intent (e.g., path tracing, maintenance history, component attributes)

2. **Intent Router**
   - Routes queries to appropriate retrieval strategies based on intent
   - Determines whether to use vector search, graph traversal, or both

3. **Retrieval Module**
   - **Vector Retrieval**: Uses embeddings to find semantically similar content
   - **Graph Traversal**: Navigates the Neo4j graph based on relationships
   - **Hybrid Retrieval**: Combines both approaches

4. **Context Synthesis**
   - Merges and prioritizes information from different retrieval methods
   - Structures information in a format suitable for the LLM
   - Filters irrelevant or redundant information

5. **Generation Module**
   - Augments the LLM prompt with synthesized context
   - Generates natural language responses

## 3. Setting Up the Environment

### Prerequisites

- Neo4j graph database with water network model (from previous guides)
- Python environment
- Access to embedding models and LLMs

### Installation of Required Libraries

```bash
# Core dependencies
pip install neo4j langchain langchain-community langchain-openai

# Vector embeddings and storage
pip install sentence-transformers chromadb

# LLM access
pip install openai

# Utilities
pip install python-dotenv numpy pandas
```

### Environment Configuration

Create a `.env` file to store your environment variables:

```
# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI API (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key

# Other configurations
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
VECTOR_DB_PATH=./vector_db
```

## 4. Implementing Vector Embeddings for Water Network Components

### Embedding Model Selection

For water networks, we need embeddings that can capture technical terminology. Good options include:

- **all-MiniLM-L6-v2**: Fast and lightweight, good general performance
- **all-mpnet-base-v2**: Better quality but slower
- **domain-specific models**: If available for water/utility domain

### Creating Component Embeddings

Let's create a script to generate embeddings for water network components:

```python
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# Load environment variables
load_dotenv()

# Neo4j connection
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

# Initialize embedding model
model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(model_name)

# Initialize vector database
chroma_client = chromadb.PersistentClient(os.getenv("VECTOR_DB_PATH", "./vector_db"))
collection = chroma_client.get_or_create_collection("water_network_components")

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

def fetch_components():
    with driver.session() as session:
        # Query for various component types
        result = session.run("""
        MATCH (c)
        WHERE c:Valve OR c:Pump OR c:Pipe OR c:Junction OR c:Tank OR 
              c:Reservoir OR c:Source OR c:TreatmentPlant OR c:PumpStation OR
              c:Meter OR c:Sensor OR c:PressureZone OR c:DMA
        RETURN c.id AS id, labels(c) AS labels, properties(c) AS properties
        LIMIT 1000
        """)
        
        return [record for record in result]

def create_component_description(record):
    """Create a textual description of the component for embedding"""
    component_id = record["id"]
    labels = record["labels"]
    properties = record["properties"]
    
    # Create a descriptive text for this component
    primary_type = labels[0] if labels else "Component"
    
    description = f"{primary_type} {component_id}: "
    
    # Add important properties based on component type
    if "Valve" in labels:
        description += f"A {properties.get('valveType', 'unknown type')} valve "
        description += f"with diameter {properties.get('diameter', 'unknown')}mm, "
        description += f"status: {properties.get('operationalStatus', 'unknown')}. "
        if 'installDate' in properties:
            description += f"Installed on {properties['installDate']}. "
    
    elif "Pipe" in labels:
        description += f"A {properties.get('material', 'unknown material')} pipe "
        description += f"with diameter {properties.get('diameter', 'unknown')}mm, "
        description += f"length {properties.get('length', 'unknown')}m. "
        if 'installDate' in properties:
            description += f"Installed on {properties['installDate']}. "
    
    # Add descriptions for other component types...
    
    # Add all remaining properties
    description += "Additional details: "
    for key, value in properties.items():
        if key not in ['id', 'valveType', 'diameter', 'operationalStatus', 'installDate', 
                       'material', 'length']:
            description += f"{key}: {value}, "
    
    return description.strip(", ")

def create_embeddings():
    components = fetch_components()
    
    ids = []
    descriptions = []
    metadatas = []
    
    for component in components:
        component_id = component["id"]
        description = create_component_description(component)
        
        ids.append(component_id)
        descriptions.append(description)
        metadatas.append({
            "id": component_id,
            "type": component["labels"][0] if component["labels"] else "Unknown",
            # Add other useful metadata
        })
    
    # Generate embeddings and add to the collection
    embeddings = embedding_model.encode(descriptions)
    
    # Store in vector database
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=descriptions
    )
    
    print(f"Created embeddings for {len(ids)} components")

# Execute the embedding creation
if __name__ == "__main__":
    create_embeddings()
    driver.close()
```

### Embedding Component Relationships

For more complex retrieval, we can also embed relationships between components:

```python
def fetch_relationships():
    with driver.session() as session:
        result = session.run("""
        MATCH (c1)-[r]->(c2)
        WHERE (c1:Valve OR c1:Pump OR c1:Pipe OR c1:Junction OR c1:Tank) AND
              (c2:Valve OR c2:Pump OR c2:Pipe OR c2:Junction OR c2:Tank)
        RETURN c1.id AS source_id, type(r) AS relationship_type, 
               properties(r) AS properties, c2.id AS target_id
        LIMIT 5000
        """)
        
        return [record for record in result]

def create_relationship_description(record):
    """Create a textual description of the relationship for embedding"""
    source_id = record["source_id"]
    target_id = record["target_id"]
    rel_type = record["relationship_type"]
    properties = record["properties"]
    
    description = f"Relationship {source_id} {rel_type} {target_id}: "
    
    # Add properties
    for key, value in properties.items():
        description += f"{key}: {value}, "
    
    return description.strip(", ")

def create_relationship_embeddings():
    relationships = fetch_relationships()
    
    rel_collection = chroma_client.get_or_create_collection("water_network_relationships")
    
    ids = []
    descriptions = []
    metadatas = []
    
    for rel in relationships:
        rel_id = f"{rel['source_id']}_{rel['relationship_type']}_{rel['target_id']}"
        description = create_relationship_description(rel)
        
        ids.append(rel_id)
        descriptions.append(description)
        metadatas.append({
            "source_id": rel["source_id"],
            "relationship_type": rel["relationship_type"],
            "target_id": rel["target_id"]
        })
    
    # Generate embeddings and add to the collection
    embeddings = embedding_model.encode(descriptions)
    
    # Store in vector database
    rel_collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=descriptions
    )
    
    print(f"Created embeddings for {len(ids)} relationships")
```

## 5. Implementing Graph Traversal Strategies

Graph traversal is the core of GraphRAG. Let's implement key traversal patterns for water networks:

### Common Water Network Traversal Patterns

```python
class WaterNetworkTraversal:
    def __init__(self, driver):
        self.driver = driver
    
    def find_upstream_components(self, component_id, max_hops=3):
        """Find components that feed into the specified component"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH path = (c)-[:FEEDS|CONNECTED_TO*1..%d]->(target {id: $component_id})
            RETURN path
            """, component_id=component_id, max_hops=max_hops)
            
            return [record["path"] for record in result]
    
    def find_downstream_components(self, component_id, max_hops=3):
        """Find components that the specified component feeds into"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH path = (source {id: $component_id})-[:FEEDS|CONNECTED_TO*1..%d]->(c)
            RETURN path
            """, component_id=component_id, max_hops=max_hops)
            
            return [record["path"] for record in result]
    
    def find_isolation_valves(self, pipe_id):
        """Find valves needed to isolate a specific pipe"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (p:Pipe {id: $pipe_id})
            CALL {
                WITH p
                MATCH path = (p)-[:CONNECTED_TO*1..5]-(v:Valve)
                WHERE NOT (v)-[:CONNECTED_TO]-(:Pipe)-[:CONNECTED_TO]-(p)
                RETURN v
            }
            RETURN DISTINCT v
            """, pipe_id=pipe_id)
            
            return [record["v"] for record in result]
    
    def find_components_in_zone(self, zone_id):
        """Find all components in a specific zone"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c)-[:PART_OF]->(z {id: $zone_id})
            RETURN c
            """, zone_id=zone_id)
            
            return [record["c"] for record in result]
    
    def find_alternative_paths(self, source_id, target_id, max_paths=3):
        """Find alternative flow paths between components"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (source {id: $source_id}), (target {id: $target_id})
            CALL apoc.algo.allSimplePaths(source, target, "FEEDS|CONNECTED_TO>", 10)
            YIELD path
            RETURN path
            LIMIT $max_paths
            """, source_id=source_id, target_id=target_id, max_paths=max_paths)
            
            return [record["path"] for record in result]
    
    def find_maintenance_history(self, component_id):
        """Find maintenance history for a component"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})-[r:HAS_MAINTENANCE]->(m:Maintenance)
            RETURN m
            ORDER BY m.date DESC
            """, component_id=component_id)
            
            return [record["m"] for record in result]
    
    def find_component_sensors(self, component_id):
        """Find sensors monitoring a component"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})-[:HAS_SENSOR]->(s:Sensor)
            RETURN s
            """, component_id=component_id)
            
            return [record["s"] for record in result]
```

### Multi-hop Reasoning Patterns

For more complex analysis, implement reasoning patterns that involve multiple steps:

```python
def analyze_failure_impact(self, component_id):
    """Analyze the impact of a component failure"""
    # Step 1: Find downstream components
    downstream = self.find_downstream_components(component_id)
    
    # Step 2: Find if there are alternative paths to these components
    affected_components = []
    for path in downstream:
        end_component = path.end_node()
        alternative_paths = self.find_alternative_paths(component_id, end_component.get("id"), 1)
        
        if not alternative_paths or len(alternative_paths) < 2:
            affected_components.append(end_component)
    
    # Step 3: Identify critical customers affected
    critical_customers = []
    with self.driver.session() as session:
        for component in affected_components:
            result = session.run("""
            MATCH (c {id: $component_id})-[:CONNECTED_TO*1..5]->(sc:ServiceConnection)
            WHERE sc.customerType IN ['Hospital', 'School', 'Government', 'Industrial']
            RETURN DISTINCT sc
            """, component_id=component.get("id"))
            
            critical_customers.extend([record["sc"] for record in result])
    
    return {
        "affected_components": affected_components,
        "critical_customers": critical_customers
    }
```

## 6. Implementing the Query Understanding Module

The query understanding module interprets natural language queries and identifies entities and intents:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import json

class QueryUnderstanding:
    def __init__(self, llm=None):
        """Initialize with an LLM for query understanding"""
        self.llm = llm or OpenAI(temperature=0)
    
    def parse_query(self, query):
        """Parse a natural language query about water networks"""
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
        response = self.llm.invoke(prompt.format(query=query))
        
        # Parse JSON response
        try:
            parsed_response = json.loads(response)
            return parsed_response
        except json.JSONDecodeError:
            # If JSON parsing fails, provide a basic fallback
            return {
                "intent": "component_info",
                "component_types": [],
                "component_ids": [],
                "attributes": [],
                "relationships": [],
                "time_period": "",
                "location": ""
            }
```

## 7. Implementing the Intent Router

The intent router determines which retrieval strategies to use based on the query understanding:

```python
class IntentRouter:
    def __init__(self, traversal, vector_retriever):
        """Initialize with traversal and vector retrieval components"""
        self.traversal = traversal
        self.vector_retriever = vector_retriever
    
    def route_query(self, parsed_query):
        """Route a parsed query to the appropriate retrieval strategies"""
        intent = parsed_query.get("intent", "component_info")
        component_ids = parsed_query.get("component_ids", [])
        component_types = parsed_query.get("component_types", [])
        attributes = parsed_query.get("attributes", [])
        relationships = parsed_query.get("relationships", [])
        location = parsed_query.get("location", "")
        
        # Determine retrieval strategies based on intent
        strategies = []
        
        if intent == "component_info":
            # For basic component info, vector search is often sufficient
            strategies.append(("vector", {
                "component_ids": component_ids,
                "component_types": component_types,
                "attributes": attributes
            }))
            
            # If specific IDs are mentioned, also fetch from graph
            if component_ids:
                strategies.append(("graph_component_details", {
                    "component_ids": component_ids
                }))
        
        elif intent == "flow_path":
            # For flow paths, graph traversal is essential
            if len(component_ids) >= 2:
                # If source and target are specified
                strategies.append(("graph_path_finding", {
                    "source_id": component_ids[0],
                    "target_id": component_ids[1]
                }))
            elif len(component_ids) == 1:
                # If only one component is specified, look upstream and downstream
                strategies.append(("graph_upstream_downstream", {
                    "component_id": component_ids[0]
                }))
            
            # Add vector retrieval for general flow patterns
            strategies.append(("vector", {
                "query_terms": ["flow", "path"] + component_types + component_ids
            }))
        
        elif intent == "maintenance_history":
            # For maintenance history, combine graph and vector
            if component_ids:
                strategies.append(("graph_maintenance_history", {
                    "component_ids": component_ids
                }))
            
            # Add vector search for similar maintenance records
            strategies.append(("vector", {
                "query_terms": ["maintenance", "repair", "history"] + component_types + component_ids
            }))
        
        elif intent == "failure_impact":
            # For failure impact, graph analysis is primary
            if component_ids:
                strategies.append(("graph_failure_impact", {
                    "component_id": component_ids[0]
                }))
            
            # Supplement with vector search for similar scenarios
            strategies.append(("vector", {
                "query_terms": ["failure", "impact", "consequence"] + component_types + component_ids
            }))
        
        elif intent == "zone_analysis":
            # For zone analysis, use graph to find components in zone
            if location:
                strategies.append(("graph_zone_components", {
                    "zone_name": location
                }))
            
            # Add vector search for zone information
            strategies.append(("vector", {
                "query_terms": ["zone", "area", "district", location]
            }))
        
        elif intent == "isolation_strategy":
            # For isolation strategies, graph traversal is key
            if component_ids:
                strategies.append(("graph_isolation_valves", {
                    "component_id": component_ids[0]
                }))
            
            # Add vector search for isolation procedures
            strategies.append(("vector", {
                "query_terms": ["isolation", "valve", "shutdown"] + component_types + component_ids
            }))
        
        # Add default vector strategy if no strategies were determined
        if not strategies:
            strategies.append(("vector", {
                "query_terms": component_types + component_ids + attributes
            }))
        
        return strategies
```

## 8. Implementing the Retrieval Module

Now let's implement the retrieval module that performs both vector and graph-based retrieval:

```python
class HybridRetriever:
    def __init__(self, chroma_client, traversal, embedding_model):
        """Initialize with vector DB client, graph traversal, and embedding model"""
        self.chroma_client = chroma_client
        self.traversal = traversal
        self.embedding_model = embedding_model
        
        # Get collections
        self.component_collection = chroma_client.get_collection("water_network_components")
        self.relationship_collection = chroma_client.get_collection("water_network_relationships")
    
    def vector_retrieval(self, query=None, query_terms=None, component_ids=None, component_types=None, 
                         attributes=None, n_results=5):
        """Retrieve relevant information using vector similarity"""
        # Build query string from various inputs
        query_string = query or ""
        
        if query_terms:
            query_string += " " + " ".join(query_terms)
        
        if component_ids:
            query_string += " " + " ".join(component_ids)
        
        if component_types:
            query_string += " " + " ".join(component_types)
        
        if attributes:
            query_string += " " + " ".join(attributes)
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode(query_string).tolist()
        
        # Search components
        component_results = self.component_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Search relationships
        relationship_results = self.relationship_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            "components": component_results,
            "relationships": relationship_results
        }
    
    def graph_component_details(self, component_ids):
        """Retrieve detailed information about specific components"""
        results = []
        
        with self.traversal.driver.session() as session:
            for component_id in component_ids:
                result = session.run("""
                MATCH (c {id: $component_id})
                RETURN c, labels(c) as labels
                """, component_id=component_id)
                
                for record in result:
                    results.append({
                        "component": record["c"],
                        "labels": record["labels"]
                    })
        
        return results
    
    def graph_maintenance_history(self, component_ids):
        """Retrieve maintenance history for components"""
        results = {}
        
        for component_id in component_ids:
            maintenance_records = self.traversal.find_maintenance_history(component_id)
            results[component_id] = maintenance_records
        
        return results
    
    def graph_path_finding(self, source_id, target_id):
        """Find paths between two components"""
        paths = self.traversal.find_alternative_paths(source_id, target_id, 3)
        return paths
    
    def graph_upstream_downstream(self, component_id):
        """Find upstream and downstream components"""
        upstream = self.traversal.find_upstream_components(component_id)
        downstream = self.traversal.find_downstream_components(component_id)
        
        return {
            "upstream": upstream,
            "downstream": downstream
        }
    
    def graph_failure_impact(self, component_id):
        """Analyze impact of component failure"""
        return self.traversal.analyze_failure_impact(component_id)
    
    def graph_zone_components(self, zone_name):
        """Find components in a specific zone"""
        with self.traversal.driver.session() as session:
            result = session.run("""
            MATCH (z)
            WHERE z.name CONTAINS $zone_name OR z.id CONTAINS $zone_name
            MATCH (c)-[:PART_OF]->(z)
            RETURN z, collect(c) as components
            """, zone_name=zone_name)
            
            zones = []
            for record in result:
                zones.append({
                    "zone": record["z"],
                    "components": record["components"]
                })
            
            return zones
    
    def graph_isolation_valves(self, component_id):
        """Find valves needed to isolate a component"""
        with self.traversal.driver.session() as session:
            # Check if component is a pipe
            is_pipe = session.run("""
            MATCH (p:Pipe {id: $component_id})
            RETURN count(p) > 0 as is_pipe
            """, component_id=component_id).single()["is_pipe"]
            
            if is_pipe:
                return self.traversal.find_isolation_valves(component_id)
            else:
                # For non-pipe components, find connected valves
                result = session.run("""
                MATCH (c {id: $component_id})-[:CONNECTED_TO*1..2]-(v:Valve)
                RETURN DISTINCT v
                """, component_id=component_id)
                
                return [record["v"] for record in result]
    
    def execute_retrieval_strategy(self, strategy_type, params):
        """Execute a specific retrieval strategy"""
        if strategy_type == "vector":
            return self.vector_retrieval(**params)
        elif strategy_type == "graph_component_details":
            return self.graph_component_details(**params)
        elif strategy_type == "graph_maintenance_history":
            return self.graph_maintenance_history(**params)
        elif strategy_type == "graph_path_finding":
            return self.graph_path_finding(**params)
        elif strategy_type == "graph_upstream_downstream":
            return self.graph_upstream_downstream(**params)
        elif strategy_type == "graph_failure_impact":
            return self.graph_failure_impact(**params)
        elif strategy_type == "graph_zone_components":
            return self.graph_zone_components(**params)
        elif strategy_type == "graph_isolation_valves":
            return self.graph_isolation_valves(**params)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy_type}")
```

## 9. Implementing the Context Synthesis Module

The context synthesis module combines information from different retrieval methods:

```python
class ContextSynthesizer:
    def __init__(self):
        """Initialize the context synthesizer"""
        pass
    
    def format_component(self, component, labels=None):
        """Format a component as a readable text"""
        if not component:
            return "No component data available"
        
        # Get component type from labels or infer from properties
        component_type = labels[0] if labels else "Component"
        
        # Get component properties
        props = dict(component)
        component_id = props.get("id", "Unknown")
        
        # Create description based on component type
        text = f"{component_type} {component_id}:\n"
        
        # Add key properties based on component type
        if component_type == "Valve":
            text += f"- Type: {props.get('valveType', 'Unknown')}\n"
            text += f"- Diameter: {props.get('diameter', 'Unknown')}mm\n"
            text += f"- Status: {props.get('operationalStatus', 'Unknown')}\n"
            
        elif component_type == "Pipe":
            text += f"- Material: {props.get('material', 'Unknown')}\n"
            text += f"- Diameter: {props.get('diameter', 'Unknown')}mm\n"
            text += f"- Length: {props.get('length', 'Unknown')}m\n"
        
        # Add generic properties
        text += "- Other properties:\n"
        for key, value in props.items():
            if key not in ["id", "valveType", "diameter", "operationalStatus", "material", "length"]:
                text += f"  - {key}: {value}\n"
        
        return text
    
    def format_path(self, path):
        """Format a path as readable text"""
        if not path:
            return "No path data available"
        
        text = "Path:\n"
        
        nodes = path.nodes
        relationships = path.relationships
        
        for i in range(len(nodes)):
            node = nodes[i]
            node_labels = node.labels
            primary_label = list(node_labels)[0] if node_labels else "Node"
            
            text += f"- {primary_label} {node.get('id', 'Unknown')}"
            
            if i < len(relationships):
                rel = relationships[i]
                rel_type = rel.type
                text += f" --[{rel_type}]--> "
            else:
                text += "\n"
        
        return text
    
    def format_maintenance_history(self, maintenance_records):
        """Format maintenance history as readable text"""
        if not maintenance_records:
            return "No maintenance history available"
        
        text = "Maintenance History:\n"
        
        for record in maintenance_records:
            props = dict(record)
            text += f"- Date: {props.get('date', 'Unknown')}\n"
            text += f"  Type: {props.get('type', 'Unknown')}\n"
            text += f"  Findings: {props.get('findings', 'None recorded')}\n"
            text += f"  Personnel: {props.get('personnel', 'Unknown')}\n\n"
        
        return text
    
    def format_failure_impact(self, impact_analysis):
        """Format failure impact analysis as readable text"""
        if not impact_analysis:
            return "No impact analysis available"
        
        affected_components = impact_analysis.get("affected_components", [])
        critical_customers = impact_analysis.get("critical_customers", [])
        
        text = "Failure Impact Analysis:\n"
        
        text += f"- Affected Components: {len(affected_components)}\n"
        for component in affected_components[:5]:  # Limit to first 5 for brevity
            text += f"  - {list(component.labels)[0] if component.labels else 'Component'} {component.get('id', 'Unknown')}\n"
        
        if len(affected_components) > 5:
            text += f"  - and {len(affected_components) - 5} more components...\n"
        
        text += f"- Critical Customers Affected: {len(critical_customers)}\n"
        for customer in critical_customers[:5]:  # Limit to first 5 for brevity
            text += f"  - {customer.get('customerType', 'Customer')} {customer.get('id', 'Unknown')}\n"
        
        if len(critical_customers) > 5:
            text += f"  - and {len(critical_customers) - 5} more customers...\n"
        
        return text
    
    def format_vector_results(self, vector_results, max_results=3):
        """Format vector search results as readable text"""
        if not vector_results:
            return "No vector search results available"
        
        components = vector_results.get("components", {})
        relationships = vector_results.get("relationships", {})
        
        text = "Relevant Information:\n\n"
        
        # Add component information
        if components and "documents" in components:
            text += "Relevant Components:\n"
            for i in range(min(max_results, len(components["documents"][0]))):
                text += f"- {components['documents'][0][i]}\n\n"
        
        # Add relationship information
        if relationships and "documents" in relationships:
            text += "Relevant Relationships:\n"
            for i in range(min(max_results, len(relationships["documents"][0]))):
                text += f"- {relationships['documents'][0][i]}\n\n"
        
        return text
    
    def synthesize_context(self, retrieval_results, query_understanding):
        """Synthesize context from retrieval results"""
        context = ""
        
        # First, add a header about what was asked
        intent = query_understanding.get("intent", "information")
        context += f"Retrieving {intent.replace('_', ' ')} for a water network query.\n\n"
        
        # Process each result based on its type
        for result_type, result_data in retrieval_results.items():
            if result_type == "vector":
                context += self.format_vector_results(result_data)
                context += "\n\n"
            
            elif result_type == "graph_component_details":
                context += "Component Details:\n"
                for item in result_data:
                    context += self.format_component(item["component"], item["labels"])
                    context += "\n"
                context += "\n"
            
            elif result_type == "graph_maintenance_history":
                context += "Maintenance Records:\n"
                for component_id, records in result_data.items():
                    context += f"For component {component_id}:\n"
                    context += self.format_maintenance_history(records)
                    context += "\n"
                context += "\n"
            
            elif result_type == "graph_path_finding":
                context += "Path Analysis:\n"
                for i, path in enumerate(result_data):
                    context += f"Path {i+1}:\n"
                    context += self.format_path(path)
                    context += "\n"
                context += "\n"
            
            elif result_type == "graph_upstream_downstream":
                context += "Connected Components:\n"
                
                context += "Upstream Components:\n"
                for path in result_data.get("upstream", [])[:3]:  # Limit to first 3 for brevity
                    context += self.format_path(path)
                    context += "\n"
                
                context += "Downstream Components:\n"
                for path in result_data.get("downstream", [])[:3]:  # Limit to first 3 for brevity
                    context += self.format_path(path)
                    context += "\n"
                
                context += "\n"
            
            elif result_type == "graph_failure_impact":
                context += self.format_failure_impact(result_data)
                context += "\n\n"
            
            elif result_type == "graph_zone_components":
                context += "Zone Analysis:\n"
                for zone_data in result_data:
                    zone = zone_data.get("zone", {})
                    components = zone_data.get("components", [])
                    
                    context += f"Zone {zone.get('id', 'Unknown')} ({zone.get('name', 'Unnamed')}):\n"
                    context += f"- Contains {len(components)} components\n"
                    
                    # Add details of first few components
                    for component in components[:5]:  # Limit to first 5 for brevity
                        component_type = list(component.labels)[0] if component.labels else "Component"
                        context += f"  - {component_type} {component.get('id', 'Unknown')}\n"
                    
                    if len(components) > 5:
                        context += f"  - and {len(components) - 5} more components...\n"
                    
                    context += "\n"
                context += "\n"
            
            elif result_type == "graph_isolation_valves":
                context += "Isolation Valves:\n"
                for i, valve in enumerate(result_data):
                    context += f"- Valve {valve.get('id', 'Unknown')}: "
                    context += f"{valve.get('valveType', 'Unknown type')}, "
                    context += f"Status: {valve.get('operationalStatus', 'Unknown')}\n"
                
                context += "\n"
        
        return context
```

## 10. Implementing the Generation Module

Finally, let's implement the module that augments LLM prompts with our synthesized context:

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class ResponseGenerator:
    def __init__(self, llm=None):
        """Initialize the response generator with an LLM"""
        self.llm = llm or ChatOpenAI(temperature=0.1)
    
    def generate_response(self, query, context, query_understanding):
        """Generate a response to the query using the provided context"""
        intent = query_understanding.get("intent", "component_info")
        
        # Create a prompt template based on the intent
        if intent == "component_info":
            prompt_template = """
            You are an AI assistant for a water utility company. A user is asking about components in the water network.
            Use the following retrieved information to answer their question.
            
            Retrieved information:
            {context}
            
            User question: {query}
            
            Provide a clear, concise answer focusing on the specific component information requested.
            Include relevant technical details but explain them in an accessible way.
            If the information is not available in the retrieved context, acknowledge this and suggest what might help.
            
            Answer:
            """
        
        elif intent == "flow_path":
            prompt_template = """
            You are an AI assistant for a water utility company. A user is asking about water flow paths in the network.
            Use the following retrieved information to answer their question.
            
            Retrieved information:
            {context}
            
            User question: {query}
            
            Provide a clear description of the flow path(s) identified. Explain:
            - The sequence of components water flows through
            - Any control points (valves, pumps) along the path
            - Direction of flow
            - Any alternative paths if available
            
            If the information is not available in the retrieved context, acknowledge this and suggest what might help.
            
            Answer:
            """
        
        elif intent == "maintenance_history":
            prompt_template = """
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
            
            If the information is not available in the retrieved context, acknowledge this and suggest what might help.
            
            Answer:
            """
        
        elif intent == "failure_impact":
            prompt_template = """
            You are an AI assistant for a water utility company. A user is asking about the impact of component failures.
            Use the following retrieved information to answer their question.
            
            Retrieved information:
            {context}
            
            User question: {query}
            
            Provide an analysis of the potential failure impact that addresses:
            - Which components would be affected
            - Whether there are redundant systems or backup paths
            - Critical customers or areas that would lose service
            - Estimated severity of the disruption
            - Potential mitigation strategies
            
            If the information is not available in the retrieved context, acknowledge this and suggest what might help.
            
            Answer:
            """
        
        else:
            # Generic prompt for other intents
            prompt_template = """
            You are an AI assistant for a water utility company. A user has asked a question about the water network.
            Use the following retrieved information to answer their question.
            
            Retrieved information:
            {context}
            
            User question: {query}
            
            Provide a helpful, accurate response based on the retrieved information.
            If the information is not available in the retrieved context, acknowledge this and suggest what might help.
            
            Answer:
            """
        
        # Create the prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )
        
        # Generate the response
        response = self.llm.invoke(prompt.format(context=context, query=query))
        
        return response.content
```

## 11. Putting It All Together: The GraphRAG System

Now let's integrate all the components into a complete GraphRAG system:

```python
class WaterNetworkGraphRAG:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, 
                 embedding_model_name="all-MiniLM-L6-v2",
                 vector_db_path="./vector_db"):
        """Initialize the GraphRAG system"""
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(vector_db_path)
        
        # Initialize components
        self.traversal = WaterNetworkTraversal(self.driver)
        self.query_understanding = QueryUnderstanding()
        self.retriever = HybridRetriever(self.chroma_client, self.traversal, self.embedding_model)
        self.router = IntentRouter(self.traversal, self.retriever)
        self.synthesizer = ContextSynthesizer()
        self.generator = ResponseGenerator()
    
    def process_query(self, query):
        """Process a natural language query about the water network"""
        # Step 1: Understand the query
        parsed_query = self.query_understanding.parse_query(query)
        
        # Step 2: Route to retrieval strategies
        retrieval_strategies = self.router.route_query(parsed_query)
        
        # Step 3: Execute retrieval strategies
        retrieval_results = {}
        for strategy_type, params in retrieval_strategies:
            result = self.retriever.execute_retrieval_strategy(strategy_type, params)
            retrieval_results[strategy_type] = result
        
        # Step 4: Synthesize context
        context = self.synthesizer.synthesize_context(retrieval_results, parsed_query)
        
        # Step 5: Generate response
        response = self.generator.generate_response(query, context, parsed_query)
        
        return response
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
```

## 12. Example Usage of the GraphRAG System

Here's how to use the GraphRAG system:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection details
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize the GraphRAG system
graph_rag = WaterNetworkGraphRAG(
    neo4j_uri=neo4j_uri,
    neo4j_user=neo4j_user,
    neo4j_password=neo4j_password
)

# Process a query
query = "What would happen if valve VLV001 failed?"
response = graph_rag.process_query(query)
print(response)

# Try another query
query = "Show me the maintenance history of the North Hill Tank"
response = graph_rag.process_query(query)
print(response)

# Close the connection when done
graph_rag.close()
```

## 13. Practical Exercises

To help you learn GraphRAG for water networks, here are some exercises:

### Exercise 1: Basic Component Querying
Implement a simplified version of the GraphRAG system that can answer basic questions about components:
- "What is valve VLV001?"
- "Tell me about the North Hill Tank"
- "What type of pipe is PIP003?"

### Exercise 2: Flow Path Analysis
Extend your system to trace water flow through the network:
- "How does water get from Highland Reservoir to North Hill Tank?"
- "What components are downstream of pump PMP001?"
- "Show me all paths from the main treatment plant to Junction JCT005"

### Exercise 3: Maintenance Analysis
Add functionality to analyze maintenance patterns:
- "When was the last time PRV001 was maintained?"
- "Which valves have maintenance issues related to their diaphragms?"
- "Are there any components that have required maintenance more than twice in the past year?"

### Exercise 4: Failure Impact Assessment
Implement failure impact analysis:
- "What would happen if pipe PIP002 broke?"
- "Which customers would be affected if the High Service Pump Station went offline?"
- "What's the most critical valve in the High Pressure Zone?"

## 14. Integration with LangGraph (Preview for Next Guide)

In the next guide, we'll cover how to integrate this GraphRAG system with LangGraph for orchestration. Here's a preview:

```python
from langgraph.graph import StateGraph
import json

# Define the state
class WaterNetworkAnalysisState(TypedDict):
    query: str
    parsed_query: dict
    retrieval_results: dict
    context: str
    follow_up_questions: list[str]
    response: str
    conversation_history: list[dict]

# Define nodes in the graph
def parse_query(state):
    query = state["query"]
    parsed_query = query_understanding.parse_query(query)
    return {"parsed_query": parsed_query}

def retrieve_information(state):
    parsed_query = state["parsed_query"]
    retrieval_strategies = router.route_query(parsed_query)
    
    retrieval_results = {}
    for strategy_type, params in retrieval_strategies:
        result = retriever.execute_retrieval_strategy(strategy_type, params)
        retrieval_results[strategy_type] = result
    
    return {"retrieval_results": retrieval_results}

def synthesize_context(state):
    retrieval_results = state["retrieval_results"]
    parsed_query = state["parsed_query"]
    context = synthesizer.synthesize_context(retrieval_results, parsed_query)
    return {"context": context}

def generate_response(state):
    query = state["query"]
    context = state["context"]
    parsed_query = state["parsed_query"]
    response = generator.generate_response(query, context, parsed_query)
    return {"response": response}

def generate_follow_ups(state):
    query = state["query"]
    response = state["response"]
    context = state["context"]
    
    # Generate potential follow-up questions
    prompt = f"""
    Based on the user's query: {query}
    And your response: {response}
    Generate 3 potential follow-up questions the user might ask next.
    Return them as a JSON array of strings.
    """
    
    follow_up_result = llm.invoke(prompt)
    try:
        follow_up_questions = json.loads(follow_up_result.content)
    except:
        follow_up_questions = []
    
    return {"follow_up_questions": follow_up_questions}

def update_history(state):
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "query": state["query"],
        "response": state["response"]
    })
    return {"conversation_history": conversation_history}

# Build the graph
workflow = StateGraph(WaterNetworkAnalysisState)

# Add nodes
workflow.add_node("parse_query", parse_query)
workflow.add_node("retrieve_information", retrieve_information)
workflow.add_node("synthesize_context", synthesize_context)
workflow.add_node("generate_response", generate_response)
workflow.add_node("generate_follow_ups", generate_follow_ups)
workflow.add_node("update_history", update_history)

# Add edges
workflow.set_entry_point("parse_query")
workflow.add_edge("parse_query", "retrieve_information")
workflow.add_edge("retrieve_information", "synthesize_context")
workflow.add_edge("synthesize_context", "generate_response")
workflow.add_edge("generate_response", "generate_follow_ups")
workflow.add_edge("generate_follow_ups", "update_history")

# Compile the graph
app = workflow.compile()
```

## 15. Resources and References

### Graph Databases and Neo4j
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)
- [Neo4j Developer Guides](https://neo4j.com/developer/get-started/)

### Vector Databases and Embeddings
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

### Retrieval-Augmented Generation
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Graph RAG Papers](https://arxiv.org/search/?query=graph+RAG&searchtype=all)

### Large Language Models
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

### Water Network Resources
- [EPANET Water Network Simulation](https://www.epa.gov/water-research/epanet)
- [International Water Association](https://iwa-network.org/)

By following this guide, you've implemented a GraphRAG system for water network intelligence. This system combines the structural knowledge in your Neo4j graph database with the semantic understanding of vector embeddings, enabling powerful, context-aware information retrieval for water network management. In the next guide, we'll explore how to use LangGraph to orchestrate complex reasoning workflows on top of this GraphRAG foundation.
