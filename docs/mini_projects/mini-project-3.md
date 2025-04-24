# Mini-Project 3: Component Information Retriever

## Overview
In this mini-project, you'll implement a basic retrieval system for water network components that showcases the principles of Retrieval-Augmented Generation (RAG). The system will combine graph database queries with vector search to provide comprehensive information about water network components when queried in natural language.

## Learning Objectives
- Understand and implement Retrieval-Augmented Generation (RAG)
- Create and use vector embeddings for water network components
- Build a simple hybrid retrieval system
- Combine graph database queries with vector search
- Create a natural language interface for component information

## Dependencies
- **Phase 2 Content**: Complete at least Weeks 1-3 of the Retrieval Layer Phase
- **Skills Required**: Python, Neo4j, Vector Embeddings, RAG concepts
- **Previous Mini-Projects**: Mini-Project 1 (or any data in Neo4j)

## Estimated Time: 2 weeks

## Project Steps

### Step 1: Set Up Environment
1. Create a new Python project with the following structure:
```
component-retriever/
├── retriever/
│   ├── __init__.py
│   ├── database.py
│   ├── embeddings.py
│   ├── retrieval.py
│   └── app.py
├── data/
│   └── vector_store/
├── tests/
│   ├── __init__.py
│   └── test_retrieval.py
├── README.md
└── requirements.txt
```

2. Install required packages:
```
pip install neo4j sentence-transformers langchain langchain-community chromadb flask
```

### Step 2: Implement Neo4j Connector
1. Create the Neo4j database connector in `database.py`:

```python
from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_component_by_id(self, component_id):
        """Get a component by its ID"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})
            RETURN c, labels(c) as labels
            """, component_id=component_id)
            
            record = result.single()
            if not record:
                return None
            
            component = dict(record["c"])
            component["type"] = record["labels"][0]
            return component
    
    def get_component_connections(self, component_id):
        """Get connections for a component"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})-[r]-(connected)
            RETURN type(r) AS relationship_type, connected.id AS connected_id, 
                   labels(connected) AS connected_labels
            """, component_id=component_id)
            
            connections = []
            for record in result:
                connections.append({
                    "relationship_type": record["relationship_type"],
                    "connected_id": record["connected_id"],
                    "connected_type": record["connected_labels"][0]
                })
            
            return connections
    
    def get_component_maintenance_history(self, component_id):
        """Get maintenance history for a component"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})-[:HAS_MAINTENANCE]->(m)
            RETURN m
            ORDER BY m.date DESC
            """, component_id=component_id)
            
            return [dict(record["m"]) for record in result]
    
    def get_all_components(self, component_type=None, limit=1000):
        """Get all components, optionally filtered by type"""
        with self.driver.session() as session:
            if component_type:
                query = f"""
                MATCH (c:{component_type})
                RETURN c, labels(c) as labels
                LIMIT $limit
                """
            else:
                query = """
                MATCH (c)
                WHERE ANY(label IN labels(c) WHERE label IN ['Valve', 'Pipe', 'Pump', 'Tank', 'Junction', 'Reservoir'])
                RETURN c, labels(c) as labels
                LIMIT $limit
                """
            
            result = session.run(query, limit=limit)
            
            components = []
            for record in result:
                component = dict(record["c"])
                component["type"] = record["labels"][0]
                components.append(component)
            
            return components
    
    def get_components_by_zone(self, zone_id):
        """Get components in a zone"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (z {id: $zone_id})
            MATCH (c)-[:PART_OF]->(z)
            RETURN c, labels(c) as labels
            """, zone_id=zone_id)
            
            components = []
            for record in result:
                component = dict(record["c"])
                component["type"] = record["labels"][0]
                components.append(component)
            
            return components
```

### Step 3: Implement Embeddings Generation
1. Create the embeddings module in `embeddings.py`:

```python
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class ComponentEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_directory="./data/vector_store"):
        """Initialize the embeddings generator"""
        self.model = SentenceTransformer(model_name)
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="water_components",
            embedding_function=self._embedding_function
        )
    
    def _embedding_function(self, texts):
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts).tolist()
    
    def _generate_component_description(self, component):
        """Generate a text description of a component for embedding"""
        component_type = component.get("type", "Unknown")
        component_id = component.get("id", "Unknown")
        
        description = f"{component_type} {component_id}: "
        
        # Add type-specific details
        if component_type == "Valve":
            description += f"A {component.get('valveType', 'unknown type')} valve "
            description += f"with diameter {component.get('diameter', 'unknown')}mm, "
            description += f"status: {component.get('status', 'unknown')}. "
        
        elif component_type == "Pipe":
            description += f"A {component.get('material', 'unknown material')} pipe "
            description += f"with diameter {component.get('diameter', 'unknown')}mm, "
            description += f"length {component.get('length', 'unknown')}m. "
        
        elif component_type == "Pump":
            description += f"A {component.get('pumpType', 'unknown type')} pump "
            description += f"with capacity {component.get('capacity', 'unknown')}, "
            description += f"power: {component.get('power', 'unknown')}. "
        
        elif component_type == "Tank":
            description += f"A water tank with capacity {component.get('capacity', 'unknown')}, "
            description += f"current level: {component.get('level', 'unknown')}. "
        
        # Add installation date if available
        if "installDate" in component:
            description += f"Installed on {component['installDate']}. "
        
        # Add all remaining properties
        description += "Additional details: "
        for key, value in component.items():
            if key not in ["id", "type", "valveType", "diameter", "status", "installDate", "material", "length", "pumpType", "capacity", "power", "level"]:
                description += f"{key}: {value}, "
        
        return description.strip(", ")
    
    def add_components(self, components):
        """Add components to the vector store"""
        ids = []
        descriptions = []
        metadatas = []
        
        for component in components:
            component_id = component.get("id")
            if not component_id:
                continue
            
            description = self._generate_component_description(component)
            
            ids.append(component_id)
            descriptions.append(description)
            metadatas.append({
                "id": component_id,
                "type": component.get("type", "Unknown")
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=descriptions,
            metadatas=metadatas
        )
        
        return len(ids)
    
    def search_components(self, query, n_results=5):
        """Search for components similar to the query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "type": results["metadatas"][0][i]["type"],
                    "description": results["documents"][0][i],
                    "score": results["distances"][0][i] if "distances" in results else None
                })
        
        return formatted_results
```

### Step 4: Implement Hybrid Retrieval System
1. Create the retrieval module in `retrieval.py`:

```python
from .database import Neo4jConnector
from .embeddings import ComponentEmbeddings

class ComponentRetriever:
    def __init__(self, neo4j_connector, component_embeddings):
        """Initialize the component retriever"""
        self.neo4j = neo4j_connector
        self.embeddings = component_embeddings
    
    def _format_component_info(self, component, connections=None, maintenance_history=None):
        """Format component information for display"""
        if not component:
            return None
        
        info = {
            "id": component.get("id"),
            "type": component.get("type"),
            "properties": {k: v for k, v in component.items() if k not in ["id", "type"]}
        }
        
        if connections:
            info["connections"] = connections
        
        if maintenance_history:
            info["maintenance_history"] = maintenance_history
        
        return info
    
    def get_component_by_id(self, component_id):
        """Get detailed information about a specific component"""
        # Get basic component information
        component = self.neo4j.get_component_by_id(component_id)
        if not component:
            return None
        
        # Get connections
        connections = self.neo4j.get_component_connections(component_id)
        
        # Get maintenance history
        maintenance_history = self.neo4j.get_component_maintenance_history(component_id)
        
        # Format and return
        return self._format_component_info(component, connections, maintenance_history)
    
    def search_components(self, query):
        """Search for components using vector similarity"""
        return self.embeddings.search_components(query)
    
    def answer_query(self, query):
        """Answer a natural language query about components"""
        # First, try to find components by vector search
        vector_results = self.embeddings.search_components(query, n_results=3)
        
        # If no results, return empty response
        if not vector_results:
            return {
                "query": query,
                "results": [],
                "answer": "No relevant components found for your query."
            }
        
        # Get detailed information for the top results
        detailed_results = []
        for result in vector_results:
            component_id = result["id"]
            detailed_info = self.get_component_by_id(component_id)
            if detailed_info:
                detailed_results.append(detailed_info)
        
        # Generate an answer based on the results
        answer = self._generate_answer(query, detailed_results)
        
        return {
            "query": query,
            "results": detailed_results,
            "answer": answer
        }
    
    def _generate_answer(self, query, results):
        """Generate a human-readable answer based on query and results"""
        if not results:
            return "I couldn't find any relevant information to answer your query."
        
        # For this simple implementation, we'll just return a formatted answer
        # In a more advanced system, you would use an LLM to generate a natural language response
        
        if "status" in query.lower() or "condition" in query.lower():
            answer = "I found the following components with status information:\n\n"
            for result in results:
                component_type = result.get("type", "Component")
                component_id = result.get("id", "Unknown")
                status = result.get("properties", {}).get("status", "Unknown")
                answer += f"- {component_type} {component_id}: Status is {status}\n"
        
        elif "install" in query.lower() or "age" in query.lower() or "old" in query.lower():
            answer = "I found the following installation information:\n\n"
            for result in results:
                component_type = result.get("type", "Component")
                component_id = result.get("id", "Unknown")
                install_date = result.get("properties", {}).get("installDate", "Unknown")
                answer += f"- {component_type} {component_id}: Installed on {install_date}\n"
        
        elif "connect" in query.lower() or "link" in query.lower():
            answer = "I found the following connection information:\n\n"
            for result in results:
                component_type = result.get("type", "Component")
                component_id = result.get("id", "Unknown")
                connections = result.get("connections", [])
                answer += f"- {component_type} {component_id} is connected to:\n"
                for conn in connections:
                    answer += f"  - {conn.get('connected_type')} {conn.get('connected_id')} via {conn.get('relationship_type')}\n"
        
        elif "maintenance" in query.lower() or "repair" in query.lower():
            answer = "I found the following maintenance history:\n\n"
            for result in results:
                component_type = result.get("type", "Component")
                component_id = result.get("id", "Unknown")
                maintenance = result.get("maintenance_history", [])
                answer += f"- {component_type} {component_id} maintenance history:\n"
                for m in maintenance[:3]:  # Limit to 3 most recent
                    answer += f"  - Date: {m.get('date', 'Unknown')}, Type: {m.get('type', 'Unknown')}\n"
                if len(maintenance) > 3:
                    answer += f"  - ... and {len(maintenance) - 3} more records\n"
        
        else:
            answer = "Here is information about the most relevant components for your query:\n\n"
            for result in results:
                component_type = result.get("type", "Component")
                component_id = result.get("id", "Unknown")
                properties = result.get("properties", {})
                answer += f"- {component_type} {component_id}:\n"
                for key, value in list(properties.items())[:5]:  # Limit to 5 properties
                    answer += f"  - {key}: {value}\n"
        
        return answer
    
    def update_vector_store(self):
        """Update the vector store with all components from Neo4j"""
        # Get all components
        components = self.neo4j.get_all_components()
        
        # Add to vector store
        count = self.embeddings.add_components(components)
        
        return count
```

### Step 5: Create a Simple Web Interface
1. Implement a Flask web app in `app.py`:

```python
import os
from flask import Flask, request, jsonify, render_template
from .database import Neo4jConnector
from .embeddings import ComponentEmbeddings
from .retrieval import ComponentRetriever

app = Flask(__name__, template_folder="templates")

# Initialize components
neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
embeddings = ComponentEmbeddings()
retriever = ComponentRetriever(neo4j, embeddings)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/components/<component_id>')
def get_component(component_id):
    """Get a specific component by ID"""
    component = retriever.get_component_by_id(component_id)
    if not component:
        return jsonify({"error": f"Component {component_id} not found"}), 404
    
    return jsonify(component)

@app.route('/api/search')
def search_components():
    """Search for components"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    results = retriever.search_components(query)
    return jsonify({"query": query, "results": results})

@app.route('/api/query')
def query_components():
    """Answer a natural language query"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    response = retriever.answer_query(query)
    return jsonify(response)

@app.route('/api/update-vectors', methods=['POST'])
def update_vectors():
    """Update vector store with components from Neo4j"""
    count = retriever.update_vector_store()
    return jsonify({"message": f"Updated vector store with {count} components"})

def create_app():
    """Create and configure the Flask app"""
    return app

# HTML template for the main page
@app.route('/templates')
def templates():
    os.makedirs('retriever/templates', exist_ok=True)
    with open('retriever/templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Component Information Retriever</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-container {
            margin-bottom: 20px;
        }
        input[type="text"] {
            width: 70%;
            padding: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 15px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .results-container {
            margin-top: 20px;
        }
        .answer {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            white-space: pre-line;
        }
        .component {
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .component h3 {
            margin-top: 0;
        }
        .properties {
            margin-left: 20px;
        }
        .connections, .maintenance {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Component Information Retriever</h1>
    <div class="search-container">
        <input type="text" id="query-input" placeholder="Enter your question about water network components...">
        <button id="search-button">Search</button>
    </div>
    <div class="results-container">
        <div id="answer" class="answer" style="display: none;"></div>
        <div id="results"></div>
    </div>

    <script>
        document.getElementById('search-button').addEventListener('click', performSearch);
        document.getElementById('query-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });

        function performSearch() {
            const query = document.getElementById('query-input').value;
            if (!query) return;

            fetch(`/api/query?q=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }

        function displayResults(data) {
            const answerElement = document.getElementById('answer');
            const resultsElement = document.getElementById('results');
            
            // Display answer
            answerElement.textContent = data.answer;
            answerElement.style.display = 'block';
            
            // Clear previous results
            resultsElement.innerHTML = '';
            
            // Display detailed results
            if (data.results && data.results.length > 0) {
                data.results.forEach(component => {
                    const componentDiv = document.createElement('div');
                    componentDiv.className = 'component';
                    
                    // Header
                    const header = document.createElement('h3');
                    header.textContent = `${component.type} ${component.id}`;
                    componentDiv.appendChild(header);
                    
                    // Properties
                    const propertiesDiv = document.createElement('div');
                    propertiesDiv.className = 'properties';
                    Object.entries(component.properties).forEach(([key, value]) => {
                        const propertyP = document.createElement('p');
                        propertyP.textContent = `${key}: ${value}`;
                        propertiesDiv.appendChild(propertyP);
                    });
                    componentDiv.appendChild(propertiesDiv);
                    
                    // Connections
                    if (component.connections && component.connections.length > 0) {
                        const connectionsDiv = document.createElement('div');
                        connectionsDiv.className = 'connections';
                        const connectionsTitle = document.createElement('h4');
                        connectionsTitle.textContent = 'Connections';
                        connectionsDiv.appendChild(connectionsTitle);
                        
                        const connectionsList = document.createElement('ul');
                        component.connections.forEach(connection => {
                            const connectionItem = document.createElement('li');
                            connectionItem.textContent = `${connection.connected_type} ${connection.connected_id} (${connection.relationship_type})`;
                            connectionsList.appendChild(connectionItem);
                        });
                        connectionsDiv.appendChild(connectionsList);
                        componentDiv.appendChild(connectionsDiv);
                    }
                    
                    // Maintenance History
                    if (component.maintenance_history && component.maintenance_history.length > 0) {
                        const maintenanceDiv = document.createElement('div');
                        maintenanceDiv.className = 'maintenance';
                        const maintenanceTitle = document.createElement('h4');
                        maintenanceTitle.textContent = 'Maintenance History';
                        maintenanceDiv.appendChild(maintenanceTitle);
                        
                        const maintenanceList = document.createElement('ul');
                        component.maintenance_history.forEach(maintenance => {
                            const maintenanceItem = document.createElement('li');
                            maintenanceItem.textContent = `${maintenance.date}: ${maintenance.type} - ${maintenance.findings || 'No findings recorded'}`;
                            maintenanceList.appendChild(maintenanceItem);
                        });
                        maintenanceDiv.appendChild(maintenanceList);
                        componentDiv.appendChild(maintenanceDiv);
                    }
                    
                    resultsElement.appendChild(componentDiv);
                });
            }
        }
    </script>
</body>
</html>
        ''')
    return "Templates created"

if __name__ == '__main__':
    # Create templates if they don't exist
    templates()
    # Run the app
    app = create_app()
    app.run(debug=True)
```

### Step 6: Build the Vector Store
1. Create a script to populate the vector store:

```python
# in retriever/__main__.py
import os
from .database import Neo4jConnector
from .embeddings import ComponentEmbeddings
from .retrieval import ComponentRetriever

def main():
    """Build the vector store with all components from Neo4j"""
    # Get environment variables
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
    
    print(f"Connecting to Neo4j at {neo4j_uri}")
    
    # Create components
    neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    embeddings = ComponentEmbeddings()
    retriever = ComponentRetriever(neo4j, embeddings)
    
    # Build vector store
    print("Building vector store from Neo4j components...")
    count = retriever.update_vector_store()
    print(f"Added {count} components to vector store")
    
    # Clean up
    neo4j.close()
    print("Done!")

if __name__ == "__main__":
    main()
```

### Step 7: Test the Retrieval System
1. Create basic tests in `tests/test_retrieval.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from retriever.retrieval import ComponentRetriever

@pytest.fixture
def mock_neo4j():
    neo4j = MagicMock()
    
    # Mock component data
    mock_component = {
        "id": "VLV001",
        "type": "Valve",
        "valveType": "Gate",
        "diameter": 300,
        "status": "Open",
        "installDate": "2015-03-15"
    }
    
    # Mock connections
    mock_connections = [
        {
            "relationship_type": "CONNECTED_TO",
            "connected_id": "PIP001",
            "connected_type": "Pipe"
        }
    ]
    
    # Mock maintenance history
    mock_maintenance = [
        {
            "id": "MNT001",
            "date": "2022-05-15",
            "type": "Inspection",
            "findings": "No issues found"
        }
    ]
    
    # Setup mock returns
    neo4j.get_component_by_id.return_value = mock_component
    neo4j.get_component_connections.return_value = mock_connections
    neo4j.get_component_maintenance_history.return_value = mock_maintenance
    
    return neo4j

@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    
    # Mock search results
    mock_results = [
        {
            "id": "VLV001",
            "type": "Valve",
            "description": "Valve VLV001: A Gate valve with diameter 300mm, status: Open. Installed on 2015-03-15.",
            "score": 0.92
        }
    ]
    
    # Setup mock returns
    embeddings.search_components.return_value = mock_results
    
    return embeddings

@pytest.fixture
def retriever(mock_neo4j, mock_embeddings):
    return ComponentRetriever(mock_neo4j, mock_embeddings)

def test_get_component_by_id(retriever, mock_neo4j):
    # Test retrieving a component by ID
    component = retriever.get_component_by_id("VLV001")
    
    # Verify the Neo4j connector was called
    mock_neo4j.get_component_by_id.assert_called_once_with("VLV001")
    mock_neo4j.get_component_connections.assert_called_once_with("VLV001")
    mock_neo4j.get_component_maintenance_history.assert_called_once_with("VLV001")
    
    # Verify the result
    assert component["id"] == "VLV001"
    assert component["type"] == "Valve"
    assert "connections" in component
    assert "maintenance_history" in component

def test_search_components(retriever, mock_embeddings):
    # Test searching for components
    results = retriever.search_components("gate valve")
    
    # Verify the embeddings service was called
    mock_embeddings.search_components.assert_called_once_with("gate valve")
    
    # Verify the results
    assert len(results) == 1
    assert results[0]["id"] == "VLV001"
    assert results[0]["type"] == "Valve"

def test_answer_query(retriever, mock_neo4j, mock_embeddings):
    # Test answering a query
    response = retriever.answer_query("What is the status of valve VLV001?")
    
    # Verify the search was performed
    mock_embeddings.search_components.assert_called_once()
    
    # Verify component details were retrieved
    mock_neo4j.get_component_by_id.assert_called_once_with("VLV001")
    
    # Verify the response structure
    assert "query" in response
    assert "results" in response
    assert "answer" in response
    assert len(response["results"]) == 1
    assert "status" in response["answer"].lower()
```

### Step 8: Run and Test
1. Populate the vector store with components from Neo4j:
```bash
python -m retriever
```

2. Run the Flask application:
```bash
FLASK_APP=retriever.app flask run
```

3. Open a web browser and navigate to `http://localhost:5000`

4. Try asking questions like:
   - "What is the status of valve VLV001?"
   - "Show me information about pump PMP001"
   - "What components are connected to junction JCT001?"
   - "When was the North Hill Tank installed?"
   - "What maintenance has been done on pipe PIP002?"

### Deliverables
1. A functional component information retrieval system
2. Vector embeddings for water network components
3. A simple web interface for natural language queries
4. Integration with Neo4j for detailed component information
5. Tests for the retrieval functionality

## Extensions
1. Add an LLM integration for more natural language responses
2. Implement support for more complex questions (e.g., comparisons between components)
3. Add multi-hop reasoning to follow relationships in the graph
4. Create a visualization of the retrieved components and their connections
5. Implement relevance feedback to improve search results
6. Add support for component-type-specific queries and responses
7. Implement a chat interface with conversation history

## Relation to Main Project
This mini-project implements a simplified version of the GraphRAG concept from Phase 2 of the main project. It demonstrates how to combine graph database queries with vector search to create a more intelligent retrieval system. The skills you develop in implementing this mini-project will directly support the full GraphRAG implementation in the main project. Additionally, the component information retriever you build can be used as a building block for the more comprehensive water network intelligence system.
