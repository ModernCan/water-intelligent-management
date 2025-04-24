# Mini-Project 1: Basic Water Network Graph Builder

## Overview
This mini-project guides you through creating a simple web-based tool to build and visualize a small water network in Neo4j. You'll practice fundamental Neo4j concepts while creating a practical tool you can use throughout the learning journey.

## Learning Objectives
- Apply basic Neo4j concepts in a practical context
- Understand graph data modeling for water networks
- Create and visualize nodes and relationships
- Practice writing and executing Cypher queries

## Dependencies
- **Phase 1 Content**: Complete at least Weeks 1-4 of the Foundation Phase
- **Skills Required**: Basic Python, HTML/CSS, JavaScript, Neo4j basics

## Estimated Time: 1-2 weeks

## Project Steps

### Step 1: Setup Development Environment
1. Install Neo4j Desktop
2. Set up a new project and database
3. Install the APOC plugin
4. Create a basic Python web server using Flask

### Step 2: Create the Data Model
1. Design a simplified water network data model with these components:
   - Valves (with properties: id, type, diameter, status)
   - Pipes (with properties: id, material, diameter, length)
   - Junctions (with properties: id, elevation)
   - Tanks (with properties: id, capacity, level)
   - Relationships: CONNECTED_TO

2. Implement the model in Neo4j with Cypher constraints:

```cypher
CREATE CONSTRAINT FOR (v:Valve) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Pipe) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (j:Junction) REQUIRE j.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:Tank) REQUIRE t.id IS UNIQUE;
```

### Step 3: Build the Web Interface
1. Create a simple HTML interface with:
   - A canvas for network visualization
   - Forms for adding new components
   - A panel for displaying component properties
   - A simple query interface

2. Use Flask to serve the web interface:

```python
from flask import Flask, render_template, request, jsonify
from neo4j import GraphDatabase

app = Flask(__name__)

# Neo4j connection
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

@app.route('/')
def index():
    return render_template('index.html')

# Add more routes for components and queries
```

### Step 4: Implement Component Creation
1. Create API endpoints for adding components:

```python
@app.route('/api/components', methods=['POST'])
def add_component():
    data = request.json
    component_type = data.get('type')
    properties = data.get('properties', {})
    
    # Create component in Neo4j
    with driver.session() as session:
        if component_type == 'Valve':
            session.run("""
            CREATE (v:Valve {id: $id, type: $valve_type, diameter: $diameter, status: $status})
            """, id=properties.get('id'), valve_type=properties.get('type'), 
            diameter=properties.get('diameter'), status=properties.get('status'))
        # Add other component types...
    
    return jsonify({"status": "success"})
```

2. Add JavaScript code to handle component creation from the UI

### Step 5: Implement Relationship Creation
1. Create API endpoint for connecting components:

```python
@app.route('/api/relationships', methods=['POST'])
def add_relationship():
    data = request.json
    from_id = data.get('from_id')
    to_id = data.get('to_id')
    
    # Create relationship in Neo4j
    with driver.session() as session:
        session.run("""
        MATCH (a), (b)
        WHERE a.id = $from_id AND b.id = $to_id
        CREATE (a)-[:CONNECTED_TO]->(b)
        """, from_id=from_id, to_id=to_id)
    
    return jsonify({"status": "success"})
```

### Step 6: Implement Network Visualization
1. Use a JavaScript library like vis.js or D3.js to visualize the network
2. Create an API endpoint to fetch the network data:

```python
@app.route('/api/network', methods=['GET'])
def get_network():
    with driver.session() as session:
        # Get all nodes
        nodes_result = session.run("""
        MATCH (n)
        RETURN n, labels(n) as labels
        """)
        
        nodes = []
        for record in nodes_result:
            node = record["n"]
            labels = record["labels"]
            nodes.append({
                "id": node.get("id"),
                "label": node.get("id"),
                "group": labels[0] if labels else "Unknown",
                "properties": dict(node)
            })
        
        # Get all relationships
        relationships_result = session.run("""
        MATCH (a)-[r]->(b)
        RETURN a.id as source, b.id as target, type(r) as type
        """)
        
        links = []
        for record in relationships_result:
            links.append({
                "from": record["source"],
                "to": record["target"],
                "label": record["type"]
            })
        
        return jsonify({"nodes": nodes, "links": links})
```

### Step 7: Implement Query Interface
1. Create a text area for entering Cypher queries
2. Add an API endpoint to execute queries and return results

```python
@app.route('/api/query', methods=['POST'])
def execute_query():
    query = request.json.get('query')
    
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
            return jsonify({"status": "success", "results": records})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
```

### Step 8: Testing and Validation
1. Test adding different components
2. Test creating connections between components
3. Verify that the visualization correctly shows the network
4. Test running common queries:
   - Find all valves
   - Find pipes connected to a specific junction
   - Find the path between two components

### Deliverables
1. A working web application that allows:
   - Adding water network components
   - Creating connections between components
   - Visualizing the network
   - Running queries against the network

## Extensions
1. Add component editing and deletion functionality
2. Implement component property editing
3. Add color-coding based on component status
4. Add validation to prevent invalid connections
5. Implement export/import functionality to save your network configurations

## Relation to Main Project
This mini-project reinforces the Neo4j graph database fundamentals from Phase 1, while creating a tool you can use to visualize and experiment with water networks throughout the project. The skills practiced here will be essential for the data modeling and query work in the main project.
