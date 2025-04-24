# Mini-Project 4: Multi-Hop Graph Reasoning Tool

## Overview
In this mini-project, you'll build a multi-hop reasoning tool that can answer complex questions about your water network by traversing multiple paths in the graph database. This tool will demonstrate the power of combining graph traversal with semantic understanding to reason about relationships that may not be directly connected.

## Learning Objectives
- Implement multi-hop reasoning algorithms
- Understand path-finding and graph traversal strategies
- Apply semantic understanding to graph queries
- Build a tool that can answer complex questions about indirect relationships
- Integrate graph traversal with natural language processing

## Dependencies
- **Phase 2 Content**: Complete at least Weeks 4-6 of the Retrieval Layer Phase
- **Skills Required**: Python, Neo4j, Graph Algorithms, Basic NLP
- **Previous Mini-Projects**: Mini-Project 3 (Component Information Retriever) helpful but not required

## Estimated Time: 2 weeks

## Project Steps

### Step 1: Project Setup
1. Create a new Python project with the following structure:
```
graph-reasoner/
├── graph_reasoner/
│   ├── __init__.py
│   ├── database.py
│   ├── reasoning.py
│   ├── query_parser.py
│   └── server.py
├── tests/
│   ├── __init__.py
│   └── test_reasoning.py
├── README.md
└── requirements.txt
```

2. Install required packages:
```
pip install neo4j networkx spacy matplotlib flask
python -m spacy download en_core_web_sm
```

### Step 2: Implement Database Connector
1. Create the database connector in `database.py`:

```python
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import os

class Neo4jConnector:
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def get_subgraph(self, start_id, max_hops=2, rel_types=None):
        """Extract a subgraph around a specific component"""
        rel_filter = ""
        if rel_types:
            rel_types_str = "|".join([f":{rel}" for rel in rel_types])
            rel_filter = f"[{rel_types_str}]"
        
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH path = (start {{id: $start_id}})-{rel_filter}*0..{max_hops}-(connected)
            RETURN path
            """, start_id=start_id)
            
            # Create a NetworkX graph
            G = nx.DiGraph()
            
            for record in result:
                path = record["path"]
                # Process path nodes and relationships
                nodes = path.nodes
                relationships = path.relationships
                
                # Add nodes to graph
                for node in nodes:
                    node_id = node["id"]
                    labels = list(node.labels)
                    
                    # Add node attributes
                    attrs = dict(node)
                    attrs["labels"] = labels
                    G.add_node(node_id, **attrs)
                
                # Add edges to graph
                for rel in relationships:
                    start_node = rel.start_node["id"]
                    end_node = rel.end_node["id"]
                    rel_type = rel.type
                    
                    # Add edge attributes
                    attrs = dict(rel)
                    attrs["type"] = rel_type
                    G.add_edge(start_node, end_node, **attrs)
            
            return G
    
    def find_paths(self, start_id, end_id, max_length=3, rel_types=None):
        """Find paths between two components"""
        rel_filter = ""
        if rel_types:
            rel_types_str = "|".join([f":{rel}" for rel in rel_types])
            rel_filter = f"[{rel_types_str}]"
        
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH path = (start {{id: $start_id}})-{rel_filter}*1..{max_length}-(end {{id: $end_id}})
            RETURN path
            ORDER BY length(path)
            LIMIT 10
            """, start_id=start_id, end_id=end_id)
            
            paths = []
            for record in result:
                path = record["path"]
                path_data = {
                    "nodes": [dict(node) for node in path.nodes],
                    "relationships": [dict(rel) for rel in path.relationships],
                    "length": len(path.relationships)
                }
                paths.append(path_data)
            
            return paths
    
    def find_common_connections(self, component_ids, max_hops=2):
        """Find components that connect to all the specified components"""
        if not component_ids or len(component_ids) < 2:
            return []
        
        with self.driver.session() as session:
            # Create a parameter for each component ID
            params = {f"id{i}": component_id for i, component_id in enumerate(component_ids)}
            
            # Build MATCH clauses for each component
            match_clauses = []
            for i, _ in enumerate(component_ids):
                match_clauses.append(f"MATCH (c{i} {{id: $id{i}}})-[*1..{max_hops}]-(common)")
            
            # Build the query
            query = "\n".join(match_clauses) + "\n" + f"""
            WHERE {" AND ".join([f"c{i} <> common" for i in range(len(component_ids))])}
            RETURN common, count(distinct common) as count
            ORDER BY count DESC
            LIMIT 10
            """
            
            result = session.run(query, **params)
            
            common_connections = []
            for record in result:
                common = dict(record["common"])
                common_connections.append(common)
            
            return common_connections
    
    def get_component_property(self, component_id, property_name):
        """Get a specific property of a component"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c {id: $component_id})
            RETURN c[$property_name] AS property_value
            """, component_id=component_id, property_name=property_name)
            
            record = result.single()
            if record:
                return record["property_value"]
            return None
    
    def compare_components(self, component_ids, property_name):
        """Compare a property across multiple components"""
        if not component_ids:
            return []
        
        with self.driver.session() as session:
            # Create parameters for the query
            params = {
                "component_ids": component_ids,
                "property_name": property_name
            }
            
            result = session.run("""
            MATCH (c)
            WHERE c.id IN $component_ids
            RETURN c.id AS component_id, labels(c)[0] AS component_type,
                   c[$property_name] AS property_value
            """, **params)
            
            comparisons = []
            for record in result:
                comparisons.append({
                    "component_id": record["component_id"],
                    "component_type": record["component_type"],
                    "property_value": record["property_value"]
                })
            
            return comparisons
    
    def find_components_by_path(self, start_component, rel_sequence):
        """Find components reached by following a specific sequence of relationships"""
        if not rel_sequence:
            return []
        
        # Build the path pattern
        path_pattern = "-"
        for i, rel in enumerate(rel_sequence):
            if i > 0:
                path_pattern += "-"
            path_pattern += f"[:{rel}]-"
        
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH (start {{id: $start_id}}){path_pattern}(end)
            RETURN end
            LIMIT 100
            """, start_id=start_component)
            
            end_components = []
            for record in result:
                end_components.append(dict(record["end"]))
            
            return end_components
    
    def visualize_subgraph(self, graph, output_file='subgraph.png'):
        """Visualize a NetworkX graph and save to file"""
        plt.figure(figsize=(12, 10))
        
        # Create node colors based on type
        node_colors = []
        node_labels = {}
        for node in graph.nodes():
            node_data = graph.nodes[node]
            labels = node_data.get("labels", [])
            
            # Assign colors based on component type
            if "Valve" in labels:
                node_colors.append('red')
            elif "Pipe" in labels:
                node_colors.append('blue')
            elif "Junction" in labels:
                node_colors.append('green')
            elif "Tank" in labels:
                node_colors.append('purple')
            elif "Pump" in labels:
                node_colors.append('orange')
            else:
                node_colors.append('gray')
            
            # Create node labels with ID and type
            node_type = labels[0] if labels else "Unknown"
            node_labels[node] = f"{node} ({node_type})"
        
        # Draw the graph
        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=700, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrows=True)
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)
        
        # Save to file
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
```

### Step 3: Implement Query Parser
1. Create a simple query parser in `query_parser.py`:

```python
import spacy
import re

class QueryParser:
    def __init__(self):
        """Initialize the query parser"""
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define regex patterns for component IDs
        self.component_id_pattern = re.compile(r'\b([A-Z]{3}\d{3})\b')
        
        # Relationship types for water networks
        self.relationship_types = {
            "connected": "CONNECTED_TO",
            "connects": "CONNECTED_TO",
            "connection": "CONNECTED_TO",
            "feeds": "FEEDS",
            "flow": "FEEDS",
            "flows": "FEEDS",
            "part": "PART_OF",
            "belongs": "PART_OF",
            "contains": "CONTAINS",
            "has": "CONTAINS"
        }
        
        # Component types
        self.component_types = {
            "valve": "Valve",
            "valves": "Valve",
            "pipe": "Pipe",
            "pipes": "Pipe",
            "pump": "Pump",
            "pumps": "Pump",
            "tank": "Tank",
            "tanks": "Tank",
            "junction": "Junction",
            "junctions": "Junction",
            "reservoir": "Reservoir",
            "reservoirs": "Reservoir"
        }
        
        # Properties of interest
        self.properties = {
            "status": "status",
            "condition": "status",
            "diameter": "diameter",
            "size": "diameter",
            "material": "material",
            "length": "length",
            "elevation": "elevation",
            "height": "elevation",
            "pressure": "pressure",
            "flow": "flow",
            "capacity": "capacity",
            "age": "installDate",
            "installation": "installDate",
            "installed": "installDate"
        }
    
    def extract_component_ids(self, query):
        """Extract component IDs from the query"""
        matches = self.component_id_pattern.findall(query)
        return matches
    
    def extract_component_types(self, query):
        """Extract component types from the query"""
        doc = self.nlp(query.lower())
        types = []
        
        for token in doc:
            if token.text in self.component_types:
                types.append(self.component_types[token.text])
        
        return list(set(types))
    
    def extract_relationships(self, query):
        """Extract relationship types from the query"""
        doc = self.nlp(query.lower())
        relationships = []
        
        for token in doc:
            if token.text in self.relationship_types:
                relationships.append(self.relationship_types[token.text])
        
        return list(set(relationships))
    
    def extract_properties(self, query):
        """Extract properties from the query"""
        doc = self.nlp(query.lower())
        properties = []
        
        for token in doc:
            if token.text in self.properties:
                properties.append(self.properties[token.text])
        
        return list(set(properties))
    
    def determine_query_type(self, query):
        """Determine the type of query being asked"""
        query_lower = query.lower()
        
        if "path" in query_lower or "route" in query_lower or "how to get" in query_lower:
            return "path"
        
        if "common" in query_lower or "connect" in query_lower or "both" in query_lower:
            return "common_connections"
        
        if "compare" in query_lower or "difference" in query_lower or "versus" in query_lower or " vs " in query_lower:
            return "compare"
        
        if any(word in query_lower for word in ["upstream", "downstream", "feeds", "supplies", "receives from"]):
            return "flow_direction"
        
        if "affect" in query_lower or "impact" in query_lower or "consequence" in query_lower:
            return "impact_analysis"
        
        if any(word in query_lower for word in ["property", "status", "diameter", "material", "installed"]):
            return "property_query"
        
        return "component_info"  # Default query type
    
    def parse_query(self, query):
        """Parse a natural language query"""
        component_ids = self.extract_component_ids(query)
        component_types = self.extract_component_types(query)
        relationships = self.extract_relationships(query)
        properties = self.extract_properties(query)
        query_type = self.determine_query_type(query)
        
        return {
            "query": query,
            "component_ids": component_ids,
            "component_types": component_types,
            "relationships": relationships,
            "properties": properties,
            "query_type": query_type
        }
```

### Step 4: Implement Multi-Hop Reasoning
1. Create the reasoning module in `reasoning.py`:

```python
from .database import Neo4jConnector
from .query_parser import QueryParser
import networkx as nx
import os

class GraphReasoner:
    def __init__(self, neo4j_connector):
        """Initialize the graph reasoner"""
        self.db = neo4j_connector
        self.parser = QueryParser()
    
    def answer_query(self, query):
        """Answer a natural language query using multi-hop reasoning"""
        # Parse the query
        parsed_query = self.parser.parse_query(query)
        
        # Determine query type and route to appropriate handler
        query_type = parsed_query["query_type"]
        
        if query_type == "path":
            return self._handle_path_query(parsed_query)
        
        elif query_type == "common_connections":
            return self._handle_common_connections_query(parsed_query)
        
        elif query_type == "compare":
            return self._handle_comparison_query(parsed_query)
        
        elif query_type == "flow_direction":
            return self._handle_flow_direction_query(parsed_query)
        
        elif query_type == "impact_analysis":
            return self._handle_impact_analysis_query(parsed_query)
        
        elif query_type == "property_query":
            return self._handle_property_query(parsed_query)
        
        else:  # component_info
            return self._handle_component_info_query(parsed_query)
    
    def _handle_path_query(self, parsed_query):
        """Handle queries about paths between components"""
        component_ids = parsed_query["component_ids"]
        
        if len(component_ids) < 2:
            return {
                "answer_type": "error",
                "message": "Need at least two component IDs to find a path"
            }
        
        # Get the first two component IDs
        start_id = component_ids[0]
        end_id = component_ids[1]
        
        # Get relationships to consider
        relationships = parsed_query["relationships"]
        
        # Find paths between the components
        paths = self.db.find_paths(start_id, end_id, max_length=5, rel_types=relationships)
        
        if not paths:
            return {
                "answer_type": "text",
                "message": f"No path found between {start_id} and {end_id}"
            }
        
        # Format the results
        result = {
            "answer_type": "path",
            "start_id": start_id,
            "end_id": end_id,
            "paths": paths
        }
        
        # Create a combined graph for visualization
        G = nx.DiGraph()
        for path in paths:
            for node in path["nodes"]:
                if "id" in node:
                    node_id = node["id"]
                    G.add_node(node_id, **node)
            
            for i, rel in enumerate(path["relationships"]):
                start_node = path["nodes"][i]["id"]
                end_node = path["nodes"][i+1]["id"]
                rel_type = rel.get("type", "UNKNOWN")
                G.add_edge(start_node, end_node, type=rel_type)
        
        # Visualize the paths
        os.makedirs("static/images", exist_ok=True)
        vis_file = f"static/images/path_{start_id}_to_{end_id}.png"
        self.db.visualize_subgraph(G, vis_file)
        result["visualization"] = vis_file
        
        return result
    
    def _handle_common_connections_query(self, parsed_query):
        """Handle queries about common connections between components"""
        component_ids = parsed_query["component_ids"]
        
        if len(component_ids) < 2:
            return {
                "answer_type": "error",
                "message": "Need at least two component IDs to find common connections"
            }
        
        # Find common connections
        common = self.db.find_common_connections(component_ids, max_hops=2)
        
        if not common:
            components_str = ", ".join(component_ids)
            return {
                "answer_type": "text",
                "message": f"No common connections found for {components_str}"
            }
        
        # Format the results
        result = {
            "answer_type": "common_connections",
            "component_ids": component_ids,
            "common_connections": common
        }
        
        # Get subgraphs for visualization
        G = nx.DiGraph()
        
        for component_id in component_ids:
            subgraph = self.db.get_subgraph(component_id, max_hops=2)
            G = nx.compose(G, subgraph)
        
        # Highlight the common connections
        for node in common:
            if "id" in node:
                node_id = node["id"]
                if node_id in G.nodes:
                    G.nodes[node_id]["highlight"] = True
        
        # Visualize the common connections
        os.makedirs("static/images", exist_ok=True)
        vis_file = f"static/images/common_{'_'.join(component_ids)}.png"
        self.db.visualize_subgraph(G, vis_file)
        result["visualization"] = vis_file
        
        return result
    
    def _handle_comparison_query(self, parsed_query):
        """Handle queries comparing properties of components"""
        component_ids = parsed_query["component_ids"]
        properties = parsed_query["properties"]
        
        if len(component_ids) < 2:
            return {
                "answer_type": "error",
                "message": "Need at least two component IDs to compare"
            }
        
        if not properties:
            return {
                "answer_type": "error",
                "message": "No property specified for comparison"
            }
        
        # Use the first property for comparison
        property_name = properties[0]
        
        # Compare the components
        comparisons = self.db.compare_components(component_ids, property_name)
        
        if not comparisons:
            return {
                "answer_type": "text",
                "message": f"Could not compare {property_name} for the specified components"
            }
        
        # Format the results
        result = {
            "answer_type": "comparison",
            "property": property_name,
            "comparisons": comparisons
        }
        
        return result
    
    def _handle_flow_direction_query(self, parsed_query):
        """Handle queries about flow direction (upstream/downstream)"""
        component_ids = parsed_query["component_ids"]
        
        if not component_ids:
            return {
                "answer_type": "error",
                "message": "No component ID specified for flow direction query"
            }
        
        # Determine if we're looking for upstream or downstream
        query = parsed_query["query"].lower()
        is_upstream = "upstream" in query or "feeds into" in query or "supplies" in query
        
        # Get the first component ID
        component_id = component_ids[0]
        
        # Get the subgraph centered on this component
        subgraph = self.db.get_subgraph(component_id, max_hops=2, rel_types=["FEEDS", "CONNECTED_TO"])
        
        # Find upstream or downstream components
        result = {
            "answer_type": "flow_direction",
            "component_id": component_id,
            "direction": "upstream" if is_upstream else "downstream",
            "components": []
        }
        
        # For upstream, find nodes that have edges to the component
        if is_upstream:
            for node in subgraph.predecessors(component_id):
                node_data = subgraph.nodes[node]
                result["components"].append(node_data)
        
        # For downstream, find nodes that the component has edges to
        else:
            for node in subgraph.successors(component_id):
                node_data = subgraph.nodes[node]
                result["components"].append(node_data)
        
        # Visualize the flow direction
        os.makedirs("static/images", exist_ok=True)
        direction = "upstream" if is_upstream else "downstream"
        vis_file = f"static/images/{direction}_{component_id}.png"
        self.db.visualize_subgraph(subgraph, vis_file)
        result["visualization"] = vis_file
        
        return result
    
    def _handle_impact_analysis_query(self, parsed_query):
        """Handle queries about impact of component failures"""
        component_ids = parsed_query["component_ids"]
        
        if not component_ids:
            return {
                "answer_type": "error",
                "message": "No component ID specified for impact analysis"
            }
        
        # Get the first component ID
        component_id = component_ids[0]
        
        # Get the subgraph centered on this component
        subgraph = self.db.get_subgraph(component_id, max_hops=3)
        
        # Find components that would be affected if this component fails
        # For simplicity, consider all downstream components as affected
        affected_components = []
        
        # BFS to find all nodes reachable from the component
        if component_id in subgraph:
            for node in nx.descendants(subgraph, component_id):
                node_data = subgraph.nodes[node]
                affected_components.append(node_data)
        
        # Format the results
        result = {
            "answer_type": "impact_analysis",
            "component_id": component_id,
            "affected_components": affected_components
        }
        
        # Visualize the impact
        os.makedirs("static/images", exist_ok=True)
        vis_file = f"static/images/impact_{component_id}.png"
        self.db.visualize_subgraph(subgraph, vis_file)
        result["visualization"] = vis_file
        
        return result
    
    def _handle_property_query(self, parsed_query):
        """Handle queries about specific properties of components"""
        component_ids = parsed_query["component_ids"]
        properties = parsed_query["properties"]
        
        if not component_ids:
            return {
                "answer_type": "error",
                "message": "No component ID specified for property query"
            }
        
        if not properties:
            return {
                "answer_type": "error",
                "message": "No property specified for query"
            }
        
        # Get the first component ID and property
        component_id = component_ids[0]
        property_name = properties[0]
        
        # Get the property value
        property_value = self.db.get_component_property(component_id, property_name)
        
        # Format the results
        result = {
            "answer_type": "property",
            "component_id": component_id,
            "property": property_name,
            "value": property_value
        }
        
        return result
    
    def _handle_component_info_query(self, parsed_query):
        """Handle general queries about component information"""
        component_ids = parsed_query["component_ids"]
        
        if not component_ids:
            return {
                "answer_type": "error",
                "message": "No component ID specified for information query"
            }
        
        # Get the first component ID
        component_id = component_ids[0]
        
        # Get the subgraph centered on this component
        subgraph = self.db.get_subgraph(component_id, max_hops=1)
        
        # Get component data
        component_data = None
        if component_id in subgraph.nodes:
            component_data = dict(subgraph.nodes[component_id])
        
        if not component_data:
            return {
                "answer_type": "text",
                "message": f"Component {component_id} not found"
            }
        
        # Get connected components
        connected_components = []
        for node in subgraph.neighbors(component_id):
            node_data = dict(subgraph.nodes[node])
            edge_data = subgraph.get_edge_data(component_id, node) or subgraph.get_edge_data(node, component_id)
            
            if edge_data:
                relationship = edge_data.get("type", "CONNECTED_TO")
                connected_components.append({
                    "component": node_data,
                    "relationship": relationship
                })
        
        # Format the results
        result = {
            "answer_type": "component_info",
            "component": component_data,
            "connected_components": connected_components
        }
        
        # Visualize the component and its connections
        os.makedirs("static/images", exist_ok=True)
        vis_file = f"static/images/info_{component_id}.png"
        self.db.visualize_subgraph(subgraph, vis_file)
        result["visualization"] = vis_file
        
        return result
    
    def format_answer(self, reasoning_result):
        """Format the reasoning result as a human-readable answer"""
        answer_type = reasoning_result.get("answer_type")
        
        if answer_type == "error":
            return reasoning_result.get("message", "An error occurred")
        
        if answer_type == "text":
            return reasoning_result.get("message", "")
        
        if answer_type == "path":
            start_id = reasoning_result.get("start_id")
            end_id = reasoning_result.get("end_id")
            paths = reasoning_result.get("paths", [])
            
            if not paths:
                return f"No path found between {start_id} and {end_id}"
            
            # Format the shortest path
            shortest_path = paths[0]
            nodes = shortest_path.get("nodes", [])
            
            answer = f"Path from {start_id} to {end_id} ({len(nodes)-1} steps):\n\n"
            
            for i, node in enumerate(nodes):
                node_id = node.get("id", "Unknown")
                labels = node.get("labels", ["Unknown"])
                node_type = labels[0] if labels else "Unknown"
                
                answer += f"{i+1}. {node_type} {node_id}"
                
                if i < len(nodes) - 1:
                    relationship = shortest_path["relationships"][i].get("type", "connected to")
                    answer += f" {relationship} →\n"
                else:
                    answer += "\n"
            
            return answer
        
        if answer_type == "common_connections":
            component_ids = reasoning_result.get("component_ids", [])
            common_connections = reasoning_result.get("common_connections", [])
            
            if not common_connections:
                return f"No common connections found for {', '.join(component_ids)}"
            
            components_str = ", ".join(component_ids)
            answer = f"Common connections for {components_str}:\n\n"
            
            for i, component in enumerate(common_connections[:5]):  # Limit to top 5
                component_id = component.get("id", "Unknown")
                labels = component.get("labels", ["Unknown"])
                component_type = labels[0] if labels else "Unknown"
                
                answer += f"{i+1}. {component_type} {component_id}\n"
            
            if len(common_connections) > 5:
                answer += f"\n... and {len(common_connections) - 5} more"
            
            return answer
        
        if answer_type == "comparison":
            property_name = reasoning_result.get("property", "")
            comparisons = reasoning_result.get("comparisons", [])
            
            if not comparisons:
                return f"Could not compare {property_name}"
            
            answer = f"Comparison of {property_name}:\n\n"
            
            for comp in comparisons:
                component_id = comp.get("component_id", "Unknown")
                component_type = comp.get("component_type", "Unknown")
                value = comp.get("property_value", "Not available")
                
                answer += f"- {component_type} {component_id}: {value}\n"
            
            return answer
        
        if answer_type == "flow_direction":
            component_id = reasoning_result.get("component_id", "")
            direction = reasoning_result.get("direction", "")
            components = reasoning_result.get("components", [])
            
            if not components:
                return f"No {direction} components found for {component_id}"
            
            answer = f"{direction.capitalize()} components for {component_id}:\n\n"
            
            for i, component in enumerate(components):
                component_id = component.get("id", "Unknown")
                labels = component.get("labels", [])
                component_type = labels[0] if labels else "Unknown"
                
                answer += f"{i+1}. {component_type} {component_id}\n"
            
            return answer
        
        if answer_type == "impact_analysis":
            component_id = reasoning_result.get("component_id", "")
            affected = reasoning_result.get("affected_components", [])
            
            if not affected:
                return f"No components would be affected if {component_id} fails"
            
            answer = f"If {component_id} fails, the following components would be affected:\n\n"
            
            for i, component in enumerate(affected[:10]):  # Limit to top 10
                component_id = component.get("id", "Unknown")
                labels = component.get("labels", [])
                component_type = labels[0] if labels else "Unknown"
                
                answer += f"{i+1}. {component_type} {component_id}\n"
            
            if len(affected) > 10:
                answer += f"\n... and {len(affected) - 10} more components"
            
            return answer
        
        if answer_type == "property":
            component_id = reasoning_result.get("component_id", "")
            property_name = reasoning_result.get("property", "")
            value = reasoning_result.get("value", "Not available")
            
            return f"The {property_name} of {component_id} is {value}"
        
        if answer_type == "component_info":
            component = reasoning_result.get("component", {})
            connected = reasoning_result.get("connected_components", [])
            
            if not component:
                return "Component information not found"
            
            component_id = component.get("id", "Unknown")
            labels = component.get("labels", [])
            component_type = labels[0] if labels else "Unknown"
            
            answer = f"Information about {component_type} {component_id}:\n\n"
            
            # Add properties
            answer += "Properties:\n"
            for key, value in component.items():
                if key not in ["id", "labels"]:
                    answer += f"- {key}: {value}\n"
            
            # Add connections
            if connected:
                answer += "\nConnections:\n"
                for conn in connected:
                    conn_component = conn.get("component", {})
                    conn_id = conn_component.get("id", "Unknown")
                    conn_labels = conn_component.get("labels", [])
                    conn_type = conn_labels[0] if conn_labels else "Unknown"
                    relationship = conn.get("relationship", "Connected to")
                    
                    answer += f"- {relationship} {conn_type} {conn_id}\n"
            
            return answer
        
        # Default response
        return "I couldn't determine how to answer that question. Please try rephrasing."
```

### Step 5: Implement Web Server
1. Create a Flask server in `server.py`:

```python
import os
from flask import Flask, request, jsonify, render_template
from .database import Neo4jConnector
from .reasoning import GraphReasoner

app = Flask(__name__, static_folder='static')

# Create static directories if they don't exist
os.makedirs('static/images', exist_ok=True)

# Initialize Neo4j connection
neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
reasoner = GraphReasoner(neo4j)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a natural language query"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Process the query with the graph reasoner
    reasoning_result = reasoner.answer_query(query)
    
    # Format the answer
    answer = reasoner.format_answer(reasoning_result)
    
    # Include visualization if available
    visualization = reasoning_result.get("visualization")
    
    return jsonify({
        "query": query,
        "answer": answer,
        "reasoning_result": reasoning_result,
        "visualization": visualization
    })

# HTML template for the main page
@app.route('/create_templates')
def create_templates():
    """Create the HTML templates for the application"""
    os.makedirs('graph_reasoner/templates', exist_ok=True)
    with open('graph_reasoner/templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Hop Graph Reasoning Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .query-container {
            margin-bottom: 20px;
            display: flex;
        }
        .query-input {
            flex-grow: 1;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .query-button {
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            margin-left: 10px;
            cursor: pointer;
        }
        .query-button:hover {
            background-color: #2980b9;
        }
        .examples {
            margin-bottom: 20px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
        }
        .examples h3 {
            margin-top: 0;
        }
        .examples ul {
            padding-left: 20px;
        }
        .examples li {
            margin-bottom: 5px;
        }
        .results-container {
            margin-top: 30px;
            display: flex;
            flex-direction: column;
        }
        .answer {
            background-color: #f0f4f8;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            white-space: pre-line;
        }
        .visualization {
            text-align: center;
            margin-top: 20px;
        }
        .visualization img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        #loading {
            text-align: center;
            display: none;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-radius: 50%;
            border-top: 4px solid #3498db;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1>Multi-Hop Graph Reasoning Tool</h1>
    
    <div class="examples">
        <h3>Example Queries</h3>
        <ul>
            <li>What is the path from VLV001 to TNK001?</li>
            <li>What components are common to both PIP001 and PIP002?</li>
            <li>Compare the diameter of VLV001 and VLV002</li>
            <li>What are the downstream components of JCT001?</li>
            <li>What would be affected if PMP001 fails?</li>
            <li>What is the status of VLV001?</li>
            <li>Show me information about TNK001</li>
        </ul>
    </div>
    
    <div class="query-container">
        <input type="text" id="query-input" class="query-input" placeholder="Enter your question about the water network...">
        <button id="query-button" class="query-button">Ask</button>
    </div>
    
    <div id="loading">
        <div class="spinner"></div>
        <span>Reasoning about your query...</span>
    </div>
    
    <div class="results-container" id="results">
        <div id="answer" class="answer" style="display: none;"></div>
        <div id="visualization" class="visualization" style="display: none;">
            <h3>Visualization</h3>
            <img id="vis-image" src="">
        </div>
    </div>

    <script>
        document.getElementById('query-button').addEventListener('click', processQuery);
        document.getElementById('query-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                processQuery();
            }
        });
        
        function processQuery() {
            const query = document.getElementById('query-input').value;
            if (!query) return;
            
            // Show loading spinner
            document.getElementById('loading').style.display = 'block';
            
            // Hide previous results
            document.getElementById('answer').style.display = 'none';
            document.getElementById('visualization').style.display = 'none';
            
            // Send query to API
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                // Display answer
                const answerElement = document.getElementById('answer');
                answerElement.textContent = data.answer;
                answerElement.style.display = 'block';
                
                // Display visualization if available
                if (data.visualization) {
                    document.getElementById('vis-image').src = '/' + data.visualization;
                    document.getElementById('visualization').style.display = 'block';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('answer').textContent = 'An error occurred while processing your query.';
                document.getElementById('answer').style.display = 'block';
            });
        }
        
        // Pre-fill with example queries on click
        document.querySelectorAll('.examples li').forEach(item => {
            item.style.cursor = 'pointer';
            item.addEventListener('click', function() {
                document.getElementById('query-input').value = this.textContent;
            });
        });
    </script>
</body>
</html>
        ''')
    return "Templates created"

def create_app():
    """Create and configure the Flask app"""
    create_templates()
    return app

if __name__ == '__main__':
    create_templates()
    app = create_app()
    app.run(debug=True)
```

### Step 6: Create Tests for Reasoning
1. Implement tests in `tests/test_reasoning.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from graph_reasoner.reasoning import GraphReasoner
from graph_reasoner.query_parser import QueryParser

@pytest.fixture
def mock_neo4j():
    """Create a mock Neo4j connector"""
    neo4j = MagicMock()
    
    # Create a sample graph
    G = nx.DiGraph()
    
    # Add sample nodes
    G.add_node("VLV001", id="VLV001", labels=["Valve"], valveType="Gate", status="Open")
    G.add_node("PIP001", id="PIP001", labels=["Pipe"], material="PVC", diameter=200)
    G.add_node("JCT001", id="JCT001", labels=["Junction"], elevation=100)
    G.add_node("TNK001", id="TNK001", labels=["Tank"], capacity=5000)
    
    # Add sample edges
    G.add_edge("VLV001", "PIP001", type="CONNECTED_TO")
    G.add_edge("PIP001", "JCT001", type="CONNECTED_TO")
    G.add_edge("JCT001", "TNK001", type="CONNECTED_TO")
    
    # Setup mock returns
    neo4j.get_subgraph.return_value = G
    
    # Setup mock path finding
    mock_path = {
        "nodes": [
            {"id": "VLV001", "labels": ["Valve"]},
            {"id": "PIP001", "labels": ["Pipe"]},
            {"id": "JCT001", "labels": ["Junction"]},
            {"id": "TNK001", "labels": ["Tank"]}
        ],
        "relationships": [
            {"type": "CONNECTED_TO"},
            {"type": "CONNECTED_TO"},
            {"type": "CONNECTED_TO"}
        ],
        "length": 3
    }
    neo4j.find_paths.return_value = [mock_path]
    
    # Setup mock property query
    neo4j.get_component_property.return_value = "Open"
    
    # Setup mock comparison
    neo4j.compare_components.return_value = [
        {"component_id": "VLV001", "component_type": "Valve", "property_value": 200},
        {"component_id": "VLV002", "component_type": "Valve", "property_value": 150}
    ]
    
    # Setup mock common connections
    neo4j.find_common_connections.return_value = [
        {"id": "JCT001", "labels": ["Junction"]}
    ]
    
    return neo4j

@pytest.fixture
def reasoner(mock_neo4j):
    """Create a graph reasoner with mock dependencies"""
    return GraphReasoner(mock_neo4j)

def test_path_query(reasoner, mock_neo4j):
    """Test handling a path query"""
    result = reasoner.answer_query("What is the path from VLV001 to TNK001?")
    
    # Verify the query was processed correctly
    mock_neo4j.find_paths.assert_called_once()
    
    # Check the result
    assert result["answer_type"] == "path"
    assert result["start_id"] == "VLV001"
    assert result["end_id"] == "TNK001"
    assert len(result["paths"]) == 1
    assert len(result["paths"][0]["nodes"]) == 4

def test_common_connections_query(reasoner, mock_neo4j):
    """Test handling a common connections query"""
    result = reasoner.answer_query("What components are common to both PIP001 and PIP002?")
    
    # Verify the query was processed correctly
    mock_neo4j.find_common_connections.assert_called_once()
    
    # Check the result
    assert result["answer_type"] == "common_connections"
    assert "PIP001" in result["component_ids"]
    assert "PIP002" in result["component_ids"]
    assert len(result["common_connections"]) == 1
    assert result["common_connections"][0]["id"] == "JCT001"

def test_comparison_query(reasoner, mock_neo4j):
    """Test handling a comparison query"""
    result = reasoner.answer_query("Compare the diameter of VLV001 and VLV002")
    
    # Verify the query was processed correctly
    mock_neo4j.compare_components.assert_called_once()
    
    # Check the result
    assert result["answer_type"] == "comparison"
    assert result["property"] == "diameter"
    assert len(result["comparisons"]) == 2
    assert result["comparisons"][0]["component_id"] == "VLV001"
    assert result["comparisons"][1]["component_id"] == "VLV002"

def test_flow_direction_query(reasoner, mock_neo4j):
    """Test handling a flow direction query"""
    result = reasoner.answer_query("What are the downstream components of JCT001?")
    
    # Verify the query was processed correctly
    mock_neo4j.get_subgraph.assert_called_once()
    
    # Check the result
    assert result["answer_type"] == "flow_direction"
    assert result["component_id"] == "JCT001"
    assert result["direction"] == "downstream"

def test_property_query(reasoner, mock_neo4j):
    """Test handling a property query"""
    result = reasoner.answer_query("What is the status of VLV001?")
    
    # Verify the query was processed correctly
    mock_neo4j.get_component_property.assert_called_once()
    
    # Check the result
    assert result["answer_type"] == "property"
    assert result["component_id"] == "VLV001"
    assert result["property"] == "status"
    assert result["value"] == "Open"

def test_format_answer(reasoner):
    """Test formatting reasoning results as human-readable answers"""
    # Test path formatting
    path_result = {
        "answer_type": "path",
        "start_id": "VLV001",
        "end_id": "TNK001",
        "paths": [{
            "nodes": [
                {"id": "VLV001", "labels": ["Valve"]},
                {"id": "PIP001", "labels": ["Pipe"]},
                {"id": "TNK001", "labels": ["Tank"]}
            ],
            "relationships": [
                {"type": "CONNECTED_TO"},
                {"type": "CONNECTED_TO"}
            ]
        }]
    }
    
    answer = reasoner.format_answer(path_result)
    assert "Path from VLV001 to TNK001" in answer
    assert "Valve VLV001" in answer
    assert "Pipe PIP001" in answer
    assert "Tank TNK001" in answer
    
    # Test property formatting
    property_result = {
        "answer_type": "property",
        "component_id": "VLV001",
        "property": "status",
        "value": "Open"
    }
    
    answer = reasoner.format_answer(property_result)
    assert "The status of VLV001 is Open" in answer
```

### Step 7: Create Setup Script
1. Create a setup script for easy installation:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="graph-reasoner",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "neo4j",
        "networkx",
        "spacy",
        "matplotlib",
        "flask",
    ],
    entry_points={
        "console_scripts": [
            "graph-reasoner=graph_reasoner.server:create_app",
        ],
    },
)
```

### Step 8: Run and Test
1. Install the package in development mode:
```bash
pip install -e .
```

2. Create templates and start the Flask application:
```bash
python -m graph_reasoner.server
```

3. Open a web browser and navigate to `http://localhost:5000`

4. Try asking questions like:
   - "What is the path from VLV001 to TNK001?"
   - "What components are common to both PIP001 and PIP002?"
   - "Compare the diameter of VLV001 and VLV002"
   - "What are the downstream components of JCT001?"
   - "What would be affected if PMP001 fails?"
   - "What is the status of VLV001?"
   - "Show me information about TNK001"

### Deliverables
1. A functional multi-hop graph reasoning tool
2. A web interface for natural language queries
3. Visualization of graph reasoning results
4. Support for different query types:
   - Path finding
   - Common connections
   - Property comparison
   - Flow direction analysis
   - Impact analysis
   - Property queries
   - Component information

## Extensions
1. Implement more sophisticated query parsing using a pre-trained language model
2. Add support for more complex reasoning patterns
3. Enhance visualizations with interactive graph elements
4. Implement a conversation history for context-aware reasoning
5. Add explanations of the reasoning process
6. Create a feature for saving and sharing reasoning results
7. Add support for querying multiple components at once

## Relation to Main Project
This mini-project builds directly on the GraphRAG concepts from Phase 2, focusing specifically on multi-hop reasoning capabilities. The tool you develop will demonstrate how to combine graph traversal with semantic understanding to answer complex questions about your water network. These capabilities will be essential for implementing the advanced retrieval system in the main project, particularly for cases where information needs to be gathered from multiple connected components in the graph.
