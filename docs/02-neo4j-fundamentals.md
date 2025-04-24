# Neo4j Fundamentals: Graph Database Concepts for Water Networks

This guide will introduce you to Neo4j and graph database concepts, with a specific focus on modeling water network infrastructure. By the end of this guide, you'll have a solid understanding of how graph databases work and how to begin modeling a water network in Neo4j.

## 1. Understanding Graph Databases

### What is a Graph Database?

A graph database is designed to treat the relationships between data as equally important as the data itself. Unlike traditional relational databases that store data in tables, graph databases use nodes (entities) and relationships (connections between entities) to represent and store data.

#### Key Components:

- **Nodes**: Represent entities (like valves, pumps, junctions in a water network)
- **Relationships**: Connect nodes and define how entities relate to each other
- **Properties**: Store attributes on both nodes and relationships
- **Labels**: Group similar nodes together (like "Valve", "Pump", etc.)

### Why Graph Databases for Water Networks?

Water distribution networks are inherently graph-like in structure:

- **Network Topology**: Pipes connect different components, creating a natural graph
- **Multi-dimensional Relationships**: Components have various types of connections (physical, operational, maintenance)
- **Path-based Analysis**: Finding efficient routes, isolating sections, and tracing flows are all graph traversal problems
- **Complex Queries**: Questions like "What areas are affected if this valve fails?" are easily expressed as graph queries

## 2. Neo4j Basics

### Installation and Setup

#### Option 1: Neo4j Desktop (Recommended for Beginners)
1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Install and launch the application
3. Create a new project for your water network
4. Add a local database (use version 5.x or later)
5. Start your database

#### Option 2: Neo4j Aura (Cloud-based)
1. Sign up for [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. Connect using the provided credentials

#### Option 3: Docker
```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### Connecting to Your Database

Once your database is running, you can connect to it:

- **Neo4j Browser**: Web interface at http://localhost:7474 (for local installations)
- **Neo4j Bloom**: Visual exploration tool included with Neo4j Desktop
- **Drivers**: Connect programmatically using drivers for Python, JavaScript, Java, etc.

### Your First Cypher Queries

Cypher is Neo4j's query language. Let's start with some basic queries:

```cypher
// Create a node
CREATE (v:Valve {id: 'V001', type: 'Gate', installDate: '2010-05-15'})

// Create multiple nodes
CREATE (j:Junction {id: 'J001', elevation: 100})
CREATE (p:Pipe {id: 'P001', diameter: 200, material: 'PVC', length: 150})

// Create a relationship
MATCH (v:Valve {id: 'V001'}), (j:Junction {id: 'J001'})
CREATE (v)-[:CONNECTED_TO]->(j)

// Create a relationship with properties
MATCH (j:Junction {id: 'J001'}), (p:Pipe {id: 'P001'})
CREATE (j)-[:CONNECTED_TO {flow_direction: 'outbound'}]->(p)

// Query the graph
MATCH (v:Valve)-[r:CONNECTED_TO]->(c)
RETURN v, r, c

// Find all components connected to a specific valve
MATCH (v:Valve {id: 'V001'})-[*1..2]-(c)
RETURN v, c
```

## 3. Modeling a Water Network in Neo4j

### Node Types (Labels)

For a water network, consider these node labels:

- **:Valve** - Control points in the network
- **:Pump** - Equipment that adds energy to the water
- **:Junction** - Connection points between pipes
- **:Reservoir** - Water sources
- **:Tank** - Storage facilities
- **:Meter** - Flow or pressure measurement points
- **:Sensor** - Monitoring equipment
- **:Zone** - Pressure or distribution zones
- **:Customer** - Service connections

### Relationship Types

Relationships define how components connect:

- **:CONNECTED_TO** - Physical connection between components
- **:FEEDS** - Direction of flow
- **:CONTROLS** - Valve or pump controlling a zone
- **:MONITORS** - Sensor monitoring a component
- **:LOCATED_IN** - Geographic relationship
- **:PART_OF** - Organizational relationship
- **:MAINTAINED_BY** - Maintenance responsibility

### Properties

Properties store attributes:

#### Node Properties
- **id**: Unique identifier
- **type**: Component subtype (e.g., gate valve, butterfly valve)
- **installDate**: When the component was installed
- **material**: What it's made of
- **manufacturer**: Who made it
- **location**: Geographic coordinates
- **status**: Operational status
- **lastMaintenance**: Date of last maintenance

#### Relationship Properties
- **distance**: Length of connection
- **diameter**: Pipe diameter
- **installDate**: When the connection was created
- **flowCapacity**: Maximum flow rate
- **material**: Pipe material

### Sample Schema

Here's a basic schema for a water network in Cypher:

```cypher
// Create constraints for unique IDs
CREATE CONSTRAINT FOR (v:Valve) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Pump) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (j:Junction) REQUIRE j.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Pipe) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (r:Reservoir) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:Tank) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT FOR (z:Zone) REQUIRE z.id IS UNIQUE;

// Create indexes for common lookups
CREATE INDEX FOR (v:Valve) ON (v.status);
CREATE INDEX FOR (p:Pipe) ON (p.material);
CREATE INDEX FOR (c:Component) ON (c.installDate);
```

## 4. Practical Exercise: Creating a Mini Water Network

Let's build a small water network to practice:

```cypher
// Clear the database first (be careful with this in production!)
MATCH (n) DETACH DELETE n;

// Create reservoirs
CREATE (r1:Reservoir {id: 'RES001', name: 'Highland Reservoir', capacity: 5000000, elevation: 250})
CREATE (r2:Reservoir {id: 'RES002', name: 'Valley Reservoir', capacity: 3000000, elevation: 200})

// Create pumps
CREATE (p1:Pump {id: 'PMP001', type: 'Centrifugal', power: 75, status: 'Active', installDate: '2015-03-10'})
CREATE (p2:Pump {id: 'PMP002', type: 'Centrifugal', power: 50, status: 'Active', installDate: '2018-07-22'})

// Create valves
CREATE (v1:Valve {id: 'VLV001', type: 'Gate', diameter: 300, status: 'Open', installDate: '2015-03-15'})
CREATE (v2:Valve {id: 'VLV002', type: 'Butterfly', diameter: 200, status: 'Open', installDate: '2015-04-20'})
CREATE (v3:Valve {id: 'VLV003', type: 'Gate', diameter: 200, status: 'Open', installDate: '2018-08-05'})

// Create junctions
CREATE (j1:Junction {id: 'JCT001', elevation: 245})
CREATE (j2:Junction {id: 'JCT002', elevation: 220})
CREATE (j3:Junction {id: 'JCT003', elevation: 195})
CREATE (j4:Junction {id: 'JCT004', elevation: 195})

// Create pipes
CREATE (pp1:Pipe {id: 'PIP001', length: 500, diameter: 300, material: 'Ductile Iron', installDate: '2015-03-12'})
CREATE (pp2:Pipe {id: 'PIP002', length: 1200, diameter: 300, material: 'Ductile Iron', installDate: '2015-03-18'})
CREATE (pp3:Pipe {id: 'PIP003', length: 800, diameter: 200, material: 'PVC', installDate: '2015-04-25'})
CREATE (pp4:Pipe {id: 'PIP004', length: 1500, diameter: 200, material: 'PVC', installDate: '2018-08-10'})

// Create tanks
CREATE (t1:Tank {id: 'TNK001', capacity: 500000, level: 0.8, elevation: 240, installDate: '2015-03-20'})

// Create zones
CREATE (z1:Zone {id: 'ZON001', name: 'High Pressure Zone', pressure: 80})
CREATE (z2:Zone {id: 'ZON002', name: 'Low Pressure Zone', pressure: 60})

// Connect components - Reservoir to pumps
MATCH (r1:Reservoir {id: 'RES001'}), (p1:Pump {id: 'PMP001'})
CREATE (r1)-[:FEEDS]->(p1)

MATCH (r2:Reservoir {id: 'RES002'}), (p2:Pump {id: 'PMP002'})
CREATE (r2)-[:FEEDS]->(p2)

// Connect pumps to valves
MATCH (p1:Pump {id: 'PMP001'}), (v1:Valve {id: 'VLV001'})
CREATE (p1)-[:FEEDS]->(v1)

MATCH (p2:Pump {id: 'PMP002'}), (v3:Valve {id: 'VLV003'})
CREATE (p2)-[:FEEDS]->(v3)

// Connect valves to junctions
MATCH (v1:Valve {id: 'VLV001'}), (j1:Junction {id: 'JCT001'})
CREATE (v1)-[:FEEDS]->(j1)

MATCH (v2:Valve {id: 'VLV002'}), (j3:Junction {id: 'JCT003'})
CREATE (v2)-[:FEEDS]->(j3)

MATCH (v3:Valve {id: 'VLV003'}), (j4:Junction {id: 'JCT004'})
CREATE (v3)-[:FEEDS]->(j4)

// Connect junctions to pipes
MATCH (j1:Junction {id: 'JCT001'}), (pp1:Pipe {id: 'PIP001'})
CREATE (j1)-[:CONNECTED_TO]->(pp1)

MATCH (pp1:Pipe {id: 'PIP001'}), (j2:Junction {id: 'JCT002'})
CREATE (pp1)-[:CONNECTED_TO]->(j2)

MATCH (j2:Junction {id: 'JCT002'}), (pp2:Pipe {id: 'PIP002'})
CREATE (j2)-[:CONNECTED_TO]->(pp2)

MATCH (pp2:Pipe {id: 'PIP002'}), (v2:Valve {id: 'VLV002'})
CREATE (pp2)-[:CONNECTED_TO]->(v2)

MATCH (j3:Junction {id: 'JCT003'}), (pp3:Pipe {id: 'PIP003'})
CREATE (j3)-[:CONNECTED_TO]->(pp3)

MATCH (j4:Junction {id: 'JCT004'}), (pp4:Pipe {id: 'PIP004'})
CREATE (j4)-[:CONNECTED_TO]->(pp4)

// Connect pipes to tanks
MATCH (pp3:Pipe {id: 'PIP003'}), (t1:Tank {id: 'TNK001'})
CREATE (pp3)-[:CONNECTED_TO]->(t1)

// Set zone relationships
MATCH (j1:Junction {id: 'JCT001'}), (z1:Zone {id: 'ZON001'})
CREATE (j1)-[:PART_OF]->(z1)

MATCH (j2:Junction {id: 'JCT002'}), (z1:Zone {id: 'ZON001'})
CREATE (j2)-[:PART_OF]->(z1)

MATCH (j3:Junction {id: 'JCT003'}), (z2:Zone {id: 'ZON002'})
CREATE (j3)-[:PART_OF]->(z2)

MATCH (j4:Junction {id: 'JCT004'}), (z2:Zone {id: 'ZON002'})
CREATE (j4)-[:PART_OF]->(z2)

MATCH (v2:Valve {id: 'VLV002'}), (z2:Zone {id: 'ZON002'})
CREATE (v2)-[:CONTROLS]->(z2)
```

## 5. Common Water Network Queries

Now that we've created a small water network, let's explore some useful queries:

### Find All Components in a Zone

```cypher
MATCH (c)-[:PART_OF]->(z:Zone {id: 'ZON001'})
RETURN c
```

### Trace Water Flow Path

```cypher
MATCH path = (r:Reservoir {id: 'RES001'})-[:FEEDS|CONNECTED_TO*]->(t:Tank)
RETURN path
```

### Find Valves that Need Maintenance (Older than 5 Years)

```cypher
MATCH (v:Valve)
WHERE date(v.installDate) < date() - duration('P5Y')
RETURN v
```

### Find Critical Components (Those that, if Removed, Disconnect the Network)

```cypher
MATCH (n1)-[r]-(n2)
WITH r, count(*) AS connections
WHERE connections = 1
RETURN startNode(r) AS component
```

### Find All Components Within 2 Hops of a Specific Valve

```cypher
MATCH (v:Valve {id: 'VLV001'})-[*1..2]-(c)
RETURN v, c
```

## 6. Advanced Modeling Techniques

### Temporal Data

Water networks change over time. Here's how to model temporal aspects:

```cypher
// Record maintenance events
MATCH (v:Valve {id: 'VLV001'})
CREATE (m:Maintenance {id: 'MNT001', date: '2022-05-10', type: 'Inspection', notes: 'No issues found'})
CREATE (v)-[:HAS_MAINTENANCE]->(m)
CREATE (t:Technician {id: 'TEC001', name: 'John Smith'})
CREATE (m)-[:PERFORMED_BY]->(t)

// Record pressure readings over time
MATCH (j:Junction {id: 'JCT001'})
CREATE (r1:PressureReading {timestamp: datetime('2023-01-01T12:00:00'), value: 78})
CREATE (r2:PressureReading {timestamp: datetime('2023-01-01T13:00:00'), value: 77})
CREATE (r3:PressureReading {timestamp: datetime('2023-01-01T14:00:00'), value: 79})
CREATE (j)-[:HAS_READING]->(r1)
CREATE (j)-[:HAS_READING]->(r2)
CREATE (j)-[:HAS_READING]->(r3)
```

### Spatial Data

Neo4j has built-in spatial capabilities:

```cypher
// Add spatial points to components
MATCH (v:Valve {id: 'VLV001'})
SET v.location = point({x: -74.0060, y: 40.7128, crs: 'WGS-84'})

// Find components within a certain distance
MATCH (c)
WHERE exists(c.location) AND 
      distance(c.location, point({x: -74.0060, y: 40.7128, crs: 'WGS-84'})) < 1000
RETURN c
```

### Modeling Complex Events

Water networks experience events like breaks, leaks, or service disruptions:

```cypher
// Record a pipe break event
MATCH (p:Pipe {id: 'PIP002'})
CREATE (e:Event {id: 'EVT001', type: 'Break', timestamp: datetime('2023-02-15T08:30:00'), 
                 severity: 'High', description: 'Major leak requiring immediate repair'})
CREATE (p)-[:EXPERIENCED]->(e)

// Record affected components
MATCH (e:Event {id: 'EVT001'}), (z:Zone {id: 'ZON002'})
CREATE (e)-[:AFFECTED]->(z)

// Record response actions
MATCH (e:Event {id: 'EVT001'})
CREATE (a:Action {id: 'ACT001', type: 'Repair', timestamp: datetime('2023-02-15T10:45:00'),
                 description: 'Emergency pipe replacement'})
CREATE (e)-[:HAS_RESPONSE]->(a)
```

## 7. Data Import Strategies

Real water networks have thousands of components. Here are strategies for importing data:

### CSV Import

Create CSV files for each component type and use LOAD CSV:

```cypher
// Load valves from CSV
LOAD CSV WITH HEADERS FROM 'file:///valves.csv' AS row
CREATE (v:Valve {id: row.id, type: row.type, diameter: toInteger(row.diameter),
                 status: row.status, installDate: row.installDate})

// Load pipes from CSV
LOAD CSV WITH HEADERS FROM 'file:///pipes.csv' AS row
CREATE (p:Pipe {id: row.id, length: toFloat(row.length), diameter: toInteger(row.diameter),
                material: row.material, installDate: row.installDate})

// Create relationships based on connection CSV
LOAD CSV WITH HEADERS FROM 'file:///connections.csv' AS row
MATCH (a {id: row.from_id}), (b {id: row.to_id})
CREATE (a)-[:CONNECTED_TO]->(b)
```

### API Import

For larger datasets, consider using the Neo4j API with the appropriate driver:

#### Python Example
```python
from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

def import_component(tx, component_id, component_type, properties):
    cypher = f"CREATE (c:{component_type} {{id: $id"
    for key in properties:
        cypher += f", {key}: ${key}"
    cypher += "})"
    
    params = {"id": component_id}
    params.update(properties)
    tx.run(cypher, **params)

with driver.session() as session:
    session.write_transaction(import_component, "VLV099", "Valve", 
                              {"type": "Gate", "diameter": 250, "status": "Open", 
                               "installDate": "2020-05-15"})

driver.close()
```

## 8. Next Steps

Now that you have a basic understanding of Neo4j and how to model water networks, here are your next steps:

1. **Practice with Real Data**: Try to import a subset of your actual water network data
2. **Explore Visualization**: Use Neo4j Bloom to create meaningful visualizations
3. **Learn Advanced Cypher**: Study more complex query patterns
4. **Connect to Applications**: Build simple applications that query the database
5. **Study Graph Algorithms**: Learn how to use Neo4j's graph algorithms for analytics

## 9. Resources

### Neo4j Documentation
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Developer Guide](https://neo4j.com/developer/get-started/)
- [APOC Library](https://neo4j.com/docs/apoc/current/) - Useful procedures and functions

### Water Network Resources
- [EPANET](https://www.epa.gov/water-research/epanet) - Water network analysis software
- [International Water Association](https://iwa-network.org/) - Resources on water networks

### Graph Modeling Resources
- [Graph Data Modeling for Neo4j](https://neo4j.com/graph-data-modeling-book/)
- [Neo4j Graph Academy](https://graphacademy.neo4j.com/)

This guide has provided you with a solid foundation for working with Neo4j and modeling water networks as graphs. In the next guide, we'll explore how to enhance this model with GraphRAG for intelligent information retrieval.
