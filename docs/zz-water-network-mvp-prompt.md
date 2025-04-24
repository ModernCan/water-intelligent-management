# Water Network Intelligence System MVP - Developer Prompt

## Project Overview
Create a web-based Minimum Viable Product (MVP) for a Water Network Intelligence System that enables water utility operators to visualize their network, access component information, and query the system using natural language. This system should provide insights about water infrastructure components such as valves, pipes, pumps, tanks, and junctions.

## Core Features for MVP

### 1. Graph Database Integration
- Connect to a Neo4j graph database containing water network components
- Components include: Valves, Pipes, Pumps, Tanks, and Junctions
- Components have relationships like: CONNECTED_TO, PART_OF, FEEDS

### 2. Network Visualization
- Display a simple network map showing components and their connections
- Allow clicking on components to view their details
- Use different colors/shapes to represent different component types

### 3. Component Details Panel
- Show detailed information about selected components
- Display properties such as status, diameter, material, installation date
- Show maintenance history if available

### 4. Natural Language Query Interface
- Implement a chat-like interface for querying the network
- Support basic queries like:
  - "What is the status of valve VLV001?"
  - "Show me all pipes connected to junction JCT001"
  - "Which components were installed before 2010?"
- Provide responses that include relevant component information

### 5. Status Dashboard
- Show overall network health metrics
- Display alerts or warnings for problematic components
- Provide a summary of component counts by type and status

## Technical Requirements

### Frontend
- Use React for the user interface
- Implement a responsive design that works on desktop browsers
- Include these main components:
  - Dashboard with status overview
  - Network visualization
  - Component details panel
  - Query interface with chat history

### Backend
- Create a REST API to serve data from Neo4j to the frontend
- Implement these key endpoints:
  - GET /api/components - List all components with filtering options
  - GET /api/components/{id} - Get details for a specific component
  - GET /api/status - Get overall network status
  - POST /api/query - Process natural language queries

### Neo4j Integration
- Use the Neo4j JavaScript driver for database connectivity
- Implement Cypher queries to:
  - Retrieve components and their connections
  - Find paths between components
  - Query component properties
  - Analyze maintenance history

## Data Model
For this MVP, work with this simplified data model:

### Nodes (Components)
- **Valve**: properties include id, type, diameter, status, installDate
- **Pipe**: properties include id, material, diameter, length, installDate
- **Pump**: properties include id, type, capacity, status, power
- **Tank**: properties include id, capacity, level, elevation
- **Junction**: properties include id, elevation, connections

### Relationships
- **CONNECTED_TO**: physical connection between components
- **PART_OF**: component belongs to a zone or area
- **FEEDS**: directional flow from one component to another

## Implementation Approach

### Step 1: Setup Project Structure
- Create a React frontend application with Material UI components
- Set up a Node.js backend with Express
- Configure Neo4j connection and basic API endpoints

### Step 2: Create Core Components
- Implement the Dashboard container component
- Create the network visualization using a library like react-graph-vis
- Build the component details panel with dynamic property display
- Develop the natural language query interface

### Step 3: Connect to Data
- Implement Neo4j data retrieval in the backend
- Create API endpoints to serve component data
- Implement basic NLP parsing for the query interface
- Connect frontend components to API endpoints

### Step 4: Enhance User Experience
- Add loading states and error handling
- Implement a responsive layout
- Add basic styling and visual improvements
- Ensure consistent error messaging

## Sample Queries to Support
The MVP should support these types of natural language queries:

1. "What is the status of valve VLV001?"
2. "Show me all components connected to junction JCT001"
3. "What is the material and diameter of pipe PIP001?"
4. "When was tank TNK001 installed?"
5. "Which valves need maintenance?"
6. "What is the path from valve VLV001 to tank TNK001?"

## Mock Data
For development purposes, create mock data for these components:

- Valves: VLV001, VLV002, VLV003
- Pipes: PIP001, PIP002, PIP003
- Junctions: JCT001, JCT002
- Tank: TNK001
- Pump: PMP001

Each component should have appropriate properties and be connected to form a simple network.

## Deliverables
1. A functional web application with all core features implemented
2. Source code with clear documentation
3. Setup instructions for running the application
4. Sample Neo4j database setup scripts or mock data

This MVP should demonstrate the core functionality of the Water Network Intelligence System while laying the foundation for future enhancements.
