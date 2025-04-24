# Mini-Project 2: Network Analysis CLI Tool

## Overview
In this mini-project, you'll build a command-line interface (CLI) tool that performs common water network analyses using Neo4j. This tool will help you practice writing complex Cypher queries and understand the analytical capabilities of graph databases for water networks.

## Learning Objectives
- Write advanced Cypher queries for water network analysis
- Implement path-finding and network traversal algorithms
- Practice Neo4j integration with Python
- Learn to build a usable CLI tool
- Understand common water network analyses

## Dependencies
- **Phase 1 Content**: Complete at least Weeks 5-8 of the Foundation Phase
- **Skills Required**: Python, Neo4j, Cypher queries, basic water network concepts
- **Previous Mini-Projects**: Mini-Project 1 (using the network created or create a new one)

## Estimated Time: 1-2 weeks

## Project Steps

### Step 1: Setup Project Structure
1. Create a new Python project with the following structure:
```
water-network-cli/
├── water_network_cli/
│   ├── __init__.py
│   ├── cli.py
│   ├── database.py
│   ├── analyses.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_analyses.py
├── README.md
├── requirements.txt
└── setup.py
```

2. Set up the dependencies in `requirements.txt`:
```
neo4j
click
tabulate
rich
pytest
```

### Step 2: Implement Neo4j Connection
1. Create the database connection module in `database.py`:

```python
from neo4j import GraphDatabase

class WaterNetworkDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def verify_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            return False
            
    def run_query(self, query, parameters=None):
        if parameters is None:
            parameters = {}
            
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]
```

### Step 3: Implement Core Network Analyses
1. Create the analyses module in `analyses.py` with these functions:

```python
from .database import WaterNetworkDB

class WaterNetworkAnalyses:
    def __init__(self, db):
        self.db = db
        
    def component_count(self):
        """Count components by type"""
        query = """
        MATCH (n)
        WITH labels(n)[0] AS component_type, count(n) AS count
        RETURN component_type, count
        ORDER BY count DESC
        """
        return self.db.run_query(query)
        
    def find_path(self, start_id, end_id):
        """Find path between two components"""
        query = """
        MATCH (start {id: $start_id}), (end {id: $end_id})
        CALL apoc.path.expandConfig(start, {
            relationshipFilter: "CONNECTED_TO",
            maxLevel: 10,
            uniqueness: "NODE_GLOBAL",
            targetNodes: [end]
        })
        YIELD path
        RETURN path, length(path) AS length
        ORDER BY length ASC
        LIMIT 1
        """
        paths = self.db.run_query(query, {"start_id": start_id, "end_id": end_id})
        if not paths:
            return None
            
        # Process the path for display
        path = paths[0]["path"]
        nodes = [dict(node) for node in path.nodes]
        return nodes
        
    def find_critical_components(self):
        """Find components that, if removed, would disconnect the network"""
        query = """
        MATCH (n)
        WITH n, size((n)--()) AS connections
        WHERE connections > 1
        WITH n
        MATCH (a)--(n)--(b)
        WHERE id(a) < id(b)
        WITH n, count(DISTINCT [a, b]) AS potential_disconnects
        MATCH (a)--(n)--(b)
        WHERE id(a) < id(b)
        WITH n, potential_disconnects, a, b
        WHERE NOT (a)--(:)--(b) AND NOT (a)--(b)
        WITH n, potential_disconnects, count(DISTINCT [a, b]) AS actual_disconnects
        WHERE actual_disconnects > 0
        RETURN n.id AS component_id, labels(n)[0] AS component_type, 
               actual_disconnects AS disconnection_impact,
               1.0 * actual_disconnects / potential_disconnects AS criticality
        ORDER BY criticality DESC
        LIMIT 10
        """
        return self.db.run_query(query)
        
    def find_isolated_components(self):
        """Find components without connections"""
        query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN n.id AS component_id, labels(n)[0] AS component_type
        """
        return self.db.run_query(query)
        
    def zone_analysis(self, zone_id=None):
        """Analyze components by zone"""
        if zone_id:
            query = """
            MATCH (z {id: $zone_id})
            MATCH (c)-[:PART_OF]->(z)
            RETURN labels(c)[0] AS component_type, count(c) AS count
            ORDER BY count DESC
            """
            params = {"zone_id": zone_id}
        else:
            query = """
            MATCH (z)
            WHERE z:Zone OR z:PressureZone OR z:DMA OR z:SupplyZone
            OPTIONAL MATCH (c)-[:PART_OF]->(z)
            RETURN z.id AS zone_id, z.name AS zone_name,
                   count(c) AS component_count
            ORDER BY component_count DESC
            """
            params = {}
            
        return self.db.run_query(query, params)
        
    def age_analysis(self):
        """Analyze component age distribution"""
        query = """
        MATCH (c)
        WHERE exists(c.installDate)
        WITH c, date(c.installDate) AS install_date,
             duration.between(date(c.installDate), date()) AS age
        RETURN labels(c)[0] AS component_type,
               min(age.years) AS min_age_years,
               avg(age.years) AS avg_age_years,
               max(age.years) AS max_age_years,
               count(c) AS count
        ORDER BY avg_age_years DESC
        """
        return self.db.run_query(query)
```

### Step 4: Implement CLI Interface
1. Create the command-line interface in `cli.py`:

```python
import click
from rich.console import Console
from rich.table import Table
from .database import WaterNetworkDB
from .analyses import WaterNetworkAnalyses

console = Console()

@click.group()
@click.option('--uri', default="bolt://localhost:7687", help="Neo4j connection URI")
@click.option('--user', default="neo4j", help="Neo4j username")
@click.option('--password', prompt=True, hide_input=True, help="Neo4j password")
@click.pass_context
def cli(ctx, uri, user, password):
    """Water Network Analysis CLI Tool"""
    ctx.ensure_object(dict)
    
    # Connect to database
    db = WaterNetworkDB(uri, user, password)
    if not db.verify_connection():
        console.print("[bold red]Failed to connect to Neo4j database[/bold red]")
        exit(1)
        
    ctx.obj['db'] = db
    ctx.obj['analyses'] = WaterNetworkAnalyses(db)
    
@cli.command()
@click.pass_context
def components(ctx):
    """Count components by type"""
    results = ctx.obj['analyses'].component_count()
    
    # Create rich table
    table = Table(title="Component Count")
    table.add_column("Component Type", style="cyan")
    table.add_column("Count", style="magenta")
    
    for result in results:
        table.add_row(result["component_type"], str(result["count"]))
    
    console.print(table)

@cli.command()
@click.argument('start_id')
@click.argument('end_id')
@click.pass_context
def path(ctx, start_id, end_id):
    """Find path between two components"""
    nodes = ctx.obj['analyses'].find_path(start_id, end_id)
    
    if not nodes:
        console.print(f"[bold red]No path found between {start_id} and {end_id}[/bold red]")
        return
    
    console.print(f"[bold green]Path found ({len(nodes)-1} steps):[/bold green]")
    
    for i, node in enumerate(nodes):
        node_type = next(iter(node.get("labels", ["Unknown"])), "Unknown")
        console.print(f"{i+1}. [cyan]{node_type}[/cyan]: [yellow]{node.get('id')}[/yellow]")
        if i < len(nodes) - 1:
            console.print("   │")
            console.print("   ▼")

@cli.command()
@click.pass_context
def critical(ctx):
    """Find critical components"""
    results = ctx.obj['analyses'].find_critical_components()
    
    if not results:
        console.print("[yellow]No critical components found[/yellow]")
        return
    
    table = Table(title="Critical Components")
    table.add_column("Component ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Disconnection Impact", style="magenta")
    table.add_column("Criticality", style="red")
    
    for result in results:
        table.add_row(
            result["component_id"],
            result["component_type"],
            str(result["disconnection_impact"]),
            f"{result['criticality']:.2f}"
        )
    
    console.print(table)

@cli.command()
@click.option('--zone-id', help="Analyze a specific zone")
@click.pass_context
def zones(ctx, zone_id):
    """Analyze zones"""
    results = ctx.obj['analyses'].zone_analysis(zone_id)
    
    if not results:
        console.print("[yellow]No zone data found[/yellow]")
        return
    
    if zone_id:
        table = Table(title=f"Components in Zone {zone_id}")
        table.add_column("Component Type", style="cyan")
        table.add_column("Count", style="magenta")
        
        for result in results:
            table.add_row(result["component_type"], str(result["count"]))
    else:
        table = Table(title="Zones Overview")
        table.add_column("Zone ID", style="cyan")
        table.add_column("Zone Name", style="green")
        table.add_column("Component Count", style="magenta")
        
        for result in results:
            table.add_row(
                result["zone_id"],
                result.get("zone_name", "Unnamed"),
                str(result["component_count"])
            )
    
    console.print(table)

@cli.command()
@click.pass_context
def age(ctx):
    """Analyze component age"""
    results = ctx.obj['analyses'].age_analysis()
    
    if not results:
        console.print("[yellow]No age data found[/yellow]")
        return
    
    table = Table(title="Component Age Analysis")
    table.add_column("Component Type", style="cyan")
    table.add_column("Min Age (years)", style="green")
    table.add_column("Avg Age (years)", style="yellow")
    table.add_column("Max Age (years)", style="red")
    table.add_column("Count", style="magenta")
    
    for result in results:
        table.add_row(
            result["component_type"],
            str(result["min_age_years"]),
            f"{result['avg_age_years']:.1f}",
            str(result["max_age_years"]),
            str(result["count"])
        )
    
    console.print(table)

def main():
    cli(obj={})

if __name__ == '__main__':
    main()
```

### Step 5: Set up Package Installation
1. Create `setup.py` for the package:

```python
from setuptools import setup, find_packages

setup(
    name="water-network-cli",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "neo4j",
        "click",
        "tabulate",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "water-network=water_network_cli.cli:main",
        ],
    },
)
```

### Step 6: Create Tests
1. Implement basic tests for the analyses in `tests/test_analyses.py`:

```python
import pytest
from unittest.mock import MagicMock
from water_network_cli.analyses import WaterNetworkAnalyses

@pytest.fixture
def mock_db():
    db = MagicMock()
    return db

@pytest.fixture
def analyses(mock_db):
    return WaterNetworkAnalyses(mock_db)

def test_component_count(analyses, mock_db):
    # Setup mock return value
    mock_db.run_query.return_value = [
        {"component_type": "Valve", "count": 10},
        {"component_type": "Pipe", "count": 20}
    ]
    
    # Run the analysis
    result = analyses.component_count()
    
    # Verify the query was called and returned expected results
    mock_db.run_query.assert_called_once()
    assert len(result) == 2
    assert result[0]["component_type"] == "Valve"
    assert result[0]["count"] == 10

# Add more tests for other analyses...
```

### Step 7: Install and Test
1. Install the package in development mode:
```bash
pip install -e .
```

2. Run the CLI commands to test:
```bash
# List components
water-network components

# Find a path
water-network path VLV001 TNK001

# List critical components
water-network critical

# Analyze zones
water-network zones

# Analyze component age
water-network age
```

### Step 8: Add Custom Queries
1. Extend the CLI to support custom queries:

```python
@cli.command()
@click.argument('query')
@click.pass_context
def custom(ctx, query):
    """Run a custom Cypher query"""
    try:
        results = ctx.obj['db'].run_query(query)
        
        if not results:
            console.print("[yellow]Query returned no results[/yellow]")
            return
        
        # Create dynamic table based on first result keys
        table = Table(title="Custom Query Results")
        columns = results[0].keys()
        
        for column in columns:
            table.add_column(column)
        
        # Add rows
        for result in results:
            table.add_row(*[str(result.get(column, "")) for column in columns])
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error executing query: {str(e)}[/bold red]")
```

### Deliverables
1. A functional CLI tool for water network analysis
2. Documentation on supported analyses
3. Sample queries and usage examples
4. Test suite for the analyses

## Extensions
1. Add data export capabilities (CSV, JSON)
2. Create visualizations for critical components
3. Add network connectivity metrics
4. Implement maintenance scheduling recommendations
5. Add performance comparison across different zones
6. Create custom network traversal algorithms specific to water networks

## Relation to Main Project
This mini-project deepens your understanding of Cypher queries and graph traversal algorithms, which are fundamental to the Foundation Phase. The CLI tool you build will be useful for quick analyses throughout the main project, and the skills you develop in creating analytical queries will directly support the GraphRAG implementation in Phase 2.
