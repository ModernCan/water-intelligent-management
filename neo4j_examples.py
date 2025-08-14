"""
Neo4j Water Network Examples
Based on 02-neo4j-fundamentals.md
"""

from neo4j import GraphDatabase
import sys


class WaterNetworkDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def test_connection(self):
        """Test the database connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful!' AS message")
                return result.single()["message"]
        except Exception as e:
            return f"Connection failed: {str(e)}"
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    def create_constraints(self):
        """Create unique constraints for water network components"""
        constraints = [
            "CREATE CONSTRAINT FOR (v:Valve) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (p:Pipe) REQUIRE p.id IS UNIQUE", 
            "CREATE CONSTRAINT FOR (j:Junction) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (t:Tank) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (pump:Pump) REQUIRE pump.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint}")
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
    
    def create_sample_network(self):
        """Create a sample water network"""
        queries = [
            # Create Junctions
            """
            CREATE (j1:Junction {
                id: 'JCT001', 
                elevation: 100.0, 
                location: 'North District'
            })
            """,
            """
            CREATE (j2:Junction {
                id: 'JCT002', 
                elevation: 95.0, 
                location: 'Central Plaza'
            })
            """,
            
            # Create Tank
            """
            CREATE (t1:Tank {
                id: 'TNK001',
                capacity: 1000000,
                level: 850000,
                elevation: 120.0,
                location: 'Reservoir Hill'
            })
            """,
            
            # Create Valves
            """
            CREATE (v1:Valve {
                id: 'VLV001',
                type: 'Gate Valve',
                diameter: 12,
                status: 'Open',
                installDate: date('2015-03-15')
            })
            """,
            """
            CREATE (v2:Valve {
                id: 'VLV002',
                type: 'Ball Valve',
                diameter: 8,
                status: 'Closed',
                installDate: date('2018-07-22')
            })
            """,
            
            # Create Pump
            """
            CREATE (pump1:Pump {
                id: 'PMP001',
                type: 'Centrifugal',
                capacity: 500,
                status: 'Running',
                power: 25.5
            })
            """,
            
            # Create Pipes
            """
            CREATE (p1:Pipe {
                id: 'PIP001',
                material: 'Cast Iron',
                diameter: 12,
                length: 500.0,
                installDate: date('2010-05-10')
            })
            """,
            """
            CREATE (p2:Pipe {
                id: 'PIP002',
                material: 'PVC',
                diameter: 8,
                length: 300.0,
                installDate: date('2020-01-15')
            })
            """
        ]
        
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
            print("Sample network components created.")
    
    def create_relationships(self):
        """Create relationships between components"""
        relationships = [
            # Tank feeds Junction 1 through Pipe 1
            "MATCH (t:Tank {id: 'TNK001'}), (p:Pipe {id: 'PIP001'}) CREATE (t)-[:FEEDS]->(p)",
            "MATCH (p:Pipe {id: 'PIP001'}), (j:Junction {id: 'JCT001'}) CREATE (p)-[:CONNECTS_TO]->(j)",
            
            # Junction 1 connects to Valve 1
            "MATCH (j:Junction {id: 'JCT001'}), (v:Valve {id: 'VLV001'}) CREATE (j)-[:CONNECTS_TO]->(v)",
            
            # Valve 1 connects to Junction 2 through Pipe 2
            "MATCH (v:Valve {id: 'VLV001'}), (p:Pipe {id: 'PIP002'}) CREATE (v)-[:CONNECTS_TO]->(p)",
            "MATCH (p:Pipe {id: 'PIP002'}), (j:Junction {id: 'JCT002'}) CREATE (p)-[:CONNECTS_TO]->(j)",
            
            # Pump connected to Junction 2
            "MATCH (pump:Pump {id: 'PMP001'}), (j:Junction {id: 'JCT002'}) CREATE (pump)-[:CONNECTS_TO]->(j)",
            
            # Valve 2 also connected to Junction 2
            "MATCH (v:Valve {id: 'VLV002'}), (j:Junction {id: 'JCT002'}) CREATE (v)-[:CONNECTS_TO]->(j)"
        ]
        
        with self.driver.session() as session:
            for rel in relationships:
                session.run(rel)
            print("Relationships created.")
    
    def run_sample_queries(self):
        """Run sample queries to demonstrate functionality"""
        queries = [
            ("Count all components", "MATCH (n) RETURN labels(n) AS type, count(n) AS count"),
            ("Find all valves", "MATCH (v:Valve) RETURN v.id, v.type, v.status"),
            ("Find components connected to JCT002", """
                MATCH (n)-[:CONNECTS_TO]-(j:Junction {id: 'JCT002'}) 
                RETURN n.id, labels(n) AS type
            """),
            ("Find path from tank to junction", """
                MATCH path = (t:Tank {id: 'TNK001'})-[*]-(j:Junction {id: 'JCT002'})
                RETURN [node in nodes(path) | node.id] AS path_components
            """),
            ("Find old pipes (installed before 2015)", """
                MATCH (p:Pipe) 
                WHERE p.installDate < date('2015-01-01')
                RETURN p.id, p.material, p.installDate
            """)
        ]
        
        with self.driver.session() as session:
            for description, query in queries:
                print(f"\n{description}:")
                print("-" * len(description))
                result = session.run(query)
                for record in result:
                    print(record)


def main():
    # Database connection parameters
    # Update these based on your Neo4j setup
    URI = "neo4j://localhost:7687"  # For local Neo4j
    # URI = "neo4j+s://your-instance.databases.neo4j.io"  # For Neo4j Aura
    USER = "neo4j"
    PASSWORD = "password"  # Update with your password
    
    print("Water Network Neo4j Examples")
    print("="*40)
    
    # Create database connection
    db = WaterNetworkDB(URI, USER, PASSWORD)
    
    try:
        # Test connection
        print("Testing connection...")
        message = db.test_connection()
        print(message)
        
        if "successful" not in message:
            print("Please update the connection parameters in the script.")
            return
        
        # Setup database
        print("\nSetting up database...")
        db.clear_database()
        db.create_constraints()
        
        # Create sample network
        print("\nCreating sample water network...")
        db.create_sample_network()
        db.create_relationships()
        
        # Run example queries
        print("\nRunning sample queries...")
        db.run_sample_queries()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Neo4j database is running")
        print("2. Connection parameters are correct")
        print("3. Database is accessible")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()