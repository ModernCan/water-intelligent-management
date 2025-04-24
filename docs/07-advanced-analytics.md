# Advanced Analytics and Simulation for Water Networks

This guide covers Phase 4 of your Water Network Intelligence System, focusing on implementing advanced analytics, simulation capabilities, and role-based interfaces. By the end of this guide, you'll be able to add predictive capabilities and domain-specific intelligence to your system.

## 1. Predictive Analytics for Water Infrastructure

Water utilities need to predict potential failures before they occur. Let's implement predictive analytics using the graph structure we've built.

### 1.1 Setting Up the Predictive Analytics Environment

First, let's set up our environment with the necessary libraries:

```python
# Install required packages
# pip install scikit-learn pandas matplotlib numpy neo4j

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import pickle
import os
```

### 1.2 Feature Engineering from Graph Data

The first step is to extract meaningful features from our graph database:

```python
class WaterNetworkFeatureExtractor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def extract_pipe_features(self):
        """Extract features for pipe failure prediction"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (p:Pipe)
            OPTIONAL MATCH (p)-[:HAS_MAINTENANCE]->(m:Maintenance)
            WITH p, 
                 p.installDate as installDate,
                 p.material as material,
                 p.diameter as diameter,
                 p.length as length,
                 p.roughness as roughness,
                 COUNT(m) as maintenanceCount,
                 COLLECT(m.date) as maintenanceDates,
                 COLLECT(m.type) as maintenanceTypes,
                 COLLECT(m.findings) as maintenanceFindings
            RETURN p.id as pipeId, 
                   installDate, 
                   material, 
                   diameter, 
                   length,
                   roughness,
                   maintenanceCount,
                   maintenanceDates,
                   maintenanceTypes,
                   maintenanceFindings,
                   // Calculate pipe age in years
                   duration.between(date(installDate), date()).years as ageYears
            """)
            
            records = [dict(record) for record in result]
            return pd.DataFrame(records)
    
    def extract_valve_features(self):
        """Extract features for valve failure prediction"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (v:Valve)
            OPTIONAL MATCH (v)-[:HAS_MAINTENANCE]->(m:Maintenance)
            WITH v, 
                 v.installDate as installDate,
                 v.valveType as valveType,
                 v.diameter as diameter,
                 COUNT(m) as maintenanceCount,
                 COLLECT(m.date) as maintenanceDates,
                 COLLECT(m.type) as maintenanceTypes,
                 COLLECT(m.findings) as maintenanceFindings
            RETURN v.id as valveId, 
                   installDate, 
                   valveType, 
                   diameter,
                   maintenanceCount,
                   maintenanceDates,
                   maintenanceTypes,
                   maintenanceFindings,
                   // Calculate valve age in years
                   duration.between(date(installDate), date()).years as ageYears
            """)
            
            records = [dict(record) for record in result]
            return pd.DataFrame(records)
    
    def extract_network_features(self):
        """Extract network-level features like connectivity"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c)
            WHERE c:Valve OR c:Pipe OR c:Junction OR c:Pump
            MATCH (c)-[r]-()
            WITH c, COUNT(r) as connectionCount
            RETURN c.id as componentId, 
                   labels(c)[0] as componentType,
                   connectionCount
            """)
            
            records = [dict(record) for record in result]
            return pd.DataFrame(records)
```

### 1.3 Creating a Pipe Failure Prediction Model

Now, let's implement a model to predict pipe failures:

```python
class PipeFailurePredictionModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def preprocess_data(self, pipe_data):
        """Preprocess data for modeling"""
        # Convert categorical features to one-hot encoding
        df = pipe_data.copy()
        
        # Create binary target (has had issues or not)
        df['has_had_issues'] = df['maintenanceFindings'].apply(
            lambda x: any('issue' in str(finding).lower() or 
                         'leak' in str(finding).lower() or 
                         'break' in str(finding).lower() 
                         for finding in x if finding)
        )
        
        # Handle material as categorical
        material_dummies = pd.get_dummies(df['material'], prefix='material')
        df = pd.concat([df, material_dummies], axis=1)
        
        # Create numerical features
        numerical_features = ['diameter', 'length', 'roughness', 'ageYears', 'maintenanceCount']
        categorical_features = [col for col in df.columns if 'material_' in col]
        
        # Store feature columns for inference
        self.feature_columns = numerical_features + categorical_features
        
        # Fill missing values
        for col in numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def train(self, pipe_data):
        """Train the pipe failure prediction model"""
        df = self.preprocess_data(pipe_data)
        
        # Prepare training data
        X = df[self.feature_columns]
        y = df['has_had_issues']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.head(10))
        
        return self.model
    
    def predict_failure_probability(self, pipe_features):
        """Predict failure probability for a pipe"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X = pipe_features[self.feature_columns]
        
        # Predict probability
        probabilities = self.model.predict_proba(X)
        
        # Return probability of failure (class 1)
        return probabilities[:, 1]
    
    def save_model(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, f)
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
```

### 1.4 Creating a Remaining Useful Life Model

Let's also implement a model to predict the remaining useful life of components:

```python
class RemainingUsefulLifeModel:
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def preprocess_data(self, component_data, target_component_type='Pipe'):
        """Preprocess data for RUL modeling"""
        df = component_data.copy()
        
        # Filter for specific component type if needed
        if 'componentType' in df.columns:
            df = df[df['componentType'] == target_component_type]
        
        # Generate RUL target based on age and typical lifespan
        typical_lifespan = {
            'Pipe': {
                'PVC': 50,
                'Ductile Iron': 75,
                'Cast Iron': 50,
                'Steel': 40,
                'Concrete': 60
            },
            'Valve': {
                'Gate': 30,
                'Butterfly': 25,
                'Check': 20,
                'Pressure Reducing': 15
            },
            'Pump': {
                'Centrifugal': 20,
                'Turbine': 25,
                'Submersible': 15
            }
        }
        
        # Calculate estimated RUL (simplified version)
        if target_component_type == 'Pipe':
            df['typical_lifespan'] = df['material'].map(
                lambda x: typical_lifespan.get('Pipe', {}).get(x, 50)
            )
        elif target_component_type == 'Valve':
            df['typical_lifespan'] = df['valveType'].map(
                lambda x: typical_lifespan.get('Valve', {}).get(x, 25)
            )
        
        # Calculate target RUL based on typical lifespan and current age
        df['rul'] = df['typical_lifespan'] - df['ageYears']
        
        # Apply maintenance impact - adjust RUL based on maintenance history
        # For simplicity, we reduce RUL if there are findings with negative words
        if 'maintenanceFindings' in df.columns:
            df['maintenance_issues'] = df['maintenanceFindings'].apply(
                lambda x: sum(1 for finding in x if finding and any(
                    word in str(finding).lower() for word in 
                    ['corrosion', 'wear', 'leak', 'damage', 'deterioration']
                ))
            )
            # Reduce RUL by a factor based on maintenance issues
            df['rul'] = df['rul'] - (df['maintenance_issues'] * 2)
        
        # Ensure RUL is never negative
        df['rul'] = df['rul'].clip(lower=0)
        
        # Define features for the model
        numerical_features = ['diameter', 'ageYears', 'maintenanceCount']
        if 'length' in df.columns:
            numerical_features.append('length')
        if 'roughness' in df.columns:
            numerical_features.append('roughness')
        
        # Material as categorical if available
        if 'material' in df.columns:
            material_dummies = pd.get_dummies(df['material'], prefix='material')
            df = pd.concat([df, material_dummies], axis=1)
            categorical_features = [col for col in df.columns if 'material_' in col]
        # Valve type as categorical if available
        elif 'valveType' in df.columns:
            valvetype_dummies = pd.get_dummies(df['valveType'], prefix='valveType')
            df = pd.concat([df, valvetype_dummies], axis=1)
            categorical_features = [col for col in df.columns if 'valveType_' in col]
        else:
            categorical_features = []
        
        # Store feature columns for inference
        self.feature_columns = numerical_features + categorical_features
        
        # Fill missing values
        for col in numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def train(self, component_data, target_component_type='Pipe'):
        """Train the RUL prediction model"""
        df = self.preprocess_data(component_data, target_component_type)
        
        # Prepare training data
        X = df[self.feature_columns]
        y = df['rul']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model RMSE: {rmse:.2f} years")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance.head(10))
        
        # Create plot of predicted vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
        plt.xlabel("Actual RUL (years)")
        plt.ylabel("Predicted RUL (years)")
        plt.title(f"{target_component_type} Remaining Useful Life Prediction")
        plt.tight_layout()
        plt.savefig(f"{target_component_type.lower()}_rul_prediction.png")
        
        return self.model
    
    def predict_rul(self, component_features):
        """Predict remaining useful life for components"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X = component_features[self.feature_columns]
        
        # Predict RUL
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, f)
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
```

### 1.5 Integrating Predictive Models with Neo4j

Now, let's integrate our predictive models with the Neo4j database:

```python
class PredictiveMaintenanceSystem:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.feature_extractor = WaterNetworkFeatureExtractor(uri, user, password)
        self.pipe_failure_model = PipeFailurePredictionModel()
        self.pipe_rul_model = RemainingUsefulLifeModel()
        self.valve_rul_model = RemainingUsefulLifeModel()
    
    def close(self):
        self.driver.close()
        self.feature_extractor.close()
    
    def train_models(self):
        """Train all predictive models"""
        # Extract features
        pipe_data = self.feature_extractor.extract_pipe_features()
        valve_data = self.feature_extractor.extract_valve_features()
        
        # Train pipe failure prediction model
        print("Training pipe failure prediction model...")
        self.pipe_failure_model.train(pipe_data)
        
        # Train pipe RUL model
        print("\nTraining pipe remaining useful life model...")
        self.pipe_rul_model.train(pipe_data, 'Pipe')
        
        # Train valve RUL model
        print("\nTraining valve remaining useful life model...")
        self.valve_rul_model.train(valve_data, 'Valve')
        
        # Save models
        os.makedirs('models', exist_ok=True)
        self.pipe_failure_model.save_model('models/pipe_failure_model.pkl')
        self.pipe_rul_model.save_model('models/pipe_rul_model.pkl')
        self.valve_rul_model.save_model('models/valve_rul_model.pkl')
    
    def update_neo4j_with_predictions(self):
        """Update Neo4j with prediction results"""
        # Extract features
        pipe_data = self.feature_extractor.extract_pipe_features()
        valve_data = self.feature_extractor.extract_valve_features()
        
        # Generate predictions
        pipe_failure_probs = self.pipe_failure_model.predict_failure_probability(pipe_data)
        pipe_rul = self.pipe_rul_model.predict_rul(pipe_data)
        valve_rul = self.valve_rul_model.predict_rul(valve_data)
        
        # Add predictions to dataframes
        pipe_data['failure_probability'] = pipe_failure_probs
        pipe_data['remaining_useful_life'] = pipe_rul
        valve_data['remaining_useful_life'] = valve_rul
        
        # Update Neo4j with pipe predictions
        with self.driver.session() as session:
            for _, row in pipe_data.iterrows():
                session.run("""
                MATCH (p:Pipe {id: $pipe_id})
                SET p.failureProbability = $failure_prob,
                    p.remainingUsefulLife = $rul,
                    p.lastPredictionDate = date()
                """, {
                    'pipe_id': row['pipeId'],
                    'failure_prob': float(row['failure_probability']),
                    'rul': float(row['remaining_useful_life'])
                })
            
            # Update Neo4j with valve predictions
            for _, row in valve_data.iterrows():
                session.run("""
                MATCH (v:Valve {id: $valve_id})
                SET v.remainingUsefulLife = $rul,
                    v.lastPredictionDate = date()
                """, {
                    'valve_id': row['valveId'],
                    'rul': float(row['remaining_useful_life'])
                })
            
            print(f"Updated {len(pipe_data)} pipes and {len(valve_data)} valves with predictions")
    
    def get_high_risk_components(self, failure_probability_threshold=0.7, rul_threshold=5):
        """Get components at high risk of failure"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (c)
            WHERE (c:Pipe AND c.failureProbability >= $failure_threshold) OR
                  ((c:Pipe OR c:Valve) AND c.remainingUsefulLife <= $rul_threshold)
            RETURN c.id as id, labels(c)[0] as type, 
                   c.failureProbability as failureProbability,
                   c.remainingUsefulLife as remainingUsefulLife
            ORDER BY coalesce(c.failureProbability, 0) DESC, 
                     coalesce(c.remainingUsefulLife, 1000)
            """, {
                'failure_threshold': failure_probability_threshold,
                'rul_threshold': rul_threshold
            })
            
            return [dict(record) for record in result]
```

### 1.6 Using the Predictive Maintenance System

Here's how you would use the predictive maintenance system:

```python
# Initialize the system
uri = "neo4j://localhost:7687"
user = "neo4j"
password = "your_password"

predictive_system = PredictiveMaintenanceSystem(uri, user, password)

# Train the models
predictive_system.train_models()

# Update Neo4j with predictions
predictive_system.update_neo4j_with_predictions()

# Get high-risk components
high_risk = predictive_system.get_high_risk_components()
print(f"Found {len(high_risk)} high-risk components:")
for component in high_risk[:10]:  # Show top 10
    print(f"{component['type']} {component['id']}: "
          f"Failure Probability: {component.get('failureProbability', 'N/A'):.2f}, "
          f"RUL: {component.get('remainingUsefulLife', 'N/A'):.1f} years")

# Clean up
predictive_system.close()
```

## 2. Water Network Simulation

Now, let's implement simulation capabilities to conduct what-if analyses for the water network.

### 2.1 Setting Up the Simulation Environment

First, let's set up our environment for network simulation:

```python
# Install required packages
# pip install epanettools networkx matplotlib

from epanettools.epanettools import EPANetSimulation, Node, Link, Networks
import networkx as nx
import matplotlib.pyplot as plt
import json
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
```

### 2.2 Converting Neo4j Graph to EPANET Model

We'll create a converter that transforms our Neo4j graph into an EPANET model for hydraulic simulation:

```python
class Neo4jToEPANETConverter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def extract_network(self):
        """Extract water network from Neo4j"""
        with self.driver.session() as session:
            # Extract junctions
            junctions = session.run("""
            MATCH (j:Junction)
            RETURN j.id as id, 
                   j.elevation as elevation,
                   j.basedemand as basedemand
            """)
            
            # Extract reservoirs
            reservoirs = session.run("""
            MATCH (r:Reservoir)
            RETURN r.id as id, 
                   r.elevation as elevation
            """)
            
            # Extract tanks
            tanks = session.run("""
            MATCH (t:Tank)
            RETURN t.id as id, 
                   t.elevation as elevation,
                   t.diameter as diameter,
                   t.height as height,
                   t.initialLevel as initialLevel,
                   t.minLevel as minLevel,
                   t.maxLevel as maxLevel
            """)
            
            # Extract pipes
            pipes = session.run("""
            MATCH (n1)-[:CONNECTED_TO]->(p:Pipe)-[:CONNECTED_TO]->(n2)
            RETURN p.id as id, 
                   n1.id as node1,
                   n2.id as node2,
                   p.length as length,
                   p.diameter as diameter,
                   p.roughness as roughness
            """)
            
            # Extract pumps
            pumps = session.run("""
            MATCH (n1)-[:CONNECTED_TO]->(p:Pump)-[:CONNECTED_TO]->(n2)
            RETURN p.id as id, 
                   n1.id as node1,
                   n2.id as node2,
                   p.power as power,
                   p.speed as speed
            """)
            
            # Extract valves
            valves = session.run("""
            MATCH (n1)-[:CONNECTED_TO]->(v:Valve)-[:CONNECTED_TO]->(n2)
            RETURN v.id as id, 
                   n1.id as node1,
                   n2.id as node2,
                   v.diameter as diameter,
                   v.valveType as type,
                   v.setting as setting
            """)
            
            # Convert to lists of dictionaries
            return {
                'junctions': [dict(record) for record in junctions],
                'reservoirs': [dict(record) for record in reservoirs],
                'tanks': [dict(record) for record in tanks],
                'pipes': [dict(record) for record in pipes],
                'pumps': [dict(record) for record in pumps],
                'valves': [dict(record) for record in valves]
            }
    
    def create_inp_file(self, network_data, output_file='water_network.inp'):
        """Create an EPANET INP file from extracted network data"""
        with open(output_file, 'w') as f:
            # Write file header
            f.write('[TITLE]\n')
            f.write('Water Network Model converted from Neo4j\n\n')
            
            # Write junctions
            f.write('[JUNCTIONS]\n')
            f.write(';ID              Elev        Demand      Pattern         \n')
            for j in network_data['junctions']:
                f.write(f"{j['id']}              {j.get('elevation', 0)}          {j.get('basedemand', 0)}          \n")
            f.write('\n')
            
            # Write reservoirs
            f.write('[RESERVOIRS]\n')
            f.write(';ID              Head        Pattern         \n')
            for r in network_data['reservoirs']:
                f.write(f"{r['id']}              {r.get('elevation', 0)}                    \n")
            f.write('\n')
            
            # Write tanks
            f.write('[TANKS]\n')
            f.write(';ID              Elevation   InitLevel   MinLevel    MaxLevel    Diameter    MinVol      VolCurve\n')
            for t in network_data['tanks']:
                f.write(f"{t['id']}              {t.get('elevation', 0)}          "
                       f"{t.get('initialLevel', 0)}          {t.get('minLevel', 0)}          "
                       f"{t.get('maxLevel', 0)}          {t.get('diameter', 0)}            0                    \n")
            f.write('\n')
            
            # Write pipes
            f.write('[PIPES]\n')
            f.write(';ID              Node1           Node2           Length      Diameter    Roughness   MinorLoss   Status\n')
            for p in network_data['pipes']:
                f.write(f"{p['id']}              {p['node1']}              {p['node2']}              "
                       f"{p.get('length', 100)}         {p.get('diameter', 100)}         {p.get('roughness', 100)}         0            Open  \n")
            f.write('\n')
            
            # Write pumps
            f.write('[PUMPS]\n')
            f.write(';ID              Node1           Node2           Parameters\n')
            for p in network_data['pumps']:
                f.write(f"{p['id']}              {p['node1']}              {p['node2']}              "
                       f"HEAD 1 POWER {p.get('power', 75)}\n")
            f.write('\n')
            
            # Write valves
            f.write('[VALVES]\n')
            f.write(';ID              Node1           Node2           Diameter    Type        Setting     MinorLoss   \n')
            for v in network_data['valves']:
                valve_type = {
                    'Pressure Reducing': 'PRV',
                    'Flow Control': 'FCV',
                    'Pressure Sustaining': 'PSV',
                    'Throttle Control': 'TCV',
                    'General Purpose': 'GPV'
                }.get(v.get('type'), 'TCV')
                
                f.write(f"{v['id']}              {v['node1']}              {v['node2']}              "
                       f"{v.get('diameter', 100)}         {valve_type}         {v.get('setting', 0)}           0            \n")
            f.write('\n')
            
            # Write options
            f.write('[OPTIONS]\n')
            f.write('Units              GPM\n')
            f.write('Headloss           H-W\n')
            f.write('Specific Gravity   1.0\n')
            f.write('Viscosity          1.0\n')
            f.write('Trials             40\n')
            f.write('Accuracy           0.001\n')
            f.write('CHECKFREQ          2\n')
            f.write('MAXCHECK           10\n')
            f.write('DAMPLIMIT          0\n')
            f.write('Unbalanced         Continue 10\n')
            f.write('Pattern            1\n')
            f.write('Demand Multiplier  1.0\n')
            f.write('Emitter Exponent   0.5\n')
            f.write('Quality            None mg/L\n')
            f.write('Diffusivity        1.0\n')
            f.write('Tolerance          0.01\n')
            f.write('\n')
            
            # Write times
            f.write('[TIMES]\n')
            f.write('Duration           24:00\n')
            f.write('Hydraulic Timestep 1:00\n')
            f.write('Quality Timestep   0:05\n')
            f.write('Pattern Timestep   1:00\n')
            f.write('Pattern Start      0:00\n')
            f.write('Report Timestep    1:00\n')
            f.write('Report Start       0:00\n')
            f.write('Start ClockTime    0:00\n')
            f.write('Statistic          NONE\n')
            f.write('\n')
            
            # Write report
            f.write('[REPORT]\n')
            f.write('Status             Yes\n')
            f.write('Summary            Yes\n')
            f.write('Page               0\n')
            f.write('\n')
            
            # End file
            f.write('[END]\n')
        
        print(f"Created EPANET input file: {output_file}")
        return output_file
```

### 2.3 Implementing the Water Network Simulator

Now let's create a simulator class that can run various scenarios:

```python
class WaterNetworkSimulator:
    def __init__(self, inp_file):
        """Initialize with an EPANET INP file"""
        self.inp_file = inp_file
        self.sim = EPANetSimulation(inp_file)
        self.net = Networks()[0]
        
        # Initialize simulation
        self.sim.loadEPANETFile(self.inp_file)
    
    def run_simulation(self):
        """Run a basic hydraulic simulation"""
        # Run the simulation
        self.sim.run()
        
        # Get results
        results = {
            'nodes': {},
            'links': {}
        }
        
        # Collect node results (pressure, demand)
        for node_id in range(1, self.sim.NNODES + 1):
            node_type = self.sim.getNodeTypeIndex(node_id)
            node_name = self.sim.getNodeNameID(node_id)
            
            if node_type == Node.JUNCTION:
                results['nodes'][node_name] = {
                    'type': 'junction',
                    'pressure': self.sim.getNodePressure(node_id),
                    'demand': self.sim.getNodeActualDemand(node_id),
                    'head': self.sim.getNodeHydraulicHead(node_id)
                }
            elif node_type == Node.RESERVOIR:
                results['nodes'][node_name] = {
                    'type': 'reservoir',
                    'head': self.sim.getNodeHydraulicHead(node_id)
                }
            elif node_type == Node.TANK:
                results['nodes'][node_name] = {
                    'type': 'tank',
                    'volume': self.sim.getNodeTankVolume(node_id),
                    'level': self.sim.getNodeActualDemand(node_id),
                    'head': self.sim.getNodeHydraulicHead(node_id)
                }
        
        # Collect link results (flow, velocity, headloss)
        for link_id in range(1, self.sim.NLINKS + 1):
            link_type = self.sim.getLinkTypeIndex(link_id)
            link_name = self.sim.getLinkNameID(link_id)
            
            if link_type == Link.PIPE:
                results['links'][link_name] = {
                    'type': 'pipe',
                    'flow': self.sim.getLinkFlows(link_id),
                    'velocity': self.sim.getLinkVelocity(link_id),
                    'headloss': self.sim.getLinkHeadloss(link_id),
                    'status': self.sim.getLinkStatus(link_id)
                }
            elif link_type == Link.PUMP:
                results['links'][link_name] = {
                    'type': 'pump',
                    'flow': self.sim.getLinkFlows(link_id),
                    'energy': self.sim.getLinkEnergy(link_id),
                    'status': self.sim.getLinkStatus(link_id)
                }
            elif link_type in [Link.PRV, Link.PSV, Link.PBV, Link.FCV, Link.TCV, Link.GPV]:
                results['links'][link_name] = {
                    'type': 'valve',
                    'flow': self.sim.getLinkFlows(link_id),
                    'velocity': self.sim.getLinkVelocity(link_id),
                    'headloss': self.sim.getLinkHeadloss(link_id),
                    'status': self.sim.getLinkStatus(link_id)
                }
        
        return results
    
    def simulate_pipe_closure(self, pipe_id):
        """Simulate closing a specific pipe"""
        # Find the link index
        link_index = self.sim.getLinkIndex(pipe_id)
        
        # Set the pipe status to closed
        self.sim.setLinkStatus(link_index, 0)  # 0 = CLOSED
        
        # Run simulation
        results = self.run_simulation()
        
        # Reset the pipe status to open for future simulations
        self.sim.setLinkStatus(link_index, 1)  # 1 = OPEN
        
        return results
    
    def simulate_increased_demand(self, junction_id, demand_multiplier=2.0):
        """Simulate increased demand at a junction"""
        # Find the node index
        node_index = self.sim.getNodeIndex(junction_id)
        
        # Get current base demand
        base_demand = self.sim.getNodeBaseDemand(node_index, 0)
        
        # Set increased demand
        self.sim.setNodeBaseDemand(node_index, base_demand * demand_multiplier, 0)
        
        # Run simulation
        results = self.run_simulation()
        
        # Reset demand to original value
        self.sim.setNodeBaseDemand(node_index, base_demand, 0)
        
        return results
    
    def simulate_pump_failure(self, pump_id):
        """Simulate a pump failure"""
        # Find the link index
        link_index = self.sim.getLinkIndex(pump_id)
        
        # Set the pump status to closed
        self.sim.setLinkStatus(link_index, 0)  # 0 = CLOSED
        
        # Run simulation
        results = self.run_simulation()
        
        # Reset the pump status to open
        self.sim.setLinkStatus(link_index, 1)  # 1 = OPEN
        
        return results
    
    def simulate_pressure_zone_changes(self, valve_id, new_setting):
        """Simulate changing pressure zone settings"""
        # Find the link index
        link_index = self.sim.getLinkIndex(valve_id)
        
        # Store original setting
        original_setting = self.sim.getLinkSettings(link_index)
        
        # Set new pressure
        self.sim.setLinkSettings(link_index, new_setting)
        
        # Run simulation
        results = self.run_simulation()
        
        # Reset to original setting
        self.sim.setLinkSettings(link_index, original_setting)
        
        return results
    
    def identify_critical_components(self):
        """Identify critical components by simulating failures"""
        # Get baseline simulation results
        baseline = self.run_simulation()
        
        critical_components = []
        
        # Test each pipe
        for link_id in range(1, self.sim.NLINKS + 1):
            if self.sim.getLinkTypeIndex(link_id) == Link.PIPE:
                link_name = self.sim.getLinkNameID(link_id)
                
                # Simulate closure
                self.sim.setLinkStatus(link_id, 0)  # Close
                self.sim.run()
                
                # Check for pressure problems
                pressure_problems = 0
                flow_problems = 0
                
                for node_id in range(1, self.sim.NNODESPLUS + 1):
                    if self.sim.getNodeTypeIndex(node_id) == Node.JUNCTION:
                        pressure = self.sim.getNodePressure(node_id)
                        baseline_pressure = baseline['nodes'][self.sim.getNodeNameID(node_id)]['pressure']
                        
                        if pressure < 20 or pressure < baseline_pressure * 0.5:
                            pressure_problems += 1
                
                for other_link_id in range(1, self.sim.NLINKS + 1):
                    if other_link_id != link_id and self.sim.getLinkTypeIndex(other_link_id) == Link.PIPE:
                        flow = abs(self.sim.getLinkFlows(other_link_id))
                        baseline_flow = abs(baseline['links'][self.sim.getLinkNameID(other_link_id)]['flow'])
                        
                        if flow > baseline_flow * 1.5:
                            flow_problems += 1
                
                # Reset the pipe status
                self.sim.setLinkStatus(link_id, 1)  # Open
                
                # If this component causes significant problems, mark as critical
                if pressure_problems > 0 or flow_problems > 0:
                    critical_components.append({
                        'id': link_name,
                        'type': 'pipe',
                        'pressure_problems': pressure_problems,
                        'flow_problems': flow_problems,
                        'criticality_score': pressure_problems + flow_problems
                    })
        
        # Sort by criticality
        critical_components.sort(key=lambda x: x['criticality_score'], reverse=True)
        
        return critical_components
```

### 2.4 Using the Water Network Simulator

Here's how to use the simulator:

```python
# Convert Neo4j graph to EPANET model
converter = Neo4jToEPANETConverter(uri, user, password)
network_data = converter.extract_network()
inp_file = converter.create_inp_file(network_data)
converter.close()

# Initialize simulator
simulator = WaterNetworkSimulator(inp_file)

# Run baseline simulation
baseline_results = simulator.run_simulation()
print("Baseline simulation completed")

# Identify critical components
critical = simulator.identify_critical_components()
print(f"Identified {len(critical)} critical components")
for comp in critical[:5]:  # Show top 5
    print(f"{comp['type']} {comp['id']}: Criticality Score: {comp['criticality_score']}")

# Simulate a specific pipe closure
pipe_id = "PIP001"  # Replace with an actual pipe ID
closure_results = simulator.simulate_pipe_closure(pipe_id)
print(f"\nSimulation results for {pipe_id} closure:")

# Calculate impact
pressure_drops = []
flow_increases = []

for node_id, node_data in closure_results['nodes'].items():
    if node_data['type'] == 'junction':
        baseline_pressure = baseline_results['nodes'][node_id]['pressure']
        new_pressure = node_data['pressure']
        if baseline_pressure > 0:
            pressure_change = (new_pressure - baseline_pressure) / baseline_pressure * 100
            if pressure_change < -10:  # More than 10% drop
                pressure_drops.append((node_id, pressure_change))

for link_id, link_data in closure_results['links'].items():
    if link_data['type'] == 'pipe':
        baseline_flow = abs(baseline_results['links'][link_id]['flow'])
        new_flow = abs(link_data['flow'])
        if baseline_flow > 0:
            flow_change = (new_flow - baseline_flow) / baseline_flow * 100
            if flow_change > 50:  # More than 50% increase
                flow_increases.append((link_id, flow_change))

print(f"Pressure drops at {len(pressure_drops)} junctions")
for node_id, change in sorted(pressure_drops, key=lambda x: x[1])[:5]:
    print(f"  {node_id}: {change:.1f}% change")

print(f"Flow increases in {len(flow_increases)} pipes")
for link_id, change in sorted(flow_increases, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {link_id}: {change:.1f}% change")
```

## 3. Advanced NLP for Domain-Specific Queries

Let's implement advanced natural language processing for water-network-specific queries.

### 3.1 Setting Up the Domain-Specific NLP Environment

```python
# Install required packages
# pip install transformers langchain-community langchain-openai

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from neo4j import GraphDatabase
import json
import re
```

### 3.2 Creating a Water Network Domain-Specific Language Parser

```python
class WaterNetworkQueryParser:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
    
    def parse_query(self, query):
        """Parse a natural language query into structured format"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an AI assistant specializing in water distribution networks. Parse the following query into a structured format.
            
            Query: {query}
            
            Extract the following elements (if present):
            1. Query type (component_info, maintenance, flow_analysis, pressure_analysis, leakage, water_quality, emergency)
            2. Component types mentioned (valve, pipe, pump, tank, reservoir, junction, zone, meter, sensor)
            3. Specific component IDs mentioned
            4. Attributes of interest (pressure, flow, age, material, status, diameter)
            5. Time periods mentioned
            6. Locations or zones mentioned
            7. Relationships of interest (connected_to, part_of, upstream, downstream)
            8. Action requested (list, show, analyze, predict, compare, optimize)
            
            Respond in JSON format:
            {{
              "query_type": string,
              "component_types": [strings],
              "component_ids": [strings],
              "attributes": [strings],
              "time_periods": string,
              "locations": [strings],
              "relationships": [strings],
              "action": string
            }}
            
            Ensure the JSON is valid and does not include any explanations outside the JSON structure.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(query=query)
        
        # Clean the response and parse JSON
        json_response = response.strip()
        
        # Handle potential formatting issues
        try:
            return json.loads(json_response)
        except json.JSONDecodeError:
            # Attempt to fix common JSON formatting errors
            json_response = re.sub(r'```json\s*', '', json_response)
            json_response = re.sub(r'\s*```', '', json_response)
            
            try:
                return json.loads(json_response)
            except json.JSONDecodeError:
                # Return a default structure if parsing fails
                return {
                    "query_type": "component_info",
                    "component_types": [],
                    "component_ids": [],
                    "attributes": [],
                    "time_periods": "",
                    "locations": [],
                    "relationships": [],
                    "action": "show"
                }
```

### 3.3 Implementing Query Translation to Cypher

```python
class QueryToCypherTranslator:
    def __init__(self):
        self.patterns = {
            "component_info": self._component_info_pattern,
            "maintenance": self._maintenance_pattern,
            "flow_analysis": self._flow_analysis_pattern,
            "pressure_analysis": self._pressure_analysis_pattern,
            "leakage": self._leakage_pattern,
            "water_quality": self._water_quality_pattern,
            "emergency": self._emergency_pattern
        }
    
    def translate(self, parsed_query):
        """Translate parsed query to Cypher"""
        query_type = parsed_query.get("query_type", "component_info")
        
        if query_type in self.patterns:
            return self.patterns[query_type](parsed_query)
        else:
            return self._component_info_pattern(parsed_query)
    
    def _component_info_pattern(self, parsed_query):
        """Generate Cypher for component information queries"""
        component_types = parsed_query.get("component_types", [])
        component_ids = parsed_query.get("component_ids", [])
        attributes = parsed_query.get("attributes", [])
        locations = parsed_query.get("locations", [])
        
        # Build component type filter
        type_filter = ""
        if component_types:
            type_filter = "WHERE " + " OR ".join([f"c:{comp_type.capitalize()}" for comp_type in component_types])
        
        # Build ID filter
        id_filter = ""
        if component_ids:
            if type_filter:
                id_filter = "AND c.id IN " + json.dumps(component_ids)
            else:
                id_filter = "WHERE c.id IN " + json.dumps(component_ids)
        
        # Build location filter
        location_filter = ""
        if locations:
            zone_match = "MATCH (c)-[:PART_OF]->(z) WHERE "
            zone_match += " OR ".join([f"z.name CONTAINS '{loc}' OR z.id CONTAINS '{loc}'" for loc in locations])
            
            if type_filter or id_filter:
                location_filter = "AND " + zone_match
            else:
                location_filter = zone_match
        
        # Build return clause
        return_clause = "RETURN c"
        if attributes:
            # If specific attributes are requested, include them
            properties = [f"c.{attr}" for attr in attributes]
            return_clause = f"RETURN c.id as id, labels(c) as types, {', '.join(properties)}"
        else:
            # Otherwise return the whole node
            return_clause = "RETURN c"
        
        # Build the full query
        cypher = f"MATCH (c) {type_filter} {id_filter} {location_filter} {return_clause} LIMIT 25"
        return cypher
    
    def _maintenance_pattern(self, parsed_query):
        """Generate Cypher for maintenance queries"""
        component_types = parsed_query.get("component_types", [])
        component_ids = parsed_query.get("component_ids", [])
        time_periods = parsed_query.get("time_periods", "")
        
        # Build component type filter
        type_filter = ""
        if component_types:
            type_filter = "WHERE " + " OR ".join([f"c:{comp_type.capitalize()}" for comp_type in component_types])
        
        # Build ID filter
        id_filter = ""
        if component_ids:
            if type_filter:
                id_filter = "AND c.id IN " + json.dumps(component_ids)
            else:
                id_filter = "WHERE c.id IN " + json.dumps(component_ids)
        
        # Build time filter
        time_filter = ""
        if time_periods:
            if "last year" in time_periods.lower():
                time_filter = "AND m.date >= date() - duration('P1Y')"
            elif "last month" in time_periods.lower():
                time_filter = "AND m.date >= date() - duration('P1M')"
            elif "last week" in time_periods.lower():
                time_filter = "AND m.date >= date() - duration('P7D')"
        
        # Build the full query
        cypher = f"""
        MATCH (c)
        {type_filter} {id_filter}
        OPTIONAL MATCH (c)-[r:HAS_MAINTENANCE]->(m)
        {time_filter}
        RETURN c.id as componentId, 
               labels(c) as componentType, 
               collect(m) as maintenanceRecords,
               count(m) as maintenanceCount
        ORDER BY maintenanceCount DESC
        LIMIT 25
        """
        
        return cypher
    
    def _flow_analysis_pattern(self, parsed_query):
        """Generate Cypher for flow analysis queries"""
        component_ids = parsed_query.get("component_ids", [])
        relationships = parsed_query.get("relationships", [])
        
        # If looking for upstream/downstream paths
        if "upstream" in relationships or "downstream" in relationships:
            if component_ids:
                component_id = component_ids[0]  # Use the first component ID
                
                if "upstream" in relationships:
                    # Find components that feed into this one
                    cypher = f"""
                    MATCH path = (c)-[:FEEDS|CONNECTED_TO*1..5]->(target {{id: '{component_id}'}})
                    RETURN path
                    LIMIT 10
                    """
                else:
                    # Find components that this one feeds into
                    cypher = f"""
                    MATCH path = (source {{id: '{component_id}'}})-[:FEEDS|CONNECTED_TO*1..5]->(c)
                    RETURN path
                    LIMIT 10
                    """
                
                return cypher
        
        # Default flow analysis query
        cypher = """
        MATCH (source:Source)-[:FEEDS|CONNECTED_TO*]->(c)
        WHERE c:Junction OR c:Tank
        WITH source, c, 
            reduce(path_length = 0, r IN relationships(shortestPath((source)-[:FEEDS|CONNECTED_TO*]->(c))) | 
                  path_length + coalesce(r.length, 0)) as distance
        RETURN source.id as sourceId, c.id as componentId, labels(c) as componentType, distance
        ORDER BY distance
        LIMIT 25
        """
        
        return cypher
    
    def _pressure_analysis_pattern(self, parsed_query):
        """Generate Cypher for pressure analysis queries"""
        locations = parsed_query.get("locations", [])
        
        if locations:
            location = locations[0]  # Use the first location
            
            cypher = f"""
            MATCH (z) 
            WHERE z.name CONTAINS '{location}' OR z.id CONTAINS '{location}'
            MATCH (j:Junction)-[:PART_OF]->(z)
            OPTIONAL MATCH (j)-[:HAS_SENSOR]->(s:Sensor)
            WHERE s.sensorType = 'Pressure'
            RETURN z.id as zoneId, 
                   z.name as zoneName,
                   j.id as junctionId, 
                   j.elevation as elevation,
                   s.id as sensorId,
                   s.lastReading as pressure
            ORDER BY pressure DESC
            LIMIT 25
            """
        else:
            # General pressure analysis
            cypher = """
            MATCH (j:Junction)
            OPTIONAL MATCH (j)-[:HAS_SENSOR]->(s:Sensor)
            WHERE s.sensorType = 'Pressure'
            WITH j, s
            ORDER BY coalesce(s.lastReading, 0) DESC
            RETURN j.id as junctionId, 
                   j.elevation as elevation,
                   s.id as sensorId,
                   s.lastReading as pressure
            LIMIT 25
            """
        
        return cypher
    
    def _leakage_pattern(self, parsed_query):
        """Generate Cypher for leakage analysis queries"""
        component_types = parsed_query.get("component_types", [])
        locations = parsed_query.get("locations", [])
        
        location_filter = ""
        if locations:
            location_filter = """
            MATCH (z)
            WHERE z.name CONTAINS '{location}' OR z.id CONTAINS '{location}'
            MATCH (c)-[:PART_OF]->(z)
            """.format(location=locations[0])
        
        type_filter = ""
        if component_types and "pipe" in component_types:
            type_filter = "WHERE c:Pipe"
        else:
            type_filter = "WHERE c:Pipe OR c:Valve"
        
        cypher = f"""
        MATCH (c)
        {location_filter}
        {type_filter}
        OPTIONAL MATCH (c)-[:HAS_MAINTENANCE]->(m)
        WHERE m.findings CONTAINS 'leak' OR m.findings CONTAINS 'break'
        WITH c, count(m) as leakCount
        OPTIONAL MATCH (c)-[:HAS_SENSOR]->(s:Sensor)
        WHERE s.sensorType = 'Flow' OR s.sensorType = 'Pressure'
        RETURN c.id as componentId, 
               labels(c) as componentType,
               c.material as material,
               c.installDate as installDate,
               duration.between(date(c.installDate), date()).years as ageYears,
               leakCount,
               collect(distinct s.id) as sensors
        ORDER BY leakCount DESC, ageYears DESC
        LIMIT 25
        """
        
        return cypher
    
    def _water_quality_pattern(self, parsed_query):
        """Generate Cypher for water quality analysis queries"""
        cypher = """
        MATCH (s:SamplingPoint)
        OPTIONAL MATCH (s)-[:HAS_READING]->(r:WaterQualityReading)
        WITH s, r
        ORDER BY r.timestamp DESC
        RETURN s.id as samplingPointId,
               s.location as location,
               s.type as type,
               collect(r)[0] as latestReading
        LIMIT 25
        """
        
        return cypher
    
    def _emergency_pattern(self, parsed_query):
        """Generate Cypher for emergency response queries"""
        component_ids = parsed_query.get("component_ids", [])
        
        if component_ids:
            component_id = component_ids[0]
            
            # Find isolation valves
            cypher = f"""
            MATCH (c {{id: '{component_id}'}})
            CALL {{
                WITH c
                MATCH path = (c)-[:CONNECTED_TO*1..5]-(v:Valve)
                RETURN v
            }}
            RETURN c.id as componentId,
                   labels(c) as componentType,
                   collect(distinct v.id) as isolationValves
            """
        else:
            # General emergency preparedness
            cypher = """
            MATCH (c)
            WHERE c:Pipe OR c:Valve OR c:Pump
            WITH c, rand() as r
            ORDER BY r
            LIMIT 5
            CALL {
                WITH c
                MATCH path = (c)-[:CONNECTED_TO*1..3]-(v:Valve)
                RETURN v
            }
            RETURN c.id as componentId,
                   labels(c) as componentType,
                   collect(distinct v.id) as isolationValves
            """
        
        return cypher
```

### 3.4 Implementing the Domain-Specific Natural Language Interface

```python
class WaterNetworkNLInterface:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.parser = WaterNetworkQueryParser()
        self.translator = QueryToCypherTranslator()
        self.llm = ChatOpenAI(temperature=0.1)
    
    def close(self):
        self.driver.close()
    
    def process_query(self, natural_language_query):
        """Process a natural language query about the water network"""
        # Parse the query
        parsed_query = self.parser.parse_query(natural_language_query)
        
        # Translate to Cypher
        cypher_query = self.translator.translate(parsed_query)
        
        # Execute Cypher query
        with self.driver.session() as session:
            result = session.run(cypher_query)
            records = [dict(record) for record in result]
        
        # Generate a natural language response
        response_prompt = PromptTemplate(
            input_variables=["original_query", "parsed_query", "cypher_query", "query_results"],
            template="""
            You are an AI assistant for a water utility company. Generate a natural language response to the user's query.
            
            Original query: {original_query}
            
            Structured interpretation: {parsed_query}
            
            Database query: {cypher_query}
            
            Query results: {query_results}
            
            Provide a clear, concise response that answers the user's question based on the query results.
            Format the information in a readable way, highlighting the most important aspects.
            If there are numerical values, include those in your response.
            If no data was found, mention that specifically.
            
            Your response:
            """
        )
        
        response_chain = LLMChain(llm=self.llm, prompt=response_prompt)
        response = response_chain.run(
            original_query=natural_language_query,
            parsed_query=json.dumps(parsed_query, indent=2),
            cypher_query=cypher_query,
            query_results=json.dumps(records[:10], indent=2, default=str)  # Limit to 10 records for brevity
        )
        
        return {
            "original_query": natural_language_query,
            "parsed_query": parsed_query,
            "cypher_query": cypher_query,
            "query_results": records,
            "response": response
        }
```

### 3.5 Using the Natural Language Interface

```python
# Initialize the interface
nl_interface = WaterNetworkNLInterface(uri, user, password)

# Process some example queries
example_queries = [
    "What is the status of valve VLV001?",
    "Show me all pipes in the High Pressure Zone installed before 2010",
    "Which components have had maintenance issues in the last year?",
    "What would happen if pipe PIP002 failed?",
    "What's the water pressure in the Downtown district?",
    "What are the isolation valves for pump PMP001?",
    "Which areas have the highest leak probability?"
]

for query in example_queries:
    print(f"\nProcessing query: '{query}'")
    result = nl_interface.process_query(query)
    print(f"Response: {result['response']}")

nl_interface.close()
```

## 4. Role-Based Interfaces for Different Users

Now let's implement role-based interfaces for different types of users:

### 4.1 Defining User Roles and Permissions

```python
# Role definitions
ROLES = {
    "operator": {
        "description": "Day-to-day operations staff",
        "permissions": [
            "view_component_status",
            "view_alarms",
            "view_basic_metrics",
            "update_valve_status",
            "view_maintenance_history"
        ]
    },
    "maintenance": {
        "description": "Maintenance personnel",
        "permissions": [
            "view_component_status",
            "view_maintenance_history",
            "update_maintenance_records",
            "view_component_details",
            "view_sensor_data",
            "view_predictive_maintenance"
        ]
    },
    "engineer": {
        "description": "Engineering staff",
        "permissions": [
            "view_component_status",
            "view_maintenance_history",
            "view_network_analysis",
            "run_simulations",
            "view_predictive_maintenance",
            "view_advanced_metrics"
        ]
    },
    "manager": {
        "description": "Management personnel",
        "permissions": [
            "view_component_status",
            "view_summary_reports",
            "view_risk_assessment",
            "view_cost_analytics",
            "view_predictive_maintenance"
        ]
    },
    "admin": {
        "description": "System administrators",
        "permissions": [
            "*"  # All permissions
        ]
    }
}
```

### 4.2 Role-Based Query Processor

```python
class RoleBasedQueryProcessor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nl_interface = WaterNetworkNLInterface(uri, user, password)
    
    def close(self):
        self.driver.close()
        self.nl_interface.close()
    
    def process_query(self, natural_language_query, user_role="operator"):
        """Process a query with role-based permissions"""
        # Get permissions for this role
        permissions = ROLES.get(user_role, {}).get("permissions", [])
        all_access = "*" in permissions
        
        # Parse the query to determine required permissions
        parsed_query = self.nl_interface.parser.parse_query(natural_language_query)
        query_type = parsed_query.get("query_type", "component_info")
        
        # Map query types to required permissions
        required_permissions = {
            "component_info": "view_component_status",
            "maintenance": "view_maintenance_history",
            "flow_analysis": "view_network_analysis",
            "pressure_analysis": "view_network_analysis",
            "leakage": "view_advanced_metrics",
            "water_quality": "view_basic_metrics",
            "emergency": "view_component_status"
        }
        
        # Check permissions
        required_permission = required_permissions.get(query_type, "view_component_status")
        
        if all_access or required_permission in permissions:
            # User has permission, process the query normally
            return self.nl_interface.process_query(natural_language_query)
        else:
            # User lacks permission
            return {
                "original_query": natural_language_query,
                "parsed_query": parsed_query,
                "cypher_query": "",
                "query_results": [],
                "response": f"Sorry, your role ({user_role}) does not have permission to access {query_type} information. Please contact an administrator if you need this access."
            }
```

### 4.3 Role-Specific Information Formatter

```python
class RoleBasedFormatter:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1)
    
    def format_for_role(self, query_results, user_role="operator"):
        """Format query results based on user role"""
        if not query_results["query_results"]:
            return query_results["response"]
        
        # Get role details
        role_desc = ROLES.get(user_role, {}).get("description", "User")
        
        # Create a formatting prompt specific to the role
        prompt_template = f"""
        You are an AI assistant for a water utility company. Format the following information for a {role_desc} ({user_role}).
        
        Original query: {{original_query}}
        
        Query results: {{query_results}}
        
        Guidelines for {user_role} role:
        """
        
        if user_role == "operator":
            prompt_template += """
            - Focus on operational status (open/closed, active/inactive)
            - Highlight any abnormal conditions
            - Keep it concise and actionable
            - Use simple, non-technical language
            - If there are maintenance needs, highlight them clearly
            """
        elif user_role == "maintenance":
            prompt_template += """
            - Focus on maintenance history and component condition
            - Highlight components needing attention
            - Include details about materials, age, and previous issues
            - Provide specific maintenance recommendations
            - Use technical language appropriate for maintenance staff
            """
        elif user_role == "engineer":
            prompt_template += """
            - Include detailed technical specifications
            - Provide depth on hydraulic performance metrics
            - Reference industry standards where relevant
            - Include numerical data and analytical insights
            - Make connections to overall system performance
            - Use proper engineering terminology
            """
        elif user_role == "manager":
            prompt_template += """
            - Focus on high-level summaries
            - Include risk assessments and priorities
            - Highlight cost implications
            - Provide context on business impact
            - Make results actionable for decision-making
            - Avoid excessive technical details
            """
        else:  # admin or other roles
            prompt_template += """
            - Provide comprehensive information
            - Include both technical details and operational context
            - Highlight any system issues or anomalies
            - Include relevant IDs and references
            """
        
        prompt_template += """
        Format your response to be clear and well-structured. Use bullet points where appropriate.
        
        Your response:
        """
        
        # Create the prompt
        prompt = PromptTemplate(
            input_variables=["original_query", "query_results"],
            template=prompt_template
        )
        
        # Run the formatting
        chain = LLMChain(llm=self.llm, prompt=prompt)
        formatted_response = chain.run(
            original_query=query_results["original_query"],
            query_results=json.dumps(query_results["query_results"][:10], indent=2, default=str)
        )
        
        return formatted_response
```

### 4.4 Integration into a Complete Role-Based System

```python
class WaterNetworkAssistant:
    def __init__(self, uri, user, password):
        """Initialize the water network assistant"""
        self.query_processor = RoleBasedQueryProcessor(uri, user, password)
        self.formatter = RoleBasedFormatter()
    
    def close(self):
        """Close connections"""
        self.query_processor.close()
    
    def ask(self, query, user_role="operator"):
        """Process a query with role-based formatting"""
        # Process the query
        results = self.query_processor.process_query(query, user_role)
        
        # Format results for the specific role
        formatted_response = self.formatter.format_for_role(results, user_role)
        
        return {
            "original_query": query,
            "role": user_role,
            "response": formatted_response,
            "raw_results": results
        }
```

### 4.5 Using the Role-Based Water Network Assistant

```python
# Initialize the assistant
assistant = WaterNetworkAssistant(uri, user, password)

# Test query for different roles
test_query = "What's the status of the water network in the North District?"

roles = ["operator", "maintenance", "engineer", "manager"]
for role in roles:
    print(f"\n--- Response for {role.upper()} role ---")
    response = assistant.ask(test_query, role)
    print(response["response"])

# Test a maintenance-specific query with different roles
maintenance_query = "Which valves need maintenance in the High Pressure Zone?"

for role in roles:
    print(f"\n--- Response for {role.upper()} role ---")
    response = assistant.ask(maintenance_query, role)
    print(response["response"])

# Close the assistant
assistant.close()
```

## 5. Practical Integration Exercise

Let's create a simple but complete exercise that integrates all the components we've built.

### Exercise: Building a Comprehensive Water Network Dashboard

In this exercise, you'll create a comprehensive dashboard that integrates predictive analytics, simulation capabilities, and natural language querying with role-based access.

#### Step 1: Set Up the Environment and Core Components

```python
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template

# Load environment variables
load_dotenv()

# Neo4j connection
uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "your_password")

# Import our components
from predictive_maintenance import PredictiveMaintenanceSystem
from water_network_simulator import Neo4jToEPANETConverter, WaterNetworkSimulator
from water_network_nl import WaterNetworkAssistant
```

#### Step 2: Create the Dashboard Backend

```python
app = Flask(__name__)

# Initialize components
predictive_system = PredictiveMaintenanceSystem(uri, user, password)
network_assistant = WaterNetworkAssistant(uri, user, password)

# Convert network for simulation (only needs to be done once)
converter = Neo4jToEPANETConverter(uri, user, password)
network_data = converter.extract_network()
inp_file = converter.create_inp_file(network_data)
converter.close()

# Initialize simulator with the created file
simulator = WaterNetworkSimulator(inp_file)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/components/high-risk', methods=['GET'])
def high_risk_components():
    # Get parameters
    failure_threshold = float(request.args.get('failure_threshold', 0.7))
    rul_threshold = int(request.args.get('rul_threshold', 5))
    
    # Get components
    components = predictive_system.get_high_risk_components(
        failure_probability_threshold=failure_threshold,
        rul_threshold=rul_threshold
    )
    
    return jsonify(components)

@app.route('/api/simulation/pipe-closure', methods=['POST'])
def simulate_pipe_closure():
    data = request.json
    pipe_id = data.get('pipe_id')
    
    if not pipe_id:
        return jsonify({"error": "No pipe ID provided"}), 400
    
    try:
        # Run baseline simulation
        baseline = simulator.run_simulation()
        
        # Run closure simulation
        closure_results = simulator.simulate_pipe_closure(pipe_id)
        
        # Calculate impact
        impact = {
            "pressure_changes": [],
            "flow_changes": []
        }
        
        for node_id, node_data in closure_results['nodes'].items():
            if node_data['type'] == 'junction' and node_id in baseline['nodes']:
                baseline_pressure = baseline['nodes'][node_id]['pressure']
                new_pressure = node_data['pressure']
                if baseline_pressure > 0:
                    pressure_change = (new_pressure - baseline_pressure) / baseline_pressure * 100
                    impact["pressure_changes"].append({
                        "node_id": node_id,
                        "baseline": baseline_pressure,
                        "new_value": new_pressure,
                        "percent_change": pressure_change
                    })
        
        for link_id, link_data in closure_results['links'].items():
            if link_data['type'] == 'pipe' and link_id in baseline['links']:
                baseline_flow = abs(baseline['links'][link_id]['flow'])
                new_flow = abs(link_data['flow'])
                if baseline_flow > 0:
                    flow_change = (new_flow - baseline_flow) / baseline_flow * 100
                    impact["flow_changes"].append({
                        "link_id": link_id,
                        "baseline": baseline_flow,
                        "new_value": new_flow,
                        "percent_change": flow_change
                    })
        
        # Sort by magnitude of change
        impact["pressure_changes"].sort(key=lambda x: abs(x["percent_change"]), reverse=True)
        impact["flow_changes"].sort(key=lambda x: abs(x["percent_change"]), reverse=True)
        
        return jsonify({
            "pipe_id": pipe_id,
            "impact": impact
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query', '')
    role = data.get('role', 'operator')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Process query
        result = network_assistant.ask(query, role)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/critical-components', methods=['GET'])
def get_critical_components():
    try:
        # Identify critical components
        critical = simulator.identify_critical_components()
        return jsonify(critical[:20])  # Return top 20
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Train predictive models first (comment out after first run)
    # predictive_system.train_models()
    
    # Update Neo4j with predictions
    predictive_system.update_neo4j_with_predictions()
    
    # Start the app
    app.run(debug=True)
```

#### Step 3: Create the Dashboard Frontend

Create a template file `templates/dashboard.html` with role-based tabs and interactive components.

#### Step 4: Test the Integrated System

1. Train the predictive models
2. Start the Flask application
3. Navigate to http://localhost:5000 to access the dashboard
4. Test different roles and query types
5. Simulate component failures and analyze the results

This exercise brings together all aspects of the water network intelligence system, from predictive analytics to simulation to natural language interfaces, with role-based access controls.

## 6. Next Steps

Now that you've implemented the Intelligence Layer (Phase 4) of your Water Network Management System, you're ready to move on to Phase 5: Deployment and Scaling. Here's what to expect in the next guide:

1. **Deployment Strategies**: How to deploy your system to production
2. **Monitoring and Alerting**: Setting up comprehensive monitoring
3. **DevOps for AI Applications**: CI/CD pipelines and automation
4. **Continuous Improvement**: Processes for ongoing enhancement

As you continue working with the Intelligence Layer, consider these advanced topics to explore:

1. **Integrating IoT Sensor Data**: Real-time data integration from field sensors
2. **Advanced Graph Analytics**: Using Neo4j's Graph Data Science for deeper network analysis
3. **Digital Twin Development**: Creating a complete digital twin of your water network
4. **Multi-Modal Interfaces**: Adding visualization and GIS integration

## 7. Resources and References

### Predictive Analytics Resources
- [Predictive Maintenance Guide](https://azure.microsoft.com/en-us/resources/predictive-maintenance-playbook/)
- [Machine Learning for Time Series Data in Python](https://www.datacamp.com/courses/machine-learning-for-time-series-data-in-python)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

### Water Network Simulation Resources
- [EPANET Documentation](https://www.epa.gov/water-research/epanet)
- [Water Network Modeling Principles](https://www.bentley.com/resources/hydraulic-and-hydrology-modeling-fundamentals/)
- [Hydraulic Simulation Best Practices](https://www.innovyze.com/en-us/resources/watermodeling)

### NLP and Domain-Specific Language Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Domain-Specific Language Processing](https://towardsdatascience.com/domain-specific-language-processing-1d362f71d475)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### Role-Based Access Control Resources
- [RBAC Principles and Best Practices](https://auth0.com/docs/authorization/rbac)
- [Neo4j Security Guide](https://neo4j.com/docs/operations-manual/current/security/)
- [UX Design for Different User Roles](https://www.interaction-design.org/literature/article/designing-for-different-user-roles)
