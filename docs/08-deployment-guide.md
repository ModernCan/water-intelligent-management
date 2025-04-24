# Deployment and Scaling Guide for Water Network Intelligence System

This guide covers Phase 5 of your Water Network Intelligence System project, focusing on deploying, monitoring, and continuously improving your system. By the end of this guide, you'll know how to take your development system into production and establish processes for long-term success.

## 1. Deployment Strategy Planning

Before deployment, it's important to design a comprehensive strategy that meets your specific requirements.

### 1.1 System Architecture Review

First, let's review the complete architecture of our Water Network Intelligence System:

```
┌─────────────────────────────────────┐
│                                     │
│  Intelligent Water Network System   │
│                                     │
└───────────────────┬─────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│               │       │               │
│  Core System  │       │  Web/API      │
│               │       │  Interface    │
│               │       │               │
└───────┬───────┘       └───────┬───────┘
        │                       │
        │     ┌─────────────────┘
        │     │
┌───────▼─────▼───┐       ┌─────────────────┐
│                 │       │                 │
│  Neo4j Graph    │◄──────►  Vector Store   │
│  Database       │       │                 │
│                 │       │                 │
└───────┬─────────┘       └─────────────────┘
        │
        │
┌───────▼───────┐       ┌─────────────────┐
│               │       │                 │
│  ML Models    │◄──────►  EPANET         │
│  Storage      │       │  Simulator      │
│               │       │                 │
└───────────────┘       └─────────────────┘
```

### 1.2 Deployment Options Analysis

Let's analyze the main deployment options:

| Deployment Option | Advantages | Disadvantages | Best For |
|-------------------|------------|--------------|----------|
| **On-Premises** | Full control, Data security, No cloud costs | Higher upfront costs, IT maintenance burden | Utilities with strict data regulations, Existing IT infrastructure |
| **Cloud-Based** | Scalability, Reduced maintenance, Managed services | Ongoing costs, Potential data privacy concerns | Small to mid-sized utilities, Limited IT resources |
| **Hybrid** | Flexibility, Critical systems on-premises, Scaling in cloud | Complexity, Integration challenges | Most water utilities with mixed requirements |
| **Edge+Cloud** | Real-time processing, Reduced bandwidth, Cloud backup | More complex architecture, Edge hardware needed | Utilities with remote infrastructure, Real-time requirements |

### 1.3 Creating a Deployment Plan

Based on a hybrid approach, here's a recommended deployment plan:

```python
# Sample deployment configuration file: deployment_config.yaml
# This would be adjusted for your specific environment

deployment:
  strategy: "hybrid"
  environments:
    production:
      neo4j:
        deployment: "on-premises"
        server: "neo4j-prod-server"
        ha_enabled: true
        replicas: 3
      ml_models:
        deployment: "on-premises"
        server: "ml-model-server"
        storage_path: "/opt/water-network/models"
      vector_store:
        deployment: "cloud"
        provider: "aws"  # or azure, gcp
        service: "managed-vector-db"
        region: "us-west-2"
      simulator:
        deployment: "on-premises"
        server: "simulation-server"
        workers: 4
      api_service:
        deployment: "cloud"
        provider: "aws"  # or azure, gcp
        service: "eks"   # kubernetes
        replicas: 3
        auto_scaling: true
      web_interface:
        deployment: "cloud"
        provider: "aws"
        service: "s3+cloudfront"
    
    staging:
      # Similar structure with reduced resources
    
    development:
      # Local development configuration
```

## 2. Container-Based Deployment

For a flexible, reproducible deployment, we'll use containerization with Docker and Kubernetes.

### 2.1 Creating Docker Images for Components

First, let's create Dockerfiles for each component:

#### Neo4j Database Container

```dockerfile
# Filename: neo4j/Dockerfile
FROM neo4j:5.5.0-enterprise

# Environment variables for configuration
ENV NEO4J_AUTH=neo4j/your_password \
    NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    NEO4J_apoc_export_file_enabled=true \
    NEO4J_apoc_import_file_enabled=true \
    NEO4J_apoc_import_file_use__neo4j__config=true \
    NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.* \
    NEO4J_dbms_memory_heap_max__size=4G

# Install APOC and GDS plugins
RUN wget -P /var/lib/neo4j/plugins https://github.com/neo4j/apoc/releases/download/5.5.0/apoc-5.5.0-core.jar \
    && wget -P /var/lib/neo4j/plugins https://github.com/neo4j/graph-data-science/releases/download/2.2.4/neo4j-graph-data-science-2.2.4.jar

# Copy initialization scripts
COPY ./init-scripts /var/lib/neo4j/init-scripts

# Copy custom configuration if needed
COPY ./conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf

EXPOSE 7474 7473 7687

# Custom entrypoint script to handle initialization
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

#### Core API Service Container

```dockerfile
# Filename: api-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including EPANET
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libgeos-dev \
    libepanet-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Setup environment variables
ENV PYTHONUNBUFFERED=1 \
    NEO4J_URI=bolt://neo4j:7687 \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=your_password \
    MODEL_PATH=/app/models \
    LOG_LEVEL=INFO

# Create volume for models
VOLUME ["/app/models"]

# Expose API port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120"]
```

#### Web Interface Container

```dockerfile
# Filename: web-interface/Dockerfile
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### 2.2 Docker Compose for Development and Testing

Create a `docker-compose.yml` file for local development:

```yaml
version: '3.8'

services:
  neo4j:
    build:
      context: ./neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/your_password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    networks:
      - water_network

  vector-store:
    build:
      context: ./vector-store
    ports:
      - "8108:8108"
    volumes:
      - vector_data:/data
    networks:
      - water_network

  ml-service:
    build:
      context: ./ml-service
    depends_on:
      - neo4j
    volumes:
      - ml_models:/app/models
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_password
    networks:
      - water_network

  simulator:
    build:
      context: ./simulator
    depends_on:
      - neo4j
    volumes:
      - simulator_data:/app/data
    networks:
      - water_network

  api-service:
    build:
      context: ./api-service
    ports:
      - "5000:5000"
    depends_on:
      - neo4j
      - vector-store
      - ml-service
      - simulator
    volumes:
      - ml_models:/app/models:ro
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_password
      - VECTOR_STORE_URL=http://vector-store:8108
    networks:
      - water_network

  web-interface:
    build:
      context: ./web-interface
    ports:
      - "80:80"
    depends_on:
      - api-service
    environment:
      - API_URL=http://api-service:5000
    networks:
      - water_network

networks:
  water_network:
    driver: bridge

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
  vector_data:
  ml_models:
  simulator_data:
```

### 2.3 Kubernetes Deployment for Production

For production deployment on Kubernetes, create the following YAML files:

#### Neo4j StatefulSet

```yaml
# Filename: kubernetes/neo4j-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: water-network
spec:
  serviceName: neo4j
  replicas: 3
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: your-registry/neo4j:latest
        ports:
        - containerPort: 7474
          name: browser
        - containerPort: 7687
          name: bolt
        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: neo4j-secrets
              key: neo4j-password
        - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
          value: "yes"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        - name: neo4j-logs
          mountPath: /logs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: neo4j-logs
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 20Gi
```

#### API Service Deployment

```yaml
# Filename: kubernetes/api-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
  namespace: water-network
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
      - name: api-service
        image: your-registry/api-service:latest
        ports:
        - containerPort: 5000
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j-service:7687"
        - name: NEO4J_USER
          value: "neo4j"
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neo4j-secrets
              key: neo4j-password
        - name: VECTOR_STORE_URL
          value: "http://vector-store-service:8108"
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: ml-models
          mountPath: /app/models
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: ml-models
        persistentVolumeClaim:
          claimName: ml-models-pvc
```

#### Horizontal Pod Autoscaler

```yaml
# Filename: kubernetes/api-service-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service-hpa
  namespace: water-network
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Ingress Configuration

```yaml
# Filename: kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: water-network-ingress
  namespace: water-network
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  tls:
  - hosts:
    - water-network.example.com
    secretName: water-network-tls
  rules:
  - host: water-network.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 5000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-interface
            port:
              number: 80
```

### 2.4 CI/CD Pipeline with GitLab CI

Create a GitLab CI configuration:

```yaml
# Filename: .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_REGISTRY: your-registry.example.com
  KUBERNETES_NAMESPACE: water-network

# Test stage
test:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest --cov=./ tests/

# Build stage for each component
build-api-service:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - cd api-service
    - docker build -t $DOCKER_REGISTRY/api-service:$CI_COMMIT_SHORT_SHA .
    - docker tag $DOCKER_REGISTRY/api-service:$CI_COMMIT_SHORT_SHA $DOCKER_REGISTRY/api-service:latest
    - docker login -u $DOCKER_USER -p $DOCKER_PASSWORD $DOCKER_REGISTRY
    - docker push $DOCKER_REGISTRY/api-service:$CI_COMMIT_SHORT_SHA
    - docker push $DOCKER_REGISTRY/api-service:latest

build-web-interface:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - cd web-interface
    - docker build -t $DOCKER_REGISTRY/web-interface:$CI_COMMIT_SHORT_SHA .
    - docker tag $DOCKER_REGISTRY/web-interface:$CI_COMMIT_SHORT_SHA $DOCKER_REGISTRY/web-interface:latest
    - docker login -u $DOCKER_USER -p $DOCKER_PASSWORD $DOCKER_REGISTRY
    - docker push $DOCKER_REGISTRY/web-interface:$CI_COMMIT_SHORT_SHA
    - docker push $DOCKER_REGISTRY/web-interface:latest

# Similar build jobs for other components

# Deploy to staging
deploy-staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
  script:
    - kubectl config use-context staging
    - kubectl -n $KUBERNETES_NAMESPACE set image deployment/api-service api-service=$DOCKER_REGISTRY/api-service:$CI_COMMIT_SHORT_SHA
    - kubectl -n $KUBERNETES_NAMESPACE set image deployment/web-interface web-interface=$DOCKER_REGISTRY/web-interface:$CI_COMMIT_SHORT_SHA
    - kubectl -n $KUBERNETES_NAMESPACE rollout status deployment/api-service
    - kubectl -n $KUBERNETES_NAMESPACE rollout status deployment/web-interface
  only:
    - develop

# Deploy to production
deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
  script:
    - kubectl config use-context production
    - kubectl -n $KUBERNETES_NAMESPACE set image deployment/api-service api-service=$DOCKER_REGISTRY/api-service:$CI_COMMIT_SHORT_SHA
    - kubectl -n $KUBERNETES_NAMESPACE set image deployment/web-interface web-interface=$DOCKER_REGISTRY/web-interface:$CI_COMMIT_SHORT_SHA
    - kubectl -n $KUBERNETES_NAMESPACE rollout status deployment/api-service
    - kubectl -n $KUBERNETES_NAMESPACE rollout status deployment/web-interface
  only:
    - main
  when: manual
```

## 3. Data Migration and System Initialization

### 3.1 Neo4j Data Migration Strategy

Create a migration script to transfer your development data to production:

```python
# Filename: scripts/migrate_neo4j_data.py
import os
import sys
import subprocess
import time
from datetime import datetime
from neo4j import GraphDatabase

def export_data(uri, user, password, output_dir):
    """Export Neo4j data to CYPHER statements"""
    print(f"Connecting to source database: {uri}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(output_dir, f"export_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        # Get node labels
        with driver.session() as session:
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            
            # Export schema (constraints and indexes)
            print("Exporting schema...")
            schema_file = os.path.join(export_dir, "schema.cypher")
            with open(schema_file, 'w') as f:
                constraints = session.run("SHOW CONSTRAINTS")
                for constraint in constraints:
                    f.write(f"{constraint['createStatement']};\n")
                
                indexes = session.run("SHOW INDEXES")
                for index in indexes:
                    f.write(f"{index['createStatement']};\n")
            
            # Export data by label
            for label in labels:
                print(f"Exporting {label} nodes...")
                label_file = os.path.join(export_dir, f"{label}.cypher")
                
                with open(label_file, 'w') as f:
                    # Export nodes
                    result = session.run(f"MATCH (n:{label}) RETURN n")
                    for record in result:
                        node = record["n"]
                        properties = {k: v for k, v in node.items()}
                        props_str = ", ".join([f"{k}: {repr(v)}" for k, v in properties.items()])
                        f.write(f"CREATE (:{label} {{{props_str}}});\n")
            
            # Export relationships
            print("Exporting relationships...")
            rel_file = os.path.join(export_dir, "relationships.cypher")
            with open(rel_file, 'w') as f:
                result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN DISTINCT type(r) as type
                """)
                
                rel_types = [record["type"] for record in result]
                
                for rel_type in rel_types:
                    print(f"Exporting {rel_type} relationships...")
                    result = session.run(f"""
                    MATCH (a)-[r:{rel_type}]->(b)
                    RETURN id(a) as source_id, id(b) as target_id, r
                    """)
                    
                    for record in result:
                        source_id = record["source_id"]
                        target_id = record["target_id"]
                        rel = record["r"]
                        properties = {k: v for k, v in rel.items()}
                        props_str = ", ".join([f"{k}: {repr(v)}" for k, v in properties.items()])
                        
                        f.write(f"""
                        MATCH (a), (b) 
                        WHERE id(a) = {source_id} AND id(b) = {target_id}
                        CREATE (a)-[:{rel_type} {{{props_str}}}]->(b);
                        """)
    
    finally:
        driver.close()
    
    print(f"Export completed to {export_dir}")
    return export_dir

def import_data(uri, user, password, import_dir):
    """Import data from CYPHER files to Neo4j"""
    print(f"Connecting to target database: {uri}")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # First import schema
            schema_file = os.path.join(import_dir, "schema.cypher")
            if os.path.exists(schema_file):
                print("Importing schema...")
                with open(schema_file, 'r') as f:
                    cypher = f.read()
                    statements = cypher.split(';')
                    for statement in statements:
                        if statement.strip():
                            try:
                                session.run(statement)
                            except Exception as e:
                                print(f"Error importing schema: {e}")
            
            # Import nodes
            for filename in os.listdir(import_dir):
                if filename.endswith(".cypher") and filename != "schema.cypher" and filename != "relationships.cypher":
                    print(f"Importing {filename}...")
                    with open(os.path.join(import_dir, filename), 'r') as f:
                        cypher = f.read()
                        statements = cypher.split(';')
                        
                        for i, statement in enumerate(statements):
                            if statement.strip():
                                try:
                                    session.run(statement)
                                    if i % 1000 == 0:
                                        print(f"Imported {i} statements...")
                                except Exception as e:
                                    print(f"Error importing node: {e}")
            
            # Import relationships
            rel_file = os.path.join(import_dir, "relationships.cypher")
            if os.path.exists(rel_file):
                print("Importing relationships...")
                with open(rel_file, 'r') as f:
                    cypher = f.read()
                    statements = cypher.split(';')
                    
                    for i, statement in enumerate(statements):
                        if statement.strip():
                            try:
                                session.run(statement)
                                if i % 1000 == 0:
                                    print(f"Imported {i} relationships...")
                            except Exception as e:
                                print(f"Error importing relationship: {e}")
    
    finally:
        driver.close()
    
    print("Import completed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python migrate_neo4j_data.py [export|import]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    # Load environment variables
    source_uri = os.getenv("SOURCE_NEO4J_URI", "bolt://localhost:7687")
    source_user = os.getenv("SOURCE_NEO4J_USER", "neo4j")
    source_password = os.getenv("SOURCE_NEO4J_PASSWORD", "password")
    
    target_uri = os.getenv("TARGET_NEO4J_URI", "bolt://neo4j-prod:7687")
    target_user = os.getenv("TARGET_NEO4J_USER", "neo4j")
    target_password = os.getenv("TARGET_NEO4J_PASSWORD", "password")
    
    data_dir = os.getenv("DATA_DIR", "./data")
    
    if action == "export":
        export_data(source_uri, source_user, source_password, data_dir)
    elif action == "import":
        if len(sys.argv) < 3:
            print("Usage: python migrate_neo4j_data.py import <export_directory>")
            sys.exit(1)
        
        import_dir = sys.argv[2]
        import_data(target_uri, target_user, target_password, import_dir)
    else:
        print(f"Unknown action: {action}")
        print("Usage: python migrate_neo4j_data.py [export|import]")
        sys.exit(1)
```

### 3.2 ML Model Deployment

Create a script to package and deploy ML models:

```python
# Filename: scripts/deploy_ml_models.py
import os
import pickle
import json
import shutil
import hashlib
from datetime import datetime

def hash_file(filename):
    """Generate a hash for the file contents"""
    h = hashlib.sha256()
    with open(filename, 'rb') as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def package_models(source_dir, output_dir, version=None):
    """Package ML models for deployment"""
    if not version:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    package_dir = os.path.join(output_dir, f"models_v{version}")
    os.makedirs(package_dir, exist_ok=True)
    
    # Find all model files
    model_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pkl') or file.endswith('.joblib'):
                model_files.append(os.path.join(root, file))
    
    # Copy models and generate metadata
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "models": []
    }
    
    for model_path in model_files:
        # Get relative path for use in metadata
        rel_path = os.path.relpath(model_path, source_dir)
        
        # Destination path
        dest_path = os.path.join(package_dir, rel_path)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(model_path, dest_path)
        
        # Add to metadata
        file_hash = hash_file(model_path)
        metadata["models"].append({
            "name": os.path.basename(model_path),
            "path": rel_path,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(model_path)
        })
    
    # Write metadata
    with open(os.path.join(package_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Packaged {len(model_files)} models to {package_dir}")
    
    # Create archive
    archive_path = f"{package_dir}.tar.gz"
    shutil.make_archive(package_dir, 'gztar', package_dir)
    
    print(f"Created archive: {archive_path}")
    return archive_path

def deploy_models(archive_path, deploy_dir, activate=True):
    """Deploy packaged models to target directory"""
    # Extract archive
    shutil.unpack_archive(archive_path, deploy_dir)
    
    # Get the extracted directory name
    extracted_dir = os.path.basename(archive_path).replace('.tar.gz', '')
    
    # Read metadata
    with open(os.path.join(deploy_dir, extracted_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Create symlink for current version if activating
    if activate:
        current_link = os.path.join(deploy_dir, "current")
        if os.path.exists(current_link):
            if os.path.islink(current_link):
                os.unlink(current_link)
            else:
                os.rename(current_link, f"{current_link}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        os.symlink(os.path.join(deploy_dir, extracted_dir), current_link)
        print(f"Activated models version {metadata['version']}")
    
    print(f"Deployed {len(metadata['models'])} models to {deploy_dir}/{extracted_dir}")
    return metadata

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deploy_ml_models.py [package|deploy]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    # Load environment variables
    source_dir = os.getenv("MODEL_SOURCE_DIR", "./models")
    output_dir = os.getenv("MODEL_OUTPUT_DIR", "./model_packages")
    deploy_dir = os.getenv("MODEL_DEPLOY_DIR", "/opt/water-network/models")
    
    if action == "package":
        version = None
        if len(sys.argv) > 2:
            version = sys.argv[2]
        
        package_models(source_dir, output_dir, version)
    
    elif action == "deploy":
        if len(sys.argv) < 3:
            print("Usage: python deploy_ml_models.py deploy <archive_path> [--no-activate]")
            sys.exit(1)
        
        archive_path = sys.argv[2]
        activate = True
        
        if len(sys.argv) > 3 and sys.argv[3] == "--no-activate":
            activate = False
        
        deploy_models(archive_path, deploy_dir, activate)
    
    else:
        print(f"Unknown action: {action}")
        print("Usage: python deploy_ml_models.py [package|deploy]")
        sys.exit(1)
```

### 3.3 Initial System Configuration

Create a system initialization script:

```python
# Filename: scripts/initialize_system.py
import os
import json
import time
import requests
from neo4j import GraphDatabase

def wait_for_service(url, max_retries=30, retry_interval=5):
    """Wait for a service to become available"""
    print(f"Waiting for service at {url}...")
    
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Service at {url} is available")
                return True
        except requests.RequestException:
            pass
        
        print(f"Attempt {i+1}/{max_retries}: Service not available. Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
    
    print(f"Service at {url} is not available after {max_retries} attempts")
    return False

def initialize_neo4j(uri, user, password, init_scripts_dir):
    """Initialize Neo4j with schema and base data"""
    print(f"Initializing Neo4j at {uri}...")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        # Execute initialization scripts in order
        script_files = sorted([f for f in os.listdir(init_scripts_dir) if f.endswith('.cypher')])
        
        for script_file in script_files:
            print(f"Executing {script_file}...")
            
            with open(os.path.join(init_scripts_dir, script_file), 'r') as f:
                script = f.read()
            
            with driver.session() as session:
                session.run(script)
            
            print(f"Completed {script_file}")
    
    finally:
        driver.close()
    
    print("Neo4j initialization completed")

def initialize_api_service(api_url, admin_user, admin_password):
    """Initialize the API service with admin user and base configuration"""
    print(f"Initializing API service at {api_url}...")
    
    # Create admin user
    response = requests.post(
        f"{api_url}/admin/users",
        json={
            "username": admin_user,
            "password": admin_password,
            "role": "admin"
        }
    )
    
    if response.status_code != 201:
        print(f"Failed to create admin user: {response.text}")
        return False
    
    # Get auth token
    response = requests.post(
        f"{api_url}/auth/login",
        json={
            "username": admin_user,
            "password": admin_password
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to authenticate: {response.text}")
        return False
    
    token = response.json().get("token")
    
    # Initialize system configuration
    response = requests.post(
        f"{api_url}/admin/config",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "system_name": "Water Network Intelligence System",
            "maintenance_check_interval": 86400,  # 24 hours in seconds
            "default_prediction_horizon": 30,     # 30 days
            "alert_notification_enabled": True,
            "auto_update_predictions": True
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to initialize system configuration: {response.text}")
        return False
    
    print("API service initialization completed")
    return True

def initialize_system():
    """Initialize the complete system"""
    # Load environment variables
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    api_url = os.getenv("API_URL", "http://localhost:5000")
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin_password")
    
    init_scripts_dir = os.getenv("INIT_SCRIPTS_DIR", "./init-scripts")
    
    # Wait for services to be available
    neo4j_browser_url = neo4j_uri.replace("bolt://", "http://").replace("7687", "7474")
    if not wait_for_service(neo4j_browser_url):
        print("Neo4j is not available. Aborting initialization.")
        return False
    
    if not wait_for_service(f"{api_url}/health"):
        print("API service is not available. Aborting initialization.")
        return False
    
    # Initialize Neo4j
    initialize_neo4j(neo4j_uri, neo4j_user, neo4j_password, init_scripts_dir)
    
    # Initialize API service
    initialize_api_service(api_url, admin_user, admin_password)
    
    print("System initialization completed successfully")
    return True

if __name__ == "__main__":
    initialize_system()
```

## 4. Monitoring and Observability

### 4.1 Setting Up Prometheus and Grafana

Create a Prometheus configuration:

```yaml
# Filename: monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'neo4j'
    metrics_path: /metrics
    static_configs:
      - targets: ['neo4j:2004']
        labels:
          instance: 'neo4j-primary'

  - job_name: 'api-service'
    metrics_path: /metrics
    static_configs:
      - targets: ['api-service:5000']

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics
```

Create Prometheus alert rules:

```yaml
# Filename: monitoring/rules/water_network_alerts.yml
groups:
- name: water_network_alerts
  rules:
  - alert: HighCPUUsage
    expr: sum(rate(process_cpu_seconds_total[5m])) by (instance, job) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage on {{ $labels.instance }}"
      description: "{{ $labels.instance }} has had high CPU usage for more than 5 minutes."

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes / process_virtual_memory_bytes > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "{{ $labels.instance }} has had high memory usage for more than 5 minutes."

  - alert: APIHighLatency
    expr: http_request_duration_seconds{quantile="0.95"} > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "API high latency"
      description: "95th percentile of HTTP request duration is above 1 second for 5 minutes."

  - alert: HighErrorRate
    expr: sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate"
      description: "Error rate is above 5% for 5 minutes."

  - alert: Neo4jHighMemoryUsage
    expr: neo4j_jvm_memory_bytes_used / neo4j_jvm_memory_bytes_max > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Neo4j high memory usage"
      description: "Neo4j is using more than 80% of allocated memory for 5 minutes."

  - alert: ModelPredictionFailure
    expr: model_prediction_errors_total > 0
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Model prediction errors"
      description: "There have been errors in model predictions for 15 minutes."
```

Set up Grafana dashboards:

```json
// Filename: monitoring/dashboards/water_network_overview.json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.6",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "sum(rate(http_requests_total[5m])) by (status_code)",
          "interval": "",
          "legendFormat": "{{status_code}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API Request Rate by Status Code",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 3,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.6",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "http_request_duration_seconds{quantile=\"0.5\"}",
          "interval": "",
          "legendFormat": "p50",
          "refId": "A"
        },
        {
          "expr": "http_request_duration_seconds{quantile=\"0.9\"}",
          "interval": "",
          "legendFormat": "p90",
          "refId": "B"
        },
        {
          "expr": "http_request_duration_seconds{quantile=\"0.95\"}",
          "interval": "",
          "legendFormat": "p95",
          "refId": "C"
        },
        {
          "expr": "http_request_duration_seconds{quantile=\"0.99\"}",
          "interval": "",
          "legendFormat": "p99",
          "refId": "D"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API Latency",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "s",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 8
      },
      "hiddenSeries": false,
      "id": 4,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.6",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "rate(process_cpu_seconds_total[5m])",
          "interval": "",
          "legendFormat": "{{instance}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "CPU Usage",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 8
      },
      "hiddenSeries": false,
      "id": 5,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.6",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "process_resident_memory_bytes",
          "interval": "",
          "legendFormat": "{{instance}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Memory Usage",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "bytes",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 16
      },
      "hiddenSeries": false,
      "id": 6,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.6",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "neo4j_jvm_memory_bytes_used{area=\"heap\"}",
          "interval": "",
          "legendFormat": "Used",
          "refId": "A"
        },
        {
          "expr": "neo4j_jvm_memory_bytes_max{area=\"heap\"}",
          "interval": "",
          "legendFormat": "Max",
          "refId": "B"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Neo4j Heap Memory",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "bytes",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    },
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 16
      },
      "hiddenSeries": false,
      "id": 7,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.6",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "model_prediction_duration_seconds_sum / model_prediction_duration_seconds_count",
          "interval": "",
          "legendFormat": "Avg Duration",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Model Prediction Duration",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "s",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "refresh": "5s",
  "schemaVersion": 26,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Water Network Overview",
  "uid": "water-network-overview",
  "version": 1
}
```

### 4.2 Implementing Application Metrics

Create a metrics module for your application:

```python
# Filename: api-service/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import threading

# HTTP request metrics
http_requests_total = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Neo4j metrics
neo4j_query_total = Counter(
    'neo4j_query_total',
    'Total Neo4j queries',
    ['query_type']
)

neo4j_query_duration_seconds = Histogram(
    'neo4j_query_duration_seconds',
    'Neo4j query duration in seconds',
    ['query_type'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Model metrics
model_prediction_total = Counter(
    'model_prediction_total',
    'Total model predictions',
    ['model_name']
)

model_prediction_errors_total = Counter(
    'model_prediction_errors_total',
    'Total model prediction errors',
    ['model_name', 'error_type']
)

model_prediction_duration_seconds = Summary(
    'model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['model_name']
)

# System metrics
active_users_gauge = Gauge(
    'active_users',
    'Number of active users'
)

active_sessions_gauge = Gauge(
    'active_sessions',
    'Number of active sessions'
)

system_components_health = Gauge(
    'system_components_health',
    'Health status of system components (1=healthy, 0=unhealthy)',
    ['component']
)

# Set initial values
system_components_health.labels(component='neo4j').set(1)
system_components_health.labels(component='api_service').set(1)
system_components_health.labels(component='ml_service').set(1)

# Decorators for measuring durations
def track_request_duration(method, endpoint):
    """Decorator to track HTTP request duration"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract status code from response
            status_code = result[1] if isinstance(result, tuple) and len(result) > 1 else 200
            
            # Update metrics
            http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            
            return result
        return wrapper
    return decorator

def track_neo4j_query(query_type):
    """Decorator to track Neo4j query execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                neo4j_query_total.labels(query_type=query_type).inc()
                return result
            finally:
                duration = time.time() - start_time
                neo4j_query_duration_seconds.labels(query_type=query_type).observe(duration)
        return wrapper
    return decorator

def track_model_prediction(model_name):
    """Decorator to track model prediction execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                model_prediction_total.labels(model_name=model_name).inc()
                return result
            except Exception as e:
                error_type = type(e).__name__
                model_prediction_errors_total.labels(model_name=model_name, error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                model_prediction_duration_seconds.labels(model_name=model_name).observe(duration)
        return wrapper
    return decorator

# Background health check
def component_health_check(interval=60):
    """Run periodic health checks on system components"""
    def check_health():
        while True:
            try:
                # Check Neo4j
                # ... neo4j health check code ...
                system_components_health.labels(component='neo4j').set(1)
            except Exception:
                system_components_health.labels(component='neo4j').set(0)
            
            try:
                # Check ML service
                # ... ml service health check code ...
                system_components_health.labels(component='ml_service').set(1)
            except Exception:
                system_components_health.labels(component='ml_service').set(0)
            
            # Sleep for the interval
            time.sleep(interval)
    
    # Start health check in background thread
    thread = threading.Thread(target=check_health, daemon=True)
    thread.start()
```

Integrate metrics into your Flask application:

```python
# Filename: api-service/app.py
from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import time

from metrics import track_request_duration, component_health_check

app = Flask(__name__)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Start background health checks
component_health_check(interval=60)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/api/components', methods=['GET'])
@track_request_duration(method='GET', endpoint='/api/components')
def get_components():
    """Get components endpoint with metrics tracking"""
    # ... your code here ...
    return jsonify({"components": []})

# ... rest of your application ...
```

### 4.3 Setting Up Centralized Logging

Create a Fluent Bit configuration for log collection:

```ini
# Filename: monitoring/fluent-bit.conf
[SERVICE]
    Flush          5
    Log_Level      info
    Daemon         off
    Parsers_File   parsers.conf

[INPUT]
    Name              tail
    Path              /var/log/containers/*.log
    Parser            docker
    Tag               kube.*
    Refresh_Interval  10
    Mem_Buf_Limit     5MB
    Skip_Long_Lines   On

[FILTER]
    Name                kubernetes
    Match               kube.*
    Kube_URL            https://kubernetes.default.svc:443
    Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
    Merge_Log           On
    K8S-Logging.Parser  On
    K8S-Logging.Exclude Off

[FILTER]
    Name            grep
    Match           kube.var.log.containers.api-service*
    Regex           $kubernetes['labels']['app'] api-service

[FILTER]
    Name            grep
    Match           kube.var.log.containers.neo4j*
    Regex           $kubernetes['labels']['app'] neo4j

[OUTPUT]
    Name            es
    Match           *
    Host            elasticsearch
    Port            9200
    Index           water-network
    Type            _doc
    Logstash_Format On
    Logstash_Prefix water-network
    Time_Key        @timestamp
    Replace_Dots    On
    Retry_Limit     False
```

Configure your application for structured logging:

```python
# Filename: api-service/logger.py
import json
import logging
import sys
import traceback
from datetime import datetime
import socket
import os

class StructuredLogFormatter(logging.Formatter):
    """Formatter for structured JSON logs"""
    
    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()
        self.service_name = os.environ.get("SERVICE_NAME", "api-service")
        self.environment = os.environ.get("ENVIRONMENT", "development")
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "hostname": self.hostname,
            "environment": self.environment,
            "thread": record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add any extra fields
        if hasattr(record, "data"):
            log_record["data"] = record.data
        
        return json.dumps(log_record)

def setup_logging(log_level=logging.INFO):
    """Setup structured JSON logging"""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    
    # Create JSON handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredLogFormatter())
    root_logger.addHandler(handler)
    
    # Suppress some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    
    return root_logger

def log_with_data(logger, level, message, data=None):
    """Log a message with structured data"""
    if data is None:
        data = {}
    
    # Create a record with the extra data field
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None
    )
    record.data = data
    
    # Process the record through the logger
    logger.handle(record)
```

## 5. Backup and Disaster Recovery

### 5.1 Neo4j Backup Strategy

Create a Neo4j backup script:

```python
# Filename: scripts/neo4j_backup.py
import os
import subprocess
import time
from datetime import datetime
import boto3
import argparse

def run_neo4j_backup(neo4j_home, backup_dir, database_name="neo4j"):
    """Run Neo4j backup command"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{database_name}_{timestamp}")
    
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    
    # Run backup command
    cmd = [
        f"{os.path.join(neo4j_home, 'bin', 'neo4j-admin')}",
        "dump",
        "--database", database_name,
        "--to", f"{backup_path}.dump"
    ]
    
    print(f"Running backup command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Backup failed with error: {result.stderr}")
        return None
    
    print(f"Backup completed successfully: {backup_path}.dump")
    return f"{backup_path}.dump"

def upload_to_s3(backup_file, bucket_name, folder_prefix=""):
    """Upload backup file to S3"""
    if not os.path.exists(backup_file):
        print(f"Backup file does not exist: {backup_file}")
        return False
    
    # Create S3 client
    s3 = boto3.client('s3')
    
    # Generate S3 key
    filename = os.path.basename(backup_file)
    s3_key = f"{folder_prefix}/{filename}" if folder_prefix else filename
    
    print(f"Uploading {backup_file} to s3://{bucket_name}/{s3_key}")
    
    try:
        # Upload file to S3
        s3.upload_file(backup_file, bucket_name, s3_key)
        print(f"Upload completed successfully")
        return True
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return False

def cleanup_old_backups(backup_dir, keep_days=7):
    """Delete backups older than specified days"""
    now = time.time()
    keep_seconds = keep_days * 86400
    
    for filename in os.listdir(backup_dir):
        file_path = os.path.join(backup_dir, filename)
        
        # Check if it's a backup file
        if os.path.isfile(file_path) and filename.endswith('.dump'):
            # Check file age
            file_age = now - os.path.getmtime(file_path)
            
            if file_age > keep_seconds:
                print(f"Removing old backup: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neo4j Backup Script")
    parser.add_argument("--neo4j-home", default="/var/lib/neo4j", help="Neo4j home directory")
    parser.add_argument("--backup-dir", default="/var/lib/neo4j/backups", help="Backup directory")
    parser.add_argument("--database", default="neo4j", help="Database name")
    parser.add_argument("--s3-bucket", help="S3 bucket for backup upload")
    parser.add_argument("--s3-prefix", default="neo4j-backups", help="S3 folder prefix")
    parser.add_argument("--keep-days", type=int, default=7, help="Days to keep local backups")
    
    args = parser.parse_args()
    
    # Run backup
    backup_file = run_neo4j_backup(args.neo4j_home, args.backup_dir, args.database)
    
    if backup_file:
        # Upload to S3 if bucket specified
        if args.s3_bucket:
            upload_to_s3(backup_file, args.s3_bucket, args.s3_prefix)
        
        # Clean up old backups
        cleanup_old_backups(args.backup_dir, args.keep_days)
```

### 5.2 Model Snapshots and Version Control

Create a model versioning and backup system:

```python
# Filename: scripts/model_snapshot.py
import os
import json
import shutil
import hashlib
import datetime
import tarfile
import boto3
import pickle
import joblib
import argparse

def compute_model_hash(model_file):
    """Generate hash for model file"""
    h = hashlib.sha256()
    with open(model_file, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def create_model_snapshot(model_dir, snapshot_dir, metadata=None):
    """Create a snapshot of all models"""
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(snapshot_dir, f"model_snapshot_{timestamp}")
    
    # Create directory
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Find all model files
    model_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pkl') or file.endswith('.joblib'):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print(f"No model files found in {model_dir}")
        return None
    
    # Copy model files and collect metadata
    model_metadata = {
        "snapshot_time": timestamp,
        "models": []
    }
    
    for src_path in model_files:
        # Get relative path for destination
        rel_path = os.path.relpath(src_path, model_dir)
        dst_path = os.path.join(snapshot_path, rel_path)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dst_path)
        
        # Compute hash
        file_hash = compute_model_hash(src_path)
        
        # Get model-specific metadata if available
        model_info = {
            "filename": rel_path,
            "hash": file_hash,
            "size_bytes": os.path.getsize(src_path)
        }
        
        # Try to load model to extract model-specific info
        try:
            if src_path.endswith('.pkl'):
                with open(src_path, 'rb') as f:
                    model = pickle.load(f)
            elif src_path.endswith('.joblib'):
                model = joblib.load(src_path)
            
            # Extract scikit-learn model info if available
            if hasattr(model, 'get_params'):
                model_info["parameters"] = model.get_params()
            
            if hasattr(model, 'feature_importances_'):
                model_info["feature_importances"] = model.feature_importances_.tolist()
            
            if hasattr(model, 'classes_'):
                model_info["classes"] = model.classes_.tolist()
        except Exception as e:
            print(f"Warning: Could not extract model details from {rel_path}: {e}")
        
        model_metadata["models"].append(model_info)
    
    # Add custom metadata if provided
    if metadata:
        model_metadata.update(metadata)
    
    # Write metadata file
    with open(os.path.join(snapshot_path, "metadata.json"), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Create archive
    archive_path = f"{snapshot_path}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(snapshot_path, arcname=os.path.basename(snapshot_path))
    
    # Remove the directory, keeping only the archive
    shutil.rmtree(snapshot_path)
    
    print(f"Created model snapshot: {archive_path}")
    return archive_path

def upload_to_s3(snapshot_file, bucket_name, folder_prefix=""):
    """Upload snapshot to S3"""
    if not os.path.exists(snapshot_file):
        print(f"Snapshot file does not exist: {snapshot_file}")
        return False
    
    # Create S3 client
    s3 = boto3.client('s3')
    
    # Generate S3 key
    filename = os.path.basename(snapshot_file)
    s3_key = f"{folder_prefix}/{filename}" if folder_prefix else filename
    
    print(f"Uploading {snapshot_file} to s3://{bucket_name}/{s3_key}")
    
    try:
        # Upload file to S3
        s3.upload_file(snapshot_file, bucket_name, s3_key)
        print(f"Upload completed successfully")
        return True
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return False

def restore_model_snapshot(snapshot_file, restore_dir):
    """Restore models from a snapshot"""
    # Create restore directory
    os.makedirs(restore_dir, exist_ok=True)
    
    # Extract archive
    with tarfile.open(snapshot_file, "r:gz") as tar:
        tar.extractall(path=restore_dir)
    
    # Get the extracted directory name
    snapshot_dirname = os.path.basename(snapshot_file).replace('.tar.gz', '')
    snapshot_path = os.path.join(restore_dir, snapshot_dirname)
    
    # Read metadata
    with open(os.path.join(snapshot_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    print(f"Restored model snapshot from {snapshot_file} to {restore_dir}")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Snapshot Utility")
    parser.add_argument("action", choices=["create", "restore"], help="Action to perform")
    parser.add_argument("--model-dir", default="./models", help="Model directory")
    parser.add_argument("--snapshot-dir", default="./snapshots", help="Snapshot directory")
    parser.add_argument("--s3-bucket", help="S3 bucket for snapshot upload")
    parser.add_argument("--s3-prefix", default="model-snapshots", help="S3 folder prefix")
    parser.add_argument("--snapshot-file", help="Snapshot file to restore")
    parser.add_argument("--restore-dir", default="./restored_models", help="Directory to restore to")
    
    args = parser.parse_args()
    
    if args.action == "create":
        # Create metadata with custom information
        metadata = {
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "description": "Scheduled model snapshot",
            "version": os.environ.get("MODEL_VERSION", "1.0.0")
        }
        
        # Create snapshot
        snapshot_file = create_model_snapshot(args.model_dir, args.snapshot_dir, metadata)
        
        # Upload to S3 if specified
        if snapshot_file and args.s3_bucket:
            upload_to_s3(snapshot_file, args.s3_bucket, args.s3_prefix)
    
    elif args.action == "restore":
        if not args.snapshot_file:
            print("Error: --snapshot-file is required for restore action")
            parser.print_help()
            exit(1)
        
        # Restore snapshot
        restore_model_snapshot(args.snapshot_file, args.restore_dir)
```

### 5.3 Disaster Recovery Plan

Create a comprehensive disaster recovery plan document:

```markdown
# Water Network Intelligence System: Disaster Recovery Plan

## 1. Introduction

This Disaster Recovery (DR) Plan outlines procedures for recovering the Water Network Intelligence System in case of a catastrophic event. It covers system components, recovery procedures, and responsibilities.

## 2. System Components

The Water Network Intelligence System consists of:

- **Neo4j Database**: Core graph database storing network topology and operational data
- **Vector Store**: Contains embeddings for semantic search
- **ML Models**: Predictive maintenance and failure analysis models
- **API Service**: Backend services providing system functionality
- **Web Interface**: User-facing application

## 3. Recovery Time Objectives (RTO)

| Component      | RTO (hours) | Priority |
|----------------|-------------|----------|
| Neo4j Database | 4           | Critical |
| API Service    | 6           | High     |
| Web Interface  | 8           | Medium   |
| ML Models      | 12          | Medium   |
| Vector Store   | 12          | Medium   |

## 4. Backup Strategy

### 4.1 Neo4j Database
- Daily full backups (00:00 UTC)
- Backups stored locally and in S3 bucket
- Retention: 7 days local, 30 days in S3

### 4.2 ML Models
- Snapshots created after each training/update
- Complete snapshot weekly (Sunday 00:00 UTC)
- Snapshots stored in S3 with version metadata

### 4.3 Vector Store
- Daily snapshots (00:00 UTC)
- Snapshots stored in S3

### 4.4 Configuration and Code
- All configuration in Git repository
- Infrastructure as Code (Terraform) for all cloud resources
- CI/CD pipelines capture build artifacts

## 5. Recovery Procedures

### 5.1 Neo4j Database Recovery

1. **Identify most recent valid backup**
   - Check S3 bucket: `s3://water-network-backups/neo4j-backups/`
   - Verify integrity with metadata

2. **Provision Neo4j instance**
   - If on Kubernetes: `kubectl apply -f kubernetes/neo4j-statefulset.yaml`
   - If on-premises: Provision server with Neo4j installed

3. **Download backup**
   ```bash
   aws s3 cp s3://water-network-backups/neo4j-backups/neo4j_YYYYMMDD_HHMMSS.dump /tmp/
   ```

4. **Restore database**
   ```bash
   neo4j-admin database load --from=/tmp/neo4j_YYYYMMDD_HHMMSS.dump --database=neo4j
   ```

5. **Verify restoration**
   - Connect to Neo4j Browser
   - Run basic connectivity queries
   - Check node and relationship counts

### 5.2 ML Model Recovery

1. **Identify most recent model snapshot**
   - Check S3 bucket: `s3://water-network-backups/model-snapshots/`

2. **Provision model server**
   - Deploy container or VM with required dependencies

3. **Download and restore models**
   ```bash
   python scripts/model_snapshot.py restore --snapshot-file=model_snapshot_YYYYMMDD_HHMMSS.tar.gz --restore-dir=/opt/models
   ```

4. **Verify model functionality**
   - Run basic inference tests
   - Check model metadata

### 5.3 Full System Recovery

1. **Restore infrastructure**
   - Apply Terraform configuration: `terraform apply`

2. **Restore Neo4j database** (See 5.1)

3. **Restore ML models** (See 5.2)

4. **Deploy API and Web services**
   - Apply Kubernetes configurations
   - Verify service connectivity

5. **Run system tests**
   - API health checks
   - Basic functionality tests
   - Integration tests

## 6. Testing Schedule

The DR plan should be tested according to the following schedule:

- **Quarterly**: Database restoration test
- **Semi-annually**: Full system recovery test
- **Annually**: Complete DR simulation (including alternate site)

## 7. Roles and Responsibilities

| Role               | Primary Responsibility                                 | Backup Person |
|--------------------|-------------------------------------------------------|---------------|
| System Admin       | Infrastructure restoration, Neo4j recovery             | DevOps Lead   |
| Data Engineer      | Data validation, ML model restoration                  | ML Engineer   |
| DevOps Lead        | Orchestration, monitoring restoration                  | System Admin  |
| Project Manager    | Communication, coordination with stakeholders          | Team Lead     |

## 8. Communication Plan

In the event of a disaster:

1. Project Manager notifies all stakeholders via email and phone
2. Regular status updates (hourly during critical recovery)
3. Final recovery report and post-incident analysis

## 9. Documentation and Updates

This DR plan should be reviewed and updated:
- After each DR test
- When significant system changes occur
- At minimum, quarterly review

Last updated: [DATE]
```

## 6. Continuous Improvement Processes

### 6.1 Feedback Collection System

Implement a feedback API for your system:

```python
# Filename: api-service/feedback.py
from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid
from neo4j import GraphDatabase
from functools import wraps
import json
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Create Blueprint
feedback_bp = Blueprint('feedback', __name__)

# Neo4j driver
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
        
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        # Verify token (implement your token verification logic)
        # ...
        
        return f(*args, **kwargs)
    return decorated

@feedback_bp.route('/api/feedback', methods=['POST'])
@token_required
def submit_feedback():
    """Submit user feedback"""
    data = request.json
    
    # Validate required fields
    required_fields = ['source', 'category', 'content']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Generate feedback ID
    feedback_id = str(uuid.uuid4())
    
    # Get user ID from token or request
    user_id = get_user_id_from_request()
    
    # Create feedback record
    feedback = {
        'id': feedback_id,
        'user_id': user_id,
        'source': data['source'],
        'category': data['category'],
        'content': data['content'],
        'rating': data.get('rating'),
        'component_id': data.get('component_id'),
        'component_type': data.get('component_type'),
        'submitted_at': datetime.utcnow().isoformat() + 'Z',
        'status': 'new'
    }
    
    # Store in Neo4j
    with driver.session() as session:
        try:
            session.run("""
            CREATE (f:Feedback {
                id: $id,
                user_id: $user_id,
                source: $source,
                category: $category,
                content: $content,
                rating: $rating,
                component_id: $component_id,
                component_type: $component_type,
                submitted_at: $submitted_at,
                status: $status
            })
            """, feedback)
            
            # If feedback relates to a specific component, create relationship
            if data.get('component_id') and data.get('component_type'):
                session.run("""
                MATCH (f:Feedback {id: $feedback_id})
                MATCH (c {id: $component_id})
                WHERE $component_type in labels(c)
                CREATE (f)-[:ABOUT]->(c)
                """, {
                    'feedback_id': feedback_id,
                    'component_id': data.get('component_id'),
                    'component_type': data.get('component_type')
                })
            
            # Log feedback submission
            logger.info(f"Feedback submitted: {feedback_id}", extra={
                'feedback_id': feedback_id,
                'user_id': user_id,
                'category': data['category']
            })
            
            return jsonify({'id': feedback_id, 'status': 'submitted'}), 201
        
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}", extra={
                'feedback_data': json.dumps(feedback)
            })
            return jsonify({'error': 'Failed to store feedback'}), 500

@feedback_bp.route('/api/feedback', methods=['GET'])
@token_required
def get_feedback():
    """Get feedback records with filtering"""
    # Get query parameters
    category = request.args.get('category')
    status = request.args.get('status')
    component_id = request.args.get('component_id')
    limit = int(request.args.get('limit', 50))
    
    # Build query conditions
    conditions = []
    parameters = {}
    
    if category:
        conditions.append("f.category = $category")
        parameters['category'] = category
    
    if status:
        conditions.append("f.status = $status")
        parameters['status'] = status
    
    if component_id:
        conditions.append("f.component_id = $component_id")
        parameters['component_id'] = component_id
    
    # Build query
    query = "MATCH (f:Feedback)"
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " RETURN f ORDER BY f.submitted_at DESC LIMIT $limit"
    parameters['limit'] = limit
    
    # Execute query
    with driver.session() as session:
        try:
            result = session.run(query, parameters)
            feedback_list = [dict(record['f']) for record in result]
            
            return jsonify({'feedback': feedback_list})
        
        except Exception as e:
            logger.error(f"Error retrieving feedback: {str(e)}")
            return jsonify({'error': 'Failed to retrieve feedback'}), 500

@feedback_bp.route('/api/feedback/<feedback_id>', methods=['PUT'])
@token_required
def update_feedback_status(feedback_id):
    """Update feedback status"""
    data = request.json
    
    if 'status' not in data:
        return jsonify({'error': 'Status is required'}), 400
    
    # Only allow specific status values
    valid_statuses = ['new', 'in_progress', 'resolved', 'closed']
    if data['status'] not in valid_statuses:
        return jsonify({'error': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'}), 400
    
    # Update in Neo4j
    with driver.session() as session:
        try:
            result = session.run("""
            MATCH (f:Feedback {id: $feedback_id})
            SET f.status = $status,
                f.updated_at = $updated_at,
                f.resolution = $resolution
            RETURN f
            """, {
                'feedback_id': feedback_id,
                'status': data['status'],
                'updated_at': datetime.utcnow().isoformat() + 'Z',
                'resolution': data.get('resolution')
            })
            
            record = result.single()
            if not record:
                return jsonify({'error': 'Feedback not found'}), 404
            
            return jsonify(dict(record['f']))
        
        except Exception as e:
            logger.error(f"Error updating feedback: {str(e)}")
            return jsonify({'error': 'Failed to update feedback'}), 500

def get_user_id_from_request():
    """Extract user ID from request authentication"""
    # Implement your authentication logic
    return "anonymous"  # Replace with actual user ID

# Register blueprint with app
def register_feedback_routes(app):
    app.register_blueprint(feedback_bp)
```

### 6.2 Automated Testing Framework

Create a comprehensive testing framework:

```python
# Filename: tests/conftest.py
import pytest
import os
import json
from neo4j import GraphDatabase
import tempfile
import subprocess
import time

# Test configuration
TEST_NEO4J_URI = os.environ.get("TEST_NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.environ.get("TEST_NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.environ.get("TEST_NEO4J_PASSWORD", "test_password")
TEST_API_URL = os.environ.get("TEST_API_URL", "http://localhost:5001")

@pytest.fixture(scope="session")
def neo4j_container():
    """Start Neo4j container for testing"""
    # Skip if using external Neo4j instance
    if os.environ.get("USE_EXTERNAL_NEO4J"):
        yield
        return
    
    # Start Neo4j container
    container_name = "neo4j-test"
    subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "7688:7687",
        "-p", "7475:7474",
        "-e", f"NEO4J_AUTH={TEST_NEO4J_USER}/{TEST_NEO4J_PASSWORD}",
        "neo4j:5.5.0"
    ], check=True)
    
    # Wait for Neo4j to start
    wait_for_neo4j(TEST_NEO4J_URI, TEST_NEO4J_USER, TEST_NEO4J_PASSWORD)
    
    yield
    
    # Stop and remove container
    subprocess.run(["docker", "stop", container_name], check=True)
    subprocess.run(["docker", "rm", container_name], check=True)

def wait_for_neo4j(uri, user, password, max_attempts=30):
    """Wait for Neo4j to become available"""
    print(f"Waiting for Neo4j at {uri}...")
    for attempt in range(max_attempts):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            print("Neo4j is ready")
            return
        except Exception:
            print(f"Attempt {attempt+1}/{max_attempts}: Neo4j not ready yet")
            time.sleep(1)
    
    raise Exception(f"Neo4j did not become available after {max_attempts} attempts")

@pytest.fixture(scope="session")
def neo4j_driver(neo4j_container):
    """Create Neo4j driver for tests"""
    driver = GraphDatabase.driver(TEST_NEO4J_URI, auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD))
    yield driver
    driver.close()

@pytest.fixture(scope="session")
def neo4j_test_data(neo4j_driver):
    """Load test data into Neo4j"""
    with neo4j_driver.session() as session:
        # Clear database
        session.run("MATCH (n) DETACH DELETE n")
        
        # Load test data from file
        with open("tests/data/test_graph.cypher", "r") as f:
            cypher = f.read()
            statements = cypher.split(";")
            for statement in statements:
                if statement.strip():
                    session.run(statement)
    
    yield

@pytest.fixture(scope="session")
def api_client(neo4j_test_data):
    """Start API service for testing"""
    # Skip if using external API
    if os.environ.get("USE_EXTERNAL_API"):
        yield TEST_API_URL
        return
    
    # Create temporary config file
    config = {
        "NEO4J_URI": TEST_NEO4J_URI,
        "NEO4J_USER": TEST_NEO4J_USER,
        "NEO4J_PASSWORD": TEST_NEO4J_PASSWORD,
        "PORT": 5001,
        "TESTING": True
    }
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    # Start API service
    process = subprocess.Popen([
        "python", "app.py",
        "--config", config_path
    ])
    
    # Wait for API to start
    wait_for_api(TEST_API_URL)
    
    yield TEST_API_URL
    
    # Stop API service
    process.terminate()
    process.wait()
    
    # Remove config file
    os.unlink(config_path)

def wait_for_api(url, max_attempts=30):
    """Wait for API to become available"""
    import requests
    
    print(f"Waiting for API at {url}...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("API is ready")
                return
        except requests.RequestException:
            pass
        
        print(f"Attempt {attempt+1}/{max_attempts}: API not ready yet")
        time.sleep(1)
    
    raise Exception(f"API did not become available after {max_attempts} attempts")

@pytest.fixture
def auth_token(api_client):
    """Get authentication token for API calls"""
    import requests
    
    response = requests.post(f"{api_client}/auth/login", json={
        "username": "test_user",
        "password": "test_password"
    })
    
    assert response.status_code == 200
    return response.json()["token"]
```

Create API test module:

```python
# Filename: tests/test_api.py
import pytest
import requests

def test_health_check(api_client):
    """Test health check endpoint"""
    response = requests.get(f"{api_client}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_component_list(api_client, auth_token):
    """Test retrieving component list"""
    response = requests.get(
        f"{api_client}/api/components",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert isinstance(data["components"], list)
    assert len(data["components"]) > 0

def test_component_detail(api_client, auth_token):
    """Test retrieving component details"""
    # First get a component ID
    list_response = requests.get(
        f"{api_client}/api/components",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    components = list_response.json()["components"]
    component_id = components[0]["id"]
    
    # Get component details
    response = requests.get(
        f"{api_client}/api/components/{component_id}",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["id"] == component_id

def test_prediction_endpoint(api_client, auth_token):
    """Test component prediction endpoint"""
    # Get a pipe component ID
    list_response = requests.get(
        f"{api_client}/api/components?type=Pipe",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    components = list_response.json()["components"]
    pipe_id = components[0]["id"]
    
    # Get failure prediction
    response = requests.get(
        f"{api_client}/api/predictions/failure?component_id={pipe_id}",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "component_id" in data
    assert data["component_id"] == pipe_id
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
```

### 6.3 Performance Benchmarking Tool

Create a performance benchmarking tool:

```python
# Filename: scripts/benchmark.py
import requests
import time
import json
import statistics
import argparse
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

def perform_request(url, method="GET", headers=None, json_data=None, params=None):
    """Perform HTTP request and measure time"""
    start_time = time.time()
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data, timeout=30)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=json_data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        duration = time.time() - start_time
        status_code = response.status_code
        
        try:
            response_size = len(response.content)
        except:
            response_size = 0
        
        return {
            "duration": duration,
            "status_code": status_code,
            "response_size": response_size,
            "success": 200 <= status_code < 300
        }
    
    except Exception as e:
        duration = time.time() - start_time
        return {
            "duration": duration,
            "status_code": 0,
            "response_size": 0,
            "success": False,
            "error": str(e)
        }

def run_benchmark(endpoint_config, base_url, auth_token=None, concurrency=1, iterations=100):
    """Run benchmark for an endpoint"""
    url = f"{base_url}{endpoint_config['endpoint']}"
    method = endpoint_config.get("method", "GET")
    params = endpoint_config.get("params")
    headers = {}
    
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    if endpoint_config.get("headers"):
        headers.update(endpoint_config["headers"])
    
    json_data = endpoint_config.get("json_data")
    
    print(f"Benchmarking {method} {url}")
    print(f"Concurrency: {concurrency}, Iterations: {iterations}")
    
    results = []
    errors = []
    
    # Define the task for each request
    def task():
        result = perform_request(url, method, headers, json_data, params)
        results.append(result)
        if not result["success"]:
            errors.append(result)
    
    # Run with appropriate concurrency
    start_time = time.time()
    
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for _ in range(iterations):
                executor.submit(task)
    else:
        for _ in range(iterations):
            task()
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if results:
        durations = [r["duration"] for r in results]
        successful = [r for r in results if r["success"]]
        success_rate = len(successful) / len(results) if results else 0
        
        response_sizes = [r["response_size"] for r in successful]
        total_bytes = sum(response_sizes)
        
        stats = {
            "endpoint": endpoint_config["endpoint"],
            "method": method,
            "iterations": iterations,
            "concurrency": concurrency,
            "total_time": total_time,
            "requests_per_second": iterations / total_time,
            "success_rate": success_rate,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "mean_duration": statistics.mean(durations) if durations else 0,
            "median_duration": statistics.median(durations) if durations else 0,
            "p95_duration": np.percentile(durations, 95) if durations else 0,
            "p99_duration": np.percentile(durations, 99) if durations else 0,
            "total_bytes": total_bytes,
            "mean_response_size": statistics.mean(response_sizes) if response_sizes else 0
        }
        
        if errors:
            stats["error_count"] = len(errors)
            stats["error_rate"] = len(errors) / len(results)
            stats["first_error"] = errors[0].get("error", "Unknown error")
        
        return stats
    
    return {
        "endpoint": endpoint_config["endpoint"],
        "method": method,
        "iterations": iterations,
        "concurrency": concurrency,
        "total_time": total_time,
        "success_rate": 0,
        "error": "No results returned"
    }

def login(base_url, username, password):
    """Authenticate and get token"""
    try:
        response = requests.post(
            f"{base_url}/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            return response.json().get("token")
        
        print(f"Authentication failed: {response.status_code}")
        return None
    
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return None

def generate_report(results, output_dir):
    """Generate benchmark report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results as JSON
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    report = ["# API Performance Benchmark Report\n"]
    report.append(f"Benchmark run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add summary table
    report.append("## Summary\n")
    report.append("| Endpoint | Method | RPS | Success Rate | Median (ms) | P95 (ms) | P99 (ms) |\n")
    report.append("|----------|--------|-----|-------------|-------------|----------|----------|\n")
    
    for result in results:
        endpoint = result["endpoint"]
        method = result["method"]
        rps = f"{result['requests_per_second']:.2f}"
        success_rate = f"{result['success_rate'] * 100:.1f}%"
        median_ms = f"{result['median_duration'] * 1000:.1f}"
        p95_ms = f"{result['p95_duration'] * 1000:.1f}"
        p99_ms = f"{result['p99_duration'] * 1000:.1f}"
        
        report.append(f"| {endpoint} | {method} | {rps} | {success_rate} | {median_ms} | {p95_ms} | {p99_ms} |\n")
    
    report.append("\n## Detailed Results\n")
    
    # Add detailed results for each endpoint
    for result in results:
        endpoint = result["endpoint"]
        method = result["method"]
        
        report.append(f"### {method} {endpoint}\n")
        report.append(f"- Requests: {result['iterations']} ({result['concurrency']} concurrent)\n")
        report.append(f"- Total Time: {result['total_time']:.2f} seconds\n")
        report.append(f"- Requests Per Second: {result['requests_per_second']:.2f}\n")
        report.append(f"- Success Rate: {result['success_rate'] * 100:.1f}%\n")
        report.append("\n**Response Time (seconds)**\n")
        report.append(f"- Min: {result['min_duration']:.4f}\n")
        report.append(f"- Max: {result['max_duration']:.4f}\n")
        report.append(f"- Mean: {result['mean_duration']:.4f}\n")
        report.append(f"- Median: {result['median_duration']:.4f}\n")
        report.append(f"- 95th Percentile: {result['p95_duration']:.4f}\n")
        report.append(f"- 99th Percentile: {result['p99_duration']:.4f}\n")
        
        if "error_count" in result:
            report.append(f"\n**Errors**\n")
            report.append(f"- Error Count: {result['error_count']}\n")
            report.append(f"- Error Rate: {result['error_rate'] * 100:.1f}%\n")
            report.append(f"- Sample Error: {result['first_error']}\n")
        
        report.append("\n")
    
    # Write report to file
    with open(os.path.join(output_dir, "benchmark_report.md"), "w") as f:
        f.writelines(report)
    
    # Generate charts
    plt.figure(figsize=(10, 6))
    endpoints = [r["endpoint"] for r in results]
    rps = [r["requests_per_second"] for r in results]
    
    plt.bar(endpoints, rps)
    plt.xlabel("Endpoint")
    plt.ylabel("Requests Per Second")
    plt.title("API Performance Benchmark")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rps_chart.png"))
    
    # Response time chart
    plt.figure(figsize=(10, 6))
    median_times = [r["median_duration"] * 1000 for r in results]  # Convert to ms
    p95_times = [r["p95_duration"] * 1000 for r in results]
    p99_times = [r["p99_duration"] * 1000 for r in results]
    
    x = np.arange(len(endpoints))
    width = 0.25
    
    plt.bar(x - width, median_times, width, label="Median")
    plt.bar(x, p95_times, width, label="P95")
    plt.bar(x + width, p99_times, width, label="P99")
    
    plt.xlabel("Endpoint")
    plt.ylabel("Response Time (ms)")
    plt.title("API Response Times")
    plt.xticks(x, endpoints, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "response_time_chart.png"))
    
    print(f"Report generated in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="API Performance Benchmark Tool")
    parser.add_argument("--config", required=True, help="Benchmark configuration file")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--base-url", required=True, help="Base URL for API")
    parser.add_argument("--username", help="Username for authentication")
    parser.add_argument("--password", help="Password for authentication")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per endpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Authenticate if credentials provided
    auth_token = None
    if args.username and args.password:
        auth_token = login(args.base_url, args.username, args.password)
        if not auth_token:
            print("Authentication failed. Proceeding without authentication.")
    
    # Run benchmarks
    results = []
    for endpoint_config in config["endpoints"]:
        # Override concurrency and iterations if specified in endpoint config
        concurrency = endpoint_config.get("concurrency", args.concurrency)
        iterations = endpoint_config.get("iterations", args.iterations)
        
        result = run_benchmark(
            endpoint_config,
            args.base_url,
            auth_token,
            concurrency,
            iterations
        )
        
        results.append(result)
        print(f"Completed: {result['method']} {result['endpoint']}")
        print(f"  RPS: {result['requests_per_second']:.2f}, Success Rate: {result['success_rate'] * 100:.1f}%")
        print(f"  Median: {result['median_duration'] * 1000:.1f}ms, P95: {result['p95_duration'] * 1000:.1f}ms")
        print()
    
    # Generate report
    generate_report(results, args.output_dir)

if __name__ == "__main__":
    main()
```

Example benchmark configuration file:

```json
{
  "endpoints": [
    {
      "endpoint": "/health",
      "method": "GET"
    },
    {
      "endpoint": "/api/components",
      "method": "GET"
    },
    {
      "endpoint": "/api/components/PIP001",
      "method": "GET"
    },
    {
      "endpoint": "/api/predictions/failure",
      "method": "GET",
      "params": {"component_id": "PIP001"}
    },
    {
      "endpoint": "/api/feedback",
      "method": "POST",
      "json_data": {
        "source": "web",
        "category": "ui",
        "content": "Benchmark feedback"
      },
      "iterations": 50,
      "concurrency": 5
    }
  ]
}
```

## 7. User Training and Documentation

### 7.1 Creating Comprehensive Documentation

Generate API documentation with Swagger:

```python
# Filename: api-service/swagger.py
from flask_swagger_ui import get_swaggerui_blueprint
import yaml
import os

# Path to the OpenAPI/Swagger YAML file
SWAGGER_YAML_PATH = os.path.join(os.path.dirname(__file__), 'swagger.yaml')

def create_swagger_blueprint():
    """Create Flask blueprint for Swagger UI"""
    swagger_ui_blueprint = get_swaggerui_blueprint(
        '/api/docs',
        '/api/swagger.yaml',
        config={
            'app_name': "Water Network Intelligence System API"
        }
    )
    
    return swagger_ui_blueprint

def serve_swagger_file(app):
    """Add route to serve the Swagger YAML file"""
    @app.route('/api/swagger.yaml')
    def get_swagger():
        with open(SWAGGER_YAML_PATH, 'r') as f:
            return yaml.safe_load(f)
```

Create a Swagger YAML file for your API:

```yaml
# Filename: api-service/swagger.yaml
openapi: 3.0.0
info:
  title: Water Network Intelligence System API
  description: API for the Intelligent Water Network Management System
  version: 1.0.0
  contact:
    email: support@example.com
servers:
  - url: /api
    description: Current API server
paths:
  /components:
    get:
      summary: Get water network components
      description: Returns a list of components in the water network
      parameters:
        - name: type
          in: query
          description: Filter components by type
          schema:
            type: string
            enum: [Valve, Pipe, Pump, Tank, Junction, Reservoir, Sensor]
        - name: zone
          in: query
          description: Filter components by zone
          schema:
            type: string
        - name: status
          in: query
          description: Filter components by status
          schema:
            type: string
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  components:
                    type: array
                    items:
                      $ref: '#/components/schemas/Component'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
      security:
        - BearerAuth: []
  
  /components/{component_id}:
    get:
      summary: Get component details
      description: Returns detailed information about a specific component
      parameters:
        - name: component_id
          in: path
          required: true
          description: ID of the component
          schema:
            type: string
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ComponentDetail'
        '404':
          description: Component not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
      security:
        - BearerAuth: []
  
  /predictions/failure:
    get:
      summary: Get failure prediction
      description: Returns failure prediction for a specific component
      parameters:
        - name: component_id
          in: query
          required: true
          description: ID of the component
          schema:
            type: string
        - name: horizon
          in: query
          description: Prediction horizon in days
          schema:
            type: integer
            default: 30
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FailurePrediction'
        '404':
          description: Component not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
      security:
        - BearerAuth: []
  
  /feedback:
    post:
      summary: Submit feedback
      description: Submit user feedback about the system
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/FeedbackRequest'
      responses:
        '201':
          description: Feedback submitted
          content:
            application/json:
              schema:
                type: object
                properties:
                  id:
                    type: string
                    description: Feedback ID
                  status:
                    type: string
                    enum: [submitted]
        '400':
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
      security:
        - BearerAuth: []
    
    get:
      summary: Get feedback records
      description: Retrieve feedback records with filtering
      parameters:
        - name: category
          in: query
          description: Filter by feedback category
          schema:
            type: string
        - name: status
          in: query
          description: Filter by feedback status
          schema:
            type: string
            enum: [new, in_progress, resolved, closed]
        - name: component_id
          in: query
          description: Filter by component ID
          schema:
            type: string
        - name: limit
          in: query
          description: Maximum number of records to return
          schema:
            type: integer
            default: 50
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  feedback:
                    type: array
                    items:
                      $ref: '#/components/schemas/FeedbackRecord'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
      security:
        - BearerAuth: []

components:
  schemas:
    Component:
      type: object
      properties:
        id:
          type: string
          description: Component unique identifier
        type:
          type: string
          description: Component type
        status:
          type: string
          description: Current status
    
    ComponentDetail:
      type: object
      properties:
        id:
          type: string
          description: Component unique identifier
        type:
          type: string
          description: Component type
        status:
          type: string
          description: Current status
        properties:
          type: object
          description: Component-specific properties
        zone:
          type: object
          properties:
            id:
              type: string
            name:
              type: string
        connected_components:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
              type:
                type: string
              relationship:
                type: string
    
    FailurePrediction:
      type: object
      properties:
        component_id:
          type: string
          description: Component identifier
        probability:
          type: number
          description: Failure probability
          format: float
          minimum: 0
          maximum: 1
        remaining_useful_life:
          type: number
          description: Estimated remaining useful life in days
          format: float
        factors:
          type: array
          description: Factors contributing to the prediction
          items:
            type: object
            properties:
              name:
                type: string
              importance:
                type: number
                format: float
              value:
                type: string
    
    FeedbackRequest:
      type: object
      required:
        - source
        - category
        - content
      properties:
        source:
          type: string
          description: Source of the feedback
          enum: [web, mobile, api]
        category:
          type: string
          description: Feedback category
          enum: [bug, feature_request, usability, performance, other]
        content:
          type: string
          description: Feedback content
        rating:
          type: integer
          description: Optional rating (1-5)
          minimum: 1
          maximum: 5
        component_id:
          type: string
          description: ID of related component (if applicable)
        component_type:
          type: string
          description: Type of related component (if applicable)
    
    FeedbackRecord:
      type: object
      properties:
        id:
          type: string
          description: Feedback unique identifier
        user_id:
          type: string
          description: User who submitted the feedback
        source:
          type: string
          description: Source of the feedback
        category:
          type: string
          description: Feedback category
        content:
          type: string
          description: Feedback content
        rating:
          type: integer
          description: Rating (if provided)
        component_id:
          type: string
          description: ID of related component (if applicable)
        component_type:
          type: string
          description: Type of related component (if applicable)
        submitted_at:
          type: string
          description: Submission timestamp
          format: date-time
        status:
          type: string
          description: Current status
          enum: [new, in_progress, resolved, closed]
        resolution:
          type: string
          description: Resolution information (if resolved)
    
    Error:
      type: object
      properties:
        error:
          type: string
          description: Error message
  
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

### 7.2 User Training Materials

Create a user training guide:

```markdown
# Water Network Intelligence System: User Training Guide

## 1. Introduction

Welcome to the Water Network Intelligence System! This guide will help you learn how to effectively use the system to monitor, analyze, and optimize your water distribution network.

## 2. Getting Started

### 2.1 System Access

To access the system:

1. Navigate to [https://water-network.example.com](https://water-network.example.com)
2. Enter your username and password
3. Click "Log In"

If you don't have access credentials, please contact your system administrator.

### 2.2 User Interface Overview

The system consists of five main sections:

1. **Dashboard**: Overview of your water network status
2. **Network Explorer**: Interactive view of your water infrastructure
3. **Analytics**: Reports and advanced analytics
4. **Maintenance**: Maintenance planning and scheduling
5. **Settings**: User preferences and system configuration

## 3. Dashboard

The dashboard provides a quick overview of your water network:

### 3.1 Key Performance Indicators

- **Network Health**: Overall health score based on component conditions
- **Alert Count**: Number of active alerts requiring attention
- **Leakage Estimate**: Estimated water loss in the network
- **Maintenance Tasks**: Upcoming and overdue maintenance

### 3.2 Interactive Map

The map shows a geographic view of your network with color-coded status:
- **Green**: Normal operation
- **Yellow**: Warning condition
- **Red**: Critical condition
- **Gray**: Offline or unknown status

Click any component on the map to see its details.

### 3.3 Alert Panel

The alert panel shows recent alerts in the system. Click an alert to:
- See detailed information
- Acknowledge the alert
- Assign to a user
- Mark as resolved

## 4. Network Explorer

The Network Explorer provides a detailed view of your water infrastructure:

### 4.1 Component Browser

The component browser allows you to:
- Filter components by type (valves, pipes, pumps, etc.)
- Search for specific components by ID or name
- Sort by various attributes (age, condition, etc.)
- Export component lists to CSV

### 4.2 Graph View

The graph view shows the connectivity between components:
- Nodes represent physical components
- Edges represent connections (pipes)
- Colors indicate component status
- Size can represent different metrics (importance, flow, etc.)

### 4.3 Component Details

Click any component to view detailed information:
- **General Information**: ID, type, installation date, etc.
- **Operational Data**: Status, pressure, flow, etc.
- **Maintenance History**: Past maintenance activities
- **Predictive Analytics**: Failure probability, remaining useful life
- **Connected Components**: List of directly connected components

## 5. Analytics

The Analytics section provides advanced insights:

### 5.1 Predefined Reports

Several predefined reports are available:
- **Age Distribution**: Component age analysis
- **Condition Assessment**: Overall condition evaluation
- **Maintenance Efficiency**: Maintenance performance metrics
- **Failure Analysis**: Historical failure patterns

### 5.2 Custom Analytics

Create custom analytics by:
1. Click "New Analysis"
2. Select analysis type (predictive, historical, comparative)
3. Choose components and metrics to analyze
4. Set parameters and time range
5. Generate and save the analysis

### 5.3 "What-If" Scenario Analysis

Simulate different scenarios:
1. Click "New Scenario"
2. Select scenario type (component failure, demand change, etc.)
3. Choose affected components
4. Set scenario parameters
5. Run simulation and view results

## 6. Maintenance

The Maintenance section helps plan and track maintenance activities:

### 6.1 Maintenance Calendar

The calendar shows scheduled maintenance tasks:
- Color-coded by priority
- Filterable by component type, zone, or team
- Click any task to view details

### 6.2 Predictive Maintenance

The system recommends maintenance based on predictive analytics:
1. Navigate to "Predictive Maintenance"
2. Review components with high failure probability
3. Select components for maintenance
4. Schedule maintenance tasks

### 6.3 Maintenance Reports

Generate reports on maintenance activities:
- **Completion Rate**: Percentage of tasks completed on time
- **Resource Utilization**: Labor and materials usage
- **Component Reliability**: Failure rates before and after maintenance
- **Cost Analysis**: Maintenance costs by component or zone

## 7. Role-Specific Functions

### 7.1 Operator Role

As an Operator, you can:
- Monitor network status and alerts
- View component details and operational data
- Update component status manually
- Record maintenance completion

### 7.2 Maintenance Role

As a Maintenance personnel, you can:
- View maintenance history and schedules
- Update maintenance records
- Record component replacements or repairs
- View predictive maintenance recommendations

### 7.3 Engineer Role

As an Engineer, you can:
- Access advanced analytics and modeling
- Run "what-if" scenarios and simulations
- View detailed network performance metrics
- Access EPANET simulation results

### 7.4 Manager Role

As a Manager, you can:
- View summary dashboards and KPIs
- Access cost and resource reports
- View maintenance efficiency metrics
- Approve major maintenance actions

## 8. Tips and Best Practices

- **Regular Updates**: Ensure component status is regularly updated
- **Alert Management**: Acknowledge and address alerts promptly
- **Data Validation**: Verify unusual readings or predictions
- **Scenario Planning**: Use "what-if" analysis for maintenance planning
- **Feedback Submission**: Submit feedback for system improvements

## 9. Getting Help

If you need assistance:
- Click the "Help" icon in the top right corner
- Email support at [support@example.com](mailto:support@example.com)
- Call the support hotline at (555) 123-4567

## 10. Appendix

- **Glossary of Terms**: Common terminology used in the system
- **Keyboard Shortcuts**: Productivity shortcuts for common actions
- **API Documentation**: Information for developers (if applicable)
```

## 8. Security Considerations

### 8.1 API Security Implementation

Create a security module for your API:

```python
# Filename: api-service/security.py
from flask import request, jsonify, current_app
import jwt
import datetime
from functools import wraps
import uuid
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
from neo4j import GraphDatabase
import os
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Neo4j connection
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# JWT settings
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key')  # Should be set in env vars
JWT_EXPIRATION = int(os.environ.get('JWT_EXPIRATION', 3600))  # 1 hour by default

# Password policy
PASSWORD_MIN_LENGTH = 10
PASSWORD_REGEX = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{10,})

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Decode token
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            
            # Check if user exists and is active
            with driver.session() as session:
                result = session.run("""
                MATCH (u:User {id: $user_id})
                RETURN u.username AS username, u.role AS role, u.active AS active
                """, user_id=data['user_id'])
                
                user = result.single()
                if not user:
                    return jsonify({'error': 'User not found'}), 401
                
                if not user['active']:
                    return jsonify({'error': 'User account is disabled'}), 401
                
                # Add user info to request context
                request.user = {
                    'id': data['user_id'],
                    'username': user['username'],
                    'role': user['role']
                }
        
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

def role_required(roles):
    """Decorator to require specific role(s)"""
    def decorator(f):
        @wraps(f)
        @token_required
        def decorated_function(*args, **kwargs):
            if not request.user:
                return jsonify({'error': 'User not authenticated'}), 401
            
            user_role = request.user.get('role')
            
            # Convert roles to list if string
            required_roles = roles
            if isinstance(roles, str):
                required_roles = [roles]
            
            if user_role not in required_roles and 'admin' not in required_roles:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def create_user(username, password, role='user', email=None, name=None):
    """Create a new user"""
    # Validate inputs
    if not username or not password:
        return False, "Username and password are required"
    
    # Check password strength
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
    
    if not PASSWORD_REGEX.match(password):
        return False, "Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character"
    
    # Check if user already exists
    with driver.session() as session:
        # Check for existing user
        result = session.run("""
        MATCH (u:User {username: $username})
        RETURN count(u) AS user_count
        """, username=username)
        
        if result.single()['user_count'] > 0:
            return False, "Username already exists"
        
        # Create user ID
        user_id = str(uuid.uuid4())
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Create user in database
        session.run("""
        CREATE (u:User {
            id: $id,
            username: $username,
            password: $password_hash,
            role: $role,
            email: $email,
            name: $name,
            active: true,
            created_at: $created_at,
            last_login: null
        })
        """, {
            'id': user_id,
            'username': username,
            'password_hash': password_hash,
            'role': role,
            'email': email,
            'name': name,
            'created_at': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
        logger.info(f"User created: {username}", extra={'user_id': user_id, 'role': role})
        
        return True, user_id

def authenticate_user(username, password):
    """Authenticate a user and return JWT token"""
    with driver.session() as session:
        # Find user
        result = session.run("""
        MATCH (u:User {username: $username})
        RETURN u.id AS id, u.password AS password_hash, u.role AS role, u.active AS active
        """, username=username)
        
        user = result.single()
        
        if not user:
            logger.warning(f"Login attempt failed: User not found - {username}")
            return None, "Invalid username or password"
        
        if not user['active']:
            logger.warning(f"Login attempt failed: Inactive account - {username}")
            return None, "Account is disabled"
        
        # Verify password
        if not check_password_hash(user['password_hash'], password):
            logger.warning(f"Login attempt failed: Invalid password - {username}")
            return None, "Invalid username or password"
        
        # Generate token
        token_payload = {
            'user_id': user['id'],
            'role': user['role'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRATION)
        }
        
        token = jwt.encode(token_payload, JWT_SECRET, algorithm='HS256')
        
        # Update last login timestamp
        session.run("""
        MATCH (u:User {id: $user_id})
        SET u.last_login = $timestamp
        """, {
            'user_id': user['id'],
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
        logger.info(f"User authenticated: {username}", extra={'user_id': user['id']})
        
        return token, None

def change_password(user_id, current_password, new_password):
    """Change a user's password"""
    # Check password strength
    if len(new_password) < PASSWORD_MIN_LENGTH:
        return False, f"New password must be at least {PASSWORD_MIN_LENGTH} characters long"
    
    if not PASSWORD_REGEX.match(new_password):
        return False, "New password must contain at least one uppercase letter, one lowercase letter, one number, and one special character"
    
    with driver.session() as session:
        # Get current password hash
        result = session.run("""
        MATCH (u:User {id: $user_id})
        RETURN u.password AS password_hash
        """, user_id=user_id)
        
        user = result.single()
        
        if not user:
            return False, "User not found"
        
        # Verify current password
        if not check_password_hash(user['password_hash'], current_password):
            return False, "Current password is incorrect"
        
        # Hash new password
        new_password_hash = generate_password_hash(new_password)
        
        # Update password
        session.run("""
        MATCH (u:User {id: $user_id})
        SET u.password = $password_hash,
            u.password_changed_at = $timestamp
        """, {
            'user_id': user_id,
            'password_hash': new_password_hash,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
        logger.info(f"Password changed for user {user_id}")
        
        return True, "Password changed successfully"

def disable_user(user_id):
    """Disable a user account"""
    with driver.session() as session:
        result = session.run("""
        MATCH (u:User {id: $user_id})
        SET u.active = false,
            u.disabled_at = $timestamp
        RETURN u.username AS username
        """, {
            'user_id': user_id,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
        user = result.single()
        
        if not user:
            return False, "User not found"
        
        logger.info(f"User disabled: {user['username']}", extra={'user_id': user_id})
        
        return True, "User account disabled"

def enable_user(user_id):
    """Enable a user account"""
    with driver.session() as session:
        result = session.run("""
        MATCH (u:User {id: $user_id})
        SET u.active = true,
            u.enabled_at = $timestamp
        RETURN u.username AS username
        """, {
            'user_id': user_id,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
        })
        
        user = result.single()
        
        if not user:
            return False, "User not found"
        
        logger.info(f"User enabled: {user['username']}", extra={'user_id': user_id})
        
        return True, "User account enabled"
```

### 8.2 HTTPS and TLS Configuration

Create a Nginx configuration for HTTPS:

```nginx
# Filename: nginx/nginx.conf
server {
    listen 80;
    server_name water-network.example.com;
    
    # Redirect HTTP to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name water-network.example.com;
    
    # SSL certificate configuration
    ssl_certificate /etc/nginx/ssl/water-network.crt;
    ssl_certificate_key /etc/nginx/ssl/water-network.key;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-XSS-Protection "1; mode=block";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self'";
    
    # Proxy to web interface
    location / {
        proxy_pass http://web-interface:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Proxy to API service
    location /api/ {
        proxy_pass http://api-service:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for longer operations
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Serve static files directly
    location /static/ {
        alias /var/www/static/;
        expires 30d;
    }
    
    # Limit request size
    client_max_body_size 10M;
}
```

## 9. Performance Tuning

### 9.1 Neo4j Performance Optimization

Create a Neo4j configuration file with performance optimizations:

```properties
# Filename: neo4j/conf/neo4j.conf
# Memory Settings
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=2G

# Network Settings
dbms.default_listen_address=0.0.0.0
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.listen_address=0.0.0.0:7474
dbms.connector.https.listen_address=0.0.0.0:7473

# Transaction Settings
dbms.tx_state.memory_allocation=ON_HEAP
dbms.memory.pagecache.flush.buffer.enabled=true
dbms.tx_log.rotation.size=100M
dbms.tx_log.rotation.retention_policy=1 days

# Query Settings
dbms.cypher.min_replan_interval=10s
dbms.cypher.statistics_divergence_threshold=0.5
dbms.cypher.debug.lenient_create_relationship=false

# Cache Settings
dbms.memory.off_heap.enabled=true
dbms.memory.off_heap.max_size=2G

# Security Settings
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*

# Plugins
dbms.unmanaged_extension_classes=org.neo4j.graphql=/graphql

# Metrics and Monitoring
metrics.enabled=true
metrics.csv.enabled=true
metrics.csv.interval=10s
metrics.jmx.enabled=true
metrics.prometheus.enabled=true
metrics.prometheus.endpoint=localhost:2004

# Logging
dbms.logs.query.enabled=true
dbms.logs.query.rotation.size=20M
dbms.logs.query.rotation.keep_number=7
dbms.logs.query.parameter_logging_enabled=true
dbms.logs.query.time_logging_enabled=true
dbms.logs.query.allocation_logging_enabled=true
dbms.logs.query.threshold=1000ms

# Thread pools
dbms.threads.worker_count=4
dbms.threads.transaction.pool.core_size=4
dbms.threads.transaction.pool.max_size=4

# Bolt connector
dbms.connector.bolt.thread_pool_min_size=4
dbms.connector.bolt.thread_pool_max_size=8

# Miscellaneous
dbms.tx_log.rotation.retention_policy=1 days
dbms.checkpoint.interval.time=15m
dbms.checkpoint.interval.tx=100000

# APOC settings
apoc.import.file.enabled=true
apoc.export.file.enabled=true
apoc.trigger.enabled=true
```

### 9.2 API Service Optimization

Configure Gunicorn for optimal performance:

```bash
# Filename: api-service/gunicorn.conf.py
import multiprocessing

# Bind to specific address and port
bind = "0.0.0.0:5000"

# Worker settings
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "water-network-api"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
backlog = 2048
max_requests = 1000
max_requests_jitter = 50

# Server hooks
def on_starting(server):
    # Set up logging, monitoring, or other startup tasks
    pass

def on_exit(server):
    # Clean up resources
    pass

def worker_abort(worker):
    # Log or handle worker aborts
    pass
```

## 10. Practical Learning Exercise: Deploying a Mini Water Network System

For hands-on learning, let's create a simplified deployment exercise:

### Exercise: Deploy a Minimal Water Network Intelligence System

In this exercise, you'll deploy a minimal version of the Water Network Intelligence System with just the essential components:

1. Neo4j database with a small sample water network
2. Basic API service with core functionality
3. Simple web interface for visualization

#### Step 1: Create a docker-compose.yml for Development

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.5.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/your_password
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*
    volumes:
      - neo4j_data:/data
      - ./init-scripts:/var/lib/neo4j/import/init-scripts
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

  api-service:
    build:
      context: ./api-service
    ports:
      - "5000:5000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_password
      - JWT_SECRET=your_jwt_secret
    depends_on:
      neo4j:
        condition: service_healthy

  web-interface:
    build:
      context: ./web-interface
    ports:
      - "80:80"
    environment:
      - API_URL=http://localhost:5000
    depends_on:
      - api-service

volumes:
  neo4j_data:
```

#### Step 2: Initialize Neo4j Database

Create an initialization script:

```cypher
// Filename: init-scripts/initialize.cypher

// Create constraints
CREATE CONSTRAINT FOR (v:Valve) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Pipe) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (j:Junction) REQUIRE j.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:Tank) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT FOR (u:User) REQUIRE u.username IS UNIQUE;

// Create sample water network
CREATE (t1:Tank {id: 'TNK001', capacity: 500000, level: 0.8, elevation: 240, installDate: '2015-03-20'})
CREATE (j1:Junction {id: 'JCT001', elevation: 245})
CREATE (j2:Junction {id: 'JCT002', elevation: 220})
CREATE (v1:Valve {id: 'VLV001', type: 'Gate', diameter: 300, status: 'Open', installDate: '2015-03-15'})
CREATE (p1:Pipe {id: 'PIP001', length: 500, diameter: 300, material: 'Ductile Iron', installDate: '2015-03-12'})
CREATE (p2:Pipe {id: 'PIP002', length: 1200, diameter: 300, material: 'Ductile Iron', installDate: '2015-03-18'})

// Create connections
CREATE (j1)-[:CONNECTED_TO]->(v1)
CREATE (v1)-[:CONNECTED_TO]->(p1)
CREATE (p1)-[:CONNECTED_TO]->(j2)
CREATE (j2)-[:CONNECTED_TO]->(p2)
CREATE (p2)-[:CONNECTED_TO]->(t1)

// Create zone
CREATE (z:Zone {id: 'ZON001', name: 'High Pressure Zone', pressure: 80})
CREATE (j1)-[:PART_OF]->(z)
CREATE (j2)-[:PART_OF]->(z)

// Create admin user
CREATE (u:User {
    id: 'user-001',
    username: 'admin',
    password: '$2b$12$1234567890123456789012uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu', // admin123!
    role: 'admin',
    active: true,
    created_at: datetime()
})
```

#### Step 3: Build and Start the System

Use Docker Compose to build and start the system:

```bash
docker-compose up -d
```

#### Step 4: Verify Deployment

1. Check Neo4j browser at http://localhost:7474
2. Verify API service at http://localhost:5000/health
3. Access web interface at http://localhost:80

#### Step 5: Add New Components Through API

With the system running, use the API to add a new component:

```bash
# Get authentication token
TOKEN=$(curl -s -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123!"}' | jq -r '.token')

# Add a new valve
curl -X POST http://localhost:5000/api/components \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "Valve",
    "id": "VLV002",
    "valveType": "Butterfly",
    "diameter": 200,
    "status": "Open",
    "installDate": "2018-05-10"
  }'
```

#### Step 6: Monitor System Performance

Check the health metrics of your deployment:

```bash
# View Neo4j metrics
curl http://localhost:7474/metrics

# View API service metrics
curl http://localhost:5000/metrics
```

This exercise provides hands-on experience with a minimal deployment of the Water Network Intelligence System. You can expand it by adding more components, implementing additional features, or scaling the deployment.

## 11. Conclusion and Next Steps

You've now reached the end of the implementation roadmap for your Water Network Intelligence System. This guide has covered the final phase of deployment and scaling, providing you with the tools and knowledge to take your development project into production.

### Key Takeaways

1. **Deployment Strategy**: Choose the right deployment approach (on-premises, cloud, or hybrid) based on your specific requirements
2. **Containerization**: Use Docker and Kubernetes for reproducible, scalable deployments
3. **Data Migration**: Implement robust data migration strategies for production transitions
4. **Monitoring**: Set up comprehensive monitoring with Prometheus and Grafana
5. **Backup and Recovery**: Create regular backups and implement disaster recovery procedures
6. **Continuous Improvement**: Establish feedback collection and performance benchmarking
7. **Security**: Implement strong security practices with authentication, HTTPS, and proper access controls
8. **Documentation**: Create comprehensive documentation for users and administrators

### Next Steps for Advanced Development

To further enhance your Water Network Intelligence System, consider these advanced development paths:

1. **Enhanced Digital Twin**: Create a more comprehensive digital twin with real-time data integration
2. **Advanced Simulations**: Implement more sophisticated hydraulic and water quality simulations
3. **Mobile App**: Develop a mobile application for field workers
4. **IoT Integration**: Connect with IoT sensors for real-time data collection
5. **Machine Learning Enhancements**: Implement more advanced ML models for anomaly detection and optimization
6. **GIS Integration**: Enhance geographic visualization capabilities
7. **Multi-tenant Support**: Enable system use across multiple water utilities
8. **Regulatory Reporting**: Add automated compliance reporting features

## 12. Resources and References

### Deployment and DevOps
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform Documentation](https://www.terraform.io/docs)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)

### Monitoring and Observability
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/index.html)

### Neo4j Resources
- [Neo4j Operations Manual](https://neo4j.com/docs/operations-manual/current/)
- [Neo4j Performance Tuning](https://neo4j.com/developer/guide-performance-tuning/)
- [Neo4j Backup and Restore](https://neo4j.com/docs/operations-manual/current/backup-restore/)

### Security Resources
- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [Web Security Fundamentals](https://developer.mozilla.org/en-US/docs/Web/Security)
- [JWT Security Best Practices](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)

### Water Utility Standards
- [AWWA Standards](https://www.awwa.org/Publications/Standards)
- [EPA Water Utility Resources](https://www.epa.gov/waterutilityresponse)

This guide has equipped you with the knowledge and tools to successfully deploy, manage, and scale your Water Network Intelligence System. By following these practices, you'll ensure a reliable, secure, and high-performance system that delivers valuable insights for water network management.
