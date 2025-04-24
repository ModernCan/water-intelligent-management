# Water Network Data Modeling in Neo4j: A Comprehensive Guide

This guide builds on your Neo4j fundamentals knowledge to create a comprehensive data model specifically for water distribution networks. We'll cover advanced modeling techniques, best practices, and how to represent real-world water system complexities in a graph database.

## 1. Water Network Components and Their Graph Representation

### Physical Network Components

#### Primary Components

| Water Network Component | Neo4j Node Label | Description | Key Properties |
|------------------------|------------------|-------------|----------------|
| **Source** | `:Source` | Water origin points (wells, intakes) | `sourceType`, `capacity`, `waterQuality` |
| **Reservoir** | `:Reservoir` | Large water storage facilities | `capacity`, `currentLevel`, `elevation` |
| **Treatment Plant** | `:TreatmentPlant` | Water treatment facilities | `capacity`, `treatmentTypes`, `operationalStatus` |
| **Pump Station** | `:PumpStation` | Facilities with multiple pumps | `location`, `totalCapacity`, `operationalStatus` |
| **Pump** | `:Pump` | Individual pumping equipment | `pumpType`, `capacity`, `efficiency`, `power`, `operationalStatus` |
| **Tank** | `:Tank` | Water storage containers | `capacity`, `material`, `diameter`, `height`, `currentLevel` |
| **Valve** | `:Valve` | Flow control devices | `valveType`, `diameter`, `operationalStatus`, `pressureRating` |
| **Pipe** | `:Pipe` | Water conveyance | `material`, `diameter`, `length`, `roughness`, `maxPressure` |
| **Junction** | `:Junction` | Connection points | `elevation`, `basedemand`, `demandPattern` |
| **Meter** | `:Meter` | Measurement devices | `meterType`, `accuracy`, `lastCalibrationDate` |
| **Hydrant** | `:Hydrant` | Fire protection access points | `hydroutSize`, `lastFlowTest`, `color` |
| **Service Connection** | `:ServiceConnection` | Customer connection points | `customerType`, `meterSize`, `installDate` |

#### Specialized Components

| Water Network Component | Neo4j Node Label | Description | Key Properties |
|------------------------|------------------|-------------|----------------|
| **Pressure Regulator** | `:PressureRegulator` | Maintains downstream pressure | `setPoint`, `controlType` |
| **Flow Control Valve** | `:FlowControlValve` | Maintains specific flow rate | `setFlow`, `controlMode` |
| **Check Valve** | `:CheckValve` | Prevents backflow | `activationPressure`, `valveSize` |
| **Air Release Valve** | `:AirReleaseValve` | Releases trapped air | `orificeSize`, `pressureRating` |
| **Surge Tank** | `:SurgeTank` | Protects against water hammer | `volume`, `airCushion` |
| **Chemical Injection** | `:ChemicalInjection` | Disinfection/treatment injection points | `chemicalType`, `injectionRate` |
| **SCADA Sensor** | `:Sensor` | Monitoring equipment | `sensorType`, `measurementRange`, `accuracy` |

### Organizational and Logical Components

| Component | Neo4j Node Label | Description | Key Properties |
|-----------|------------------|-------------|----------------|
| **Pressure Zone** | `:PressureZone` | Area of similar pressure | `targetPressure`, `maxElevation`, `minElevation` |
| **District Metered Area** | `:DMA` | Isolated area for monitoring | `totalConnections`, `averageConsumption` |
| **Supply Zone** | `:SupplyZone` | Area supplied by common source | `population`, `averageDemand`, `peakDemand` |
| **Administrative Area** | `:AdminArea` | Management/billing district | `name`, `population`, `manager` |
| **Maintenance Zone** | `:MaintenanceZone` | Area for maintenance teams | `responsibleTeam`, `priority` |

### Relationship Types

#### Physical Connections

| Relationship Type | Description | Key Properties |
|-------------------|-------------|----------------|
| `:FEEDS` | Flow direction from one component to another | `flowDirection`, `maxCapacity` |
| `:CONNECTED_TO` | Physical connection without flow direction | `connectionType`, `installDate` |
| `:CONTAINS` | Component hierarchy (e.g., pump station contains pumps) | `installDate`, `configuration` |
| `:BYPASSES` | Component that provides alternate flow path | `bypassType`, `normallyOpen` |
| `:REGULATES` | Control relationship | `setPoint`, `controlMode` |
| `:MEASURES` | Monitoring relationship | `accuracy`, `measurementType` |

#### Organizational Relationships

| Relationship Type | Description | Key Properties |
|-------------------|-------------|----------------|
| `:PART_OF` | Component belongs to logical zone | `since`, `primaryConnection` |
| `:SERVES` | Component provides service to area | `serviceType`, `priority` |
| `:MAINTAINED_BY` | Maintenance responsibility | `scheduleInterval`, `lastMaintenance` |
| `:OPERATES` | Operational responsibility | `accessLevel`, `controlProtocol` |
| `:MONITORED_BY` | Monitoring responsibility | `alertThreshold`, `checkFrequency` |

#### Temporal Relationships

| Relationship Type | Description | Key Properties |
|-------------------|-------------|----------------|
| `:REPLACED` | Component replacement history | `replacementDate`, `reason` |
| `:INSPECTED` | Inspection history | `inspectionDate`, `findings` |
| `:REPAIRED` | Repair history | `repairDate`, `issue`, `solution` |
| `:EXPERIENCED` | Event history | `eventTimestamp`, `severity` |

## 2. Comprehensive Water Network Schema

Let's create a more detailed schema that represents a real-world water network:

```cypher
// Core components
CREATE CONSTRAINT FOR (s:Source) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT FOR (r:Reservoir) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:TreatmentPlant) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT FOR (ps:PumpStation) REQUIRE ps.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Pump) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (t:Tank) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT FOR (v:Valve) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT FOR (p:Pipe) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT FOR (j:Junction) REQUIRE j.id IS UNIQUE;
CREATE CONSTRAINT FOR (m:Meter) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT FOR (h:Hydrant) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT FOR (sc:ServiceConnection) REQUIRE sc.id IS UNIQUE;

// Specialized components
CREATE CONSTRAINT FOR (pr:PressureRegulator) REQUIRE pr.id IS UNIQUE;
CREATE CONSTRAINT FOR (fcv:FlowControlValve) REQUIRE fcv.id IS UNIQUE;
CREATE CONSTRAINT FOR (cv:CheckValve) REQUIRE cv.id IS UNIQUE;
CREATE CONSTRAINT FOR (arv:AirReleaseValve) REQUIRE arv.id IS UNIQUE;
CREATE CONSTRAINT FOR (st:SurgeTank) REQUIRE st.id IS UNIQUE;
CREATE CONSTRAINT FOR (ci:ChemicalInjection) REQUIRE ci.id IS UNIQUE;
CREATE CONSTRAINT FOR (s:Sensor) REQUIRE s.id IS UNIQUE;

// Organizational elements
CREATE CONSTRAINT FOR (pz:PressureZone) REQUIRE pz.id IS UNIQUE;
CREATE CONSTRAINT FOR (dma:DMA) REQUIRE dma.id IS UNIQUE;
CREATE CONSTRAINT FOR (sz:SupplyZone) REQUIRE sz.id IS UNIQUE;
CREATE CONSTRAINT FOR (aa:AdminArea) REQUIRE aa.id IS UNIQUE;
CREATE CONSTRAINT FOR (mz:MaintenanceZone) REQUIRE mz.id IS UNIQUE;

// Events and activities
CREATE CONSTRAINT FOR (e:Event) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT FOR (m:Maintenance) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT FOR (i:Inspection) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT FOR (r:Reading) REQUIRE r.id IS UNIQUE;

// Create indexes for common lookups
CREATE INDEX FOR (c:Component) ON (c.installDate);
CREATE INDEX FOR (c:Component) ON (c.operationalStatus);
CREATE INDEX FOR (v:Valve) ON (v.valveType);
CREATE INDEX FOR (p:Pipe) ON (p.material);
CREATE INDEX FOR (e:Event) ON (e.timestamp);
CREATE INDEX FOR (r:Reading) ON (r.timestamp);
```

## 3. Advanced Modeling Techniques

### Multi-Label Nodes for Component Classification

In water networks, components often belong to multiple categories. Use multiple labels to represent this:

```cypher
// A valve that is also a pressure regulator and remotely controlled
CREATE (v:Valve:PressureRegulator:RemoteControlled {
    id: 'PRV001',
    installDate: '2018-05-10',
    diameter: 150,
    setPoint: 45,
    controlProtocol: 'SCADA'
})
```

### Hierarchical Component Structures

Many water system components have hierarchical relationships:

```cypher
// Create a pump station with multiple pumps
CREATE (ps:PumpStation {id: 'PS001', name: 'Main Booster Station', location: 'North District'})

// Create pumps within the station
CREATE (p1:Pump {id: 'P001', type: 'Centrifugal', power: 75})
CREATE (p2:Pump {id: 'P002', type: 'Centrifugal', power: 75})
CREATE (p3:Pump {id: 'P003', type: 'Centrifugal', power: 75})

// Connect pumps to the station
CREATE (ps)-[:CONTAINS {installDate: '2018-01-15', position: 'Primary'}]->(p1)
CREATE (ps)-[:CONTAINS {installDate: '2018-01-15', position: 'Secondary'}]->(p2)
CREATE (ps)-[:CONTAINS {installDate: '2018-01-15', position: 'Backup'}]->(p3)
```

### Modeling Operational States and Configurations

Water networks have dynamic operational states:

```cypher
// Create operational configuration nodes
CREATE (c:Configuration {
    id: 'CFG001',
    name: 'Summer Peak Demand',
    activeFrom: date('2023-06-01'),
    activeTo: date('2023-09-30')
})

// Connect components to configurations with operational parameters
MATCH (p:Pump {id: 'P001'}), (c:Configuration {id: 'CFG001'})
CREATE (p)-[:OPERATES_IN {
    speed: 0.85,
    schedule: 'Monday-Friday: 7:00-19:00',
    priority: 'High'
}]->(c)

// Valve settings in this configuration
MATCH (v:Valve {id: 'V001'}), (c:Configuration {id: 'CFG001'})
CREATE (v)-[:OPERATES_IN {
    position: 'Partially Open',
    openingPercentage: 65,
    overridePermitted: true
}]->(c)
```

### Time-Series Data Modeling

Water systems generate massive amounts of time-series data:

```cypher
// Create time-series nodes for readings
MATCH (s:Sensor {id: 'S001'})
CREATE (r1:Reading {
    id: 'READ0001',
    timestamp: datetime('2023-01-01T00:00:00'),
    value: 68.5,
    unit: 'psi',
    quality: 'Good'
})
CREATE (r2:Reading {
    id: 'READ0002',
    timestamp: datetime('2023-01-01T01:00:00'),
    value: 69.2,
    unit: 'psi',
    quality: 'Good'
})
CREATE (s)-[:HAS_READING]->(r1)
CREATE (s)-[:HAS_READING]->(r2)
```

For high-volume time-series data, consider these approaches:

1. **Aggregation**: Store hourly/daily summaries rather than every reading
2. **Bucketing**: Group readings into time buckets with min/max/avg values
3. **External Storage**: Use specialized time-series databases with Neo4j for topology

Example of bucketed readings:

```cypher
MATCH (s:Sensor {id: 'S001'})
CREATE (b:ReadingBucket {
    id: 'BUCKET-S001-20230101',
    date: date('2023-01-01'),
    minValue: 65.2,
    maxValue: 72.8,
    avgValue: 68.7,
    readings: 24,
    unit: 'psi'
})
CREATE (s)-[:HAS_READINGS]->(b)
```

### Geographic and Spatial Modeling

Water networks have strong spatial components:

```cypher
// Create components with spatial points
MATCH (v:Valve {id: 'V001'})
SET v.location = point({x: -74.0060, y: 40.7128, crs: 'WGS-84'})

// Create a pipe with a spatial path
MATCH (p:Pipe {id: 'P001'})
SET p.path = [
    point({x: -74.0060, y: 40.7128, crs: 'WGS-84'}),
    point({x: -74.0065, y: 40.7130, crs: 'WGS-84'}),
    point({x: -74.0070, y: 40.7135, crs: 'WGS-84'})
]
```

### Event and Incident Modeling

Track infrastructure events:

```cypher
// Create an event
CREATE (e:Event {
    id: 'E001',
    type: 'MainBreak',
    timestamp: datetime('2023-02-15T08:30:00'),
    severity: 'Major',
    description: 'Main break on Market Street',
    reportedBy: 'Public'
})

// Connect affected components
MATCH (e:Event {id: 'E001'}), (p:Pipe {id: 'P001'})
CREATE (e)-[:AFFECTED]->(p)

// Record response actions
MATCH (e:Event {id: 'E001'})
CREATE (a:Action {
    id: 'A001',
    type: 'Shutdown',
    timestamp: datetime('2023-02-15T09:15:00'),
    personnel: 'Emergency Crew A'
})
CREATE (e)-[:HAS_RESPONSE]->(a)

// Record customer impacts
MATCH (e:Event {id: 'E001'}), (z:SupplyZone {id: 'Z001'})
CREATE (e)-[:IMPACTED {
    customersAffected: 250,
    estimatedDuration: duration('PT6H'),
    notificationSent: true
}]->(z)
```

## 4. Complex Water Network Modeling Scenarios

### Modeling Pressure Zones and District Metered Areas (DMAs)

```cypher
// Create pressure zones
CREATE (pz1:PressureZone {
    id: 'PZ001',
    name: 'Downtown High Pressure Zone',
    targetPressure: 80,
    minElevation: 10,
    maxElevation: 50
})

CREATE (pz2:PressureZone {
    id: 'PZ002',
    name: 'Uptown Medium Pressure Zone',
    targetPressure: 60,
    minElevation: 50,
    maxElevation: 100
})

// Create District Metered Areas
CREATE (dma1:DMA {
    id: 'DMA001',
    name: 'North District DMA',
    connectionCount: 1250,
    averageConsumption: 650000,
    leakageTarget: 0.15
})

// Connect components to zones
MATCH (j:Junction), (pz:PressureZone {id: 'PZ001'})
WHERE j.elevation >= pz.minElevation AND j.elevation < pz.maxElevation
CREATE (j)-[:PART_OF]->(pz)

// Set boundary valves for DMAs
MATCH (v:Valve {id: 'V025'}), (dma:DMA {id: 'DMA001'})
CREATE (v)-[:BOUNDARY_OF {role: 'Inlet', normallyOpen: true}]->(dma)
```

### Modeling Alternative Flow Paths and Redundancy

```cypher
// Create parallel pipes for redundancy
MATCH (j1:Junction {id: 'J001'}), (j2:Junction {id: 'J002'})

CREATE (p1:Pipe {
    id: 'P101',
    diameter: 300,
    material: 'Ductile Iron',
    length: 500,
    installDate: '2010-05-15'
})

CREATE (p2:Pipe {
    id: 'P102',
    diameter: 300,
    material: 'Ductile Iron',
    length: 520,
    installDate: '2010-05-15',
    status: 'Standby'
})

CREATE (j1)-[:CONNECTED_TO]->(p1)-[:CONNECTED_TO]->(j2)
CREATE (j1)-[:CONNECTED_TO]->(p2)-[:CONNECTED_TO]->(j2)

// Create relationship between parallel components
CREATE (p1)-[:HAS_ALTERNATE {alternateType: 'Parallel', activationCondition: 'Primary Failure'}]->(p2)
```

### Modeling Maintenance Schedules and History

```cypher
// Create maintenance templates
CREATE (mt:MaintenanceTemplate {
    id: 'MT001',
    type: 'Valve Exercise',
    frequency: duration('P6M'),
    estimatedDuration: duration('PT1H'),
    requiredPersonnel: 2,
    procedures: 'Standard valve exercise procedure'
})

// Assign maintenance templates to components
MATCH (v:Valve), (mt:MaintenanceTemplate {id: 'MT001'})
WHERE v.valveType IN ['Gate', 'Butterfly']
CREATE (v)-[:REQUIRES_MAINTENANCE {priority: 'Medium'}]->(mt)

// Record maintenance history
MATCH (v:Valve {id: 'V001'})
CREATE (m:Maintenance {
    id: 'M001',
    date: date('2023-01-15'),
    personnel: 'John Smith, Maria Garcia',
    findings: 'Valve operates smoothly, no issues',
    nextScheduled: date('2023-07-15')
})
CREATE (v)-[:HAS_MAINTENANCE]->(m)
```

### Modeling Water Quality and Sampling Points

```cypher
// Create sampling points
CREATE (sp:SamplingPoint {
    id: 'SP001',
    location: 'Main Street Fire Station',
    type: 'Bacteriological',
    frequency: 'Weekly'
})

// Connect sampling points to the network
MATCH (j:Junction {id: 'J025'}), (sp:SamplingPoint {id: 'SP001'})
CREATE (j)-[:HAS_SAMPLING_POINT]->(sp)

// Record water quality readings
MATCH (sp:SamplingPoint {id: 'SP001'})
CREATE (wq:WaterQualityReading {
    id: 'WQ001',
    timestamp: datetime('2023-01-10T09:15:00'),
    chlorineResidual: 0.8,
    pH: 7.2,
    turbidity: 0.2,
    temperature: 12.5,
    bacteriological: 'Negative',
    takenBy: 'Sarah Johnson'
})
CREATE (sp)-[:HAS_READING]->(wq)
```

### Modeling Demand Patterns and Usage Types

```cypher
// Create demand patterns
CREATE (dp:DemandPattern {
    id: 'DP001',
    name: 'Residential',
    patternType: 'Weekly',
    multipliers: [
        0.6, 0.5, 0.4, 0.4, 0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.2, 1.1,
        1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7
    ]
})

// Assign patterns to junctions
MATCH (j:Junction {id: 'J025'}), (dp:DemandPattern {id: 'DP001'})
WHERE j.customerType = 'Residential'
CREATE (j)-[:HAS_DEMAND_PATTERN]->(dp)

// Create customer types
CREATE (ct:CustomerType {
    id: 'CT001',
    name: 'Residential',
    averageDemand: 0.4,
    peakFactor: 1.8
})

CREATE (ct:CustomerType {
    id: 'CT002',
    name: 'Commercial',
    averageDemand: 1.2,
    peakFactor: 1.5
})

// Assign customer types to service connections
MATCH (sc:ServiceConnection), (ct:CustomerType {id: 'CT001'})
WHERE sc.id STARTS WITH 'RES'
CREATE (sc)-[:HAS_CUSTOMER_TYPE]->(ct)
```

## 5. Integration with GIS and External Systems

### Importing from GIS

Most water utilities maintain their network in GIS systems. Here's how to import from common formats:

```cypher
// Load valve data from GIS export
LOAD CSV WITH HEADERS FROM 'file:///valves_gis.csv' AS row
CREATE (v:Valve {
    id: row.AssetID,
    valveType: row.ValveType,
    diameter: toInteger(row.Diameter),
    installDate: row.InstallDate,
    material: row.Material,
    location: point({
        x: toFloat(row.Longitude), 
        y: toFloat(row.Latitude),
        crs: 'WGS-84'
    })
})

// Load pipe data with geometry
LOAD CSV WITH HEADERS FROM 'file:///pipes_gis.csv' AS row
WITH row, apoc.convert.fromJsonList(row.Geometry) AS coordinates
CREATE (p:Pipe {
    id: row.AssetID,
    material: row.Material,
    diameter: toInteger(row.Diameter),
    length: toFloat(row.Length),
    installDate: row.InstallDate
})
WITH p, coordinates
CALL {
    WITH p, coordinates
    WITH p, [coord in coordinates | point({x: coord[0], y: coord[1], crs: 'WGS-84'})] AS points
    SET p.path = points
}
```

### Modeling External System Connections

```cypher
// Create SCADA system node
CREATE (scada:ExternalSystem {
    id: 'SYS001',
    name: 'SCADA System',
    vendor: 'Schneider Electric',
    version: '10.2',
    apiEndpoint: 'https://scada.utility.com/api/v1'
})

// Link components to SCADA tags
MATCH (v:Valve {id: 'V001'}), (scada:ExternalSystem {id: 'SYS001'})
CREATE (v)-[:HAS_EXTERNAL_REFERENCE {
    externalId: 'TAG:V001_POS',
    dataType: 'Integer',
    updateFrequency: 'PT30S',
    readWrite: 'ReadWrite'
}]->(scada)

// Create customer information system
CREATE (cis:ExternalSystem {
    id: 'SYS002',
    name: 'Customer Information System',
    vendor: 'Oracle',
    version: '8.5',
    apiEndpoint: 'https://cis.utility.com/api/v1'
})

// Link service connections to customer accounts
MATCH (sc:ServiceConnection {id: 'SC001'}), (cis:ExternalSystem {id: 'SYS002'})
CREATE (sc)-[:HAS_EXTERNAL_REFERENCE {
    externalId: 'ACCT:1095623',
    dataType: 'Account',
    lastSync: datetime('2023-01-01T00:00:00')
}]->(cis)
```

## 6. Practical Exercise: Complete Water Network Model

Let's build a more complete water network model based on these principles:

```cypher
// Clear the database (be careful with this in production!)
MATCH (n) DETACH DELETE n;

// Create water sources
CREATE (s1:Source:Reservoir {
    id: 'SRC001',
    name: 'Highland Reservoir',
    capacity: 10000000,
    currentLevel: 0.85,
    elevation: 320,
    waterQuality: 'Good',
    sourceType: 'Surface Water'
})

CREATE (s2:Source:Well {
    id: 'SRC002',
    name: 'Valley Well Field',
    capacity: 5000000,
    depth: 150,
    waterQuality: 'Good',
    sourceType: 'Groundwater'
})

// Create treatment plants
CREATE (tp1:TreatmentPlant {
    id: 'TP001',
    name: 'Main Water Treatment Plant',
    capacity: 15000000,
    treatmentTypes: ['Sedimentation', 'Filtration', 'Disinfection'],
    operationalStatus: 'Active'
})

// Create pump stations
CREATE (ps1:PumpStation {
    id: 'PS001',
    name: 'High Service Pump Station',
    totalCapacity: 12000000,
    operationalStatus: 'Active'
})

// Create individual pumps
CREATE (p1:Pump {
    id: 'PMP001',
    pumpType: 'Vertical Turbine',
    capacity: 4000000,
    power: 150,
    efficiency: 0.85,
    installDate: '2015-06-12',
    operationalStatus: 'Active'
})

CREATE (p2:Pump {
    id: 'PMP002',
    pumpType: 'Vertical Turbine',
    capacity: 4000000,
    power: 150,
    efficiency: 0.85,
    installDate: '2015-06-12',
    operationalStatus: 'Active'
})

CREATE (p3:Pump {
    id: 'PMP003',
    pumpType: 'Vertical Turbine',
    capacity: 4000000,
    power: 150,
    efficiency: 0.85,
    installDate: '2015-06-15',
    operationalStatus: 'Standby'
})

// Create tanks
CREATE (t1:Tank {
    id: 'TNK001',
    name: 'North Hill Tank',
    capacity: 2000000,
    diameter: 25,
    height: 30,
    material: 'Steel',
    elevation: 280,
    currentLevel: 0.75,
    installDate: '2010-08-20',
    operationalStatus: 'Active'
})

// Create pressure zones
CREATE (pz1:PressureZone {
    id: 'PZ001',
    name: 'High Pressure Zone',
    targetPressure: 80,
    minElevation: 200,
    maxElevation: 250
})

CREATE (pz2:PressureZone {
    id: 'PZ002',
    name: 'Medium Pressure Zone',
    targetPressure: 60,
    minElevation: 150,
    maxElevation: 200
})

// Create pressure reducing valves
CREATE (prv1:Valve:PressureRegulator {
    id: 'PRV001',
    valveType: 'Pressure Reducing',
    diameter: 300,
    setPoint: 60,
    operationalStatus: 'Active',
    installDate: '2016-04-10'
})

// Create isolation valves
CREATE (v1:Valve {
    id: 'VLV001',
    valveType: 'Gate',
    diameter: 400,
    operationalStatus: 'Open',
    installDate: '2015-03-15'
})

CREATE (v2:Valve {
    id: 'VLV002',
    valveType: 'Butterfly',
    diameter: 300,
    operationalStatus: 'Open',
    installDate: '2016-05-20'
})

// Create junctions
CREATE (j1:Junction {
    id: 'JCT001',
    elevation: 240,
    basedemand: 0
})

CREATE (j2:Junction {
    id: 'JCT002',
    elevation: 235,
    basedemand: 0
})

CREATE (j3:Junction {
    id: 'JCT003',
    elevation: 190,
    basedemand: 0
})

CREATE (j4:Junction {
    id: 'JCT004',
    elevation: 185,
    basedemand: 0
})

CREATE (j5:Junction {
    id: 'JCT005',
    elevation: 175,
    basedemand: 0
})

// Create pipes
CREATE (pp1:Pipe {
    id: 'PIP001',
    material: 'Ductile Iron',
    diameter: 600,
    length: 1200,
    installDate: '2015-03-20',
    roughness: 130
})

CREATE (pp2:Pipe {
    id: 'PIP002',
    material: 'Ductile Iron',
    diameter: 400,
    length: 800,
    installDate: '2015-04-15',
    roughness: 130
})

CREATE (pp3:Pipe {
    id: 'PIP003',
    material: 'PVC',
    diameter: 300,
    length: 1500,
    installDate: '2016-06-10',
    roughness: 150
})

CREATE (pp4:Pipe {
    id: 'PIP004',
    material: 'PVC',
    diameter: 300,
    length: 1200,
    installDate: '2016-07-05',
    roughness: 150
})

// Create flow meters
CREATE (m1:Meter {
    id: 'MTR001',
    meterType: 'Magnetic',
    diameter: 400,
    accuracy: 0.5,
    installDate: '2015-03-25',
    lastCalibrationDate: '2022-08-15'
})

CREATE (m2:Meter {
    id: 'MTR002',
    meterType: 'Ultrasonic',
    diameter: 300,
    accuracy: 0.5,
    installDate: '2016-06-15',
    lastCalibrationDate: '2022-09-20'
})

// Create SCADA sensors
CREATE (s1:Sensor {
    id: 'SEN001',
    sensorType: 'Pressure',
    measurementRange: '0-150psi',
    accuracy: 0.1,
    installDate: '2015-03-25',
    lastCalibrationDate: '2022-10-05'
})

// Build connections
// Source to treatment plant
MATCH (s:Source {id: 'SRC001'}), (tp:TreatmentPlant {id: 'TP001'})
CREATE (s)-[:FEEDS {flowCapacity: 10000000}]->(tp)

// Treatment plant to pump station
MATCH (tp:TreatmentPlant {id: 'TP001'}), (ps:PumpStation {id: 'PS001'})
CREATE (tp)-[:FEEDS {flowCapacity: 12000000}]->(ps)

// Pumps in pump station
MATCH (ps:PumpStation {id: 'PS001'}), (p:Pump)
WHERE p.id IN ['PMP001', 'PMP002', 'PMP003']
CREATE (ps)-[:CONTAINS]->(p)

// Pump station to first junction
MATCH (ps:PumpStation {id: 'PS001'}), (j:Junction {id: 'JCT001'})
CREATE (ps)-[:FEEDS]->(j)

// First junction to valve and pipe
MATCH (j:Junction {id: 'JCT001'}), (v:Valve {id: 'VLV001'})
CREATE (j)-[:CONNECTED_TO]->(v)

MATCH (v:Valve {id: 'VLV001'}), (p:Pipe {id: 'PIP001'})
CREATE (v)-[:CONNECTED_TO]->(p)

// Add flow meter to pipe
MATCH (p:Pipe {id: 'PIP001'}), (m:Meter {id: 'MTR001'})
CREATE (p)-[:HAS_METER]->(m)

// Connect pipe to next junction
MATCH (p:Pipe {id: 'PIP001'}), (j:Junction {id: 'JCT002'})
CREATE (p)-[:CONNECTED_TO]->(j)

// Junction to tank
MATCH (j:Junction {id: 'JCT002'}), (t:Tank {id: 'TNK001'})
CREATE (j)-[:CONNECTED_TO]->(t)

// Junction to next pipe
MATCH (j:Junction {id: 'JCT002'}), (p:Pipe {id: 'PIP002'})
CREATE (j)-[:CONNECTED_TO]->(p)

// Pipe to pressure reducing valve
MATCH (p:Pipe {id: 'PIP002'}), (prv:Valve:PressureRegulator {id: 'PRV001'})
CREATE (p)-[:CONNECTED_TO]->(prv)

// Pressure reducing valve to junction
MATCH (prv:Valve:PressureRegulator {id: 'PRV001'}), (j:Junction {id: 'JCT003'})
CREATE (prv)-[:CONNECTED_TO]->(j)

// Add sensor to junction
MATCH (j:Junction {id: 'JCT003'}), (s:Sensor {id: 'SEN001'})
CREATE (j)-[:HAS_SENSOR]->(s)

// Junction to next pipes
MATCH (j:Junction {id: 'JCT003'}), (p:Pipe {id: 'PIP003'})
CREATE (j)-[:CONNECTED_TO]->(p)

// Add meter to pipe
MATCH (p:Pipe {id: 'PIP003'}), (m:Meter {id: 'MTR002'})
CREATE (p)-[:HAS_METER]->(m)

// Connect pipe to final junction
MATCH (p:Pipe {id: 'PIP003'}), (j:Junction {id: 'JCT004'})
CREATE (p)-[:CONNECTED_TO]->(j)

// Junction to final pipe
MATCH (j:Junction {id: 'JCT003'}), (p:Pipe {id: 'PIP004'})
CREATE (j)-[:CONNECTED_TO]->(p)

// Connect pipe to final junction
MATCH (p:Pipe {id: 'PIP004'}), (j:Junction {id: 'JCT005'})
CREATE (p)-[:CONNECTED_TO]->(j)

// Assign junctions to pressure zones
MATCH (j:Junction), (pz:PressureZone {id: 'PZ001'})
WHERE j.id IN ['JCT001', 'JCT002'] 
CREATE (j)-[:PART_OF]->(pz)

MATCH (j:Junction), (pz:PressureZone {id: 'PZ002'})
WHERE j.id IN ['JCT003', 'JCT004', 'JCT005']
CREATE (j)-[:PART_OF]->(pz)

// Set PRV as zone boundary
MATCH (prv:Valve:PressureRegulator {id: 'PRV001'}), 
      (pz1:PressureZone {id: 'PZ001'}),
      (pz2:PressureZone {id: 'PZ002'})
CREATE (prv)-[:BOUNDARY_OF {fromZone: pz1.id, toZone: pz2.id}]->(pz2)

// Add maintenance records
MATCH (v:Valve {id: 'PRV001'})
CREATE (m:Maintenance {
    id: 'MNT001',
    date: date('2022-05-15'),
    type: 'Inspection',
    findings: 'Diaphragm shows signs of wear, schedule replacement in next 6 months',
    personnel: 'John Smith'
})
CREATE (v)-[:HAS_MAINTENANCE]->(m)
```

## 7. Common Water Network Analysis Queries

Once your water network is modeled, here are valuable analysis queries:

### Trace Water Flow from Source to Endpoint

```cypher
// Trace water path from source to a specific junction
MATCH path = (s:Source {id: 'SRC001'})-[:FEEDS|CONNECTED_TO*]->(j:Junction {id: 'JCT005'})
RETURN path
```

### Find Critical Valves for Isolation

```cypher
// Find valves needed to isolate a specific pipe segment
MATCH (p:Pipe {id: 'PIP003'})
MATCH (v:Valve)-[:CONNECTED_TO*1..5]-(p)
WHERE v.operationalStatus = 'Open'
RETURN v
```

### Identify Components in a Pressure Zone

```cypher
// Find all components in a specific pressure zone
MATCH (c)-[:PART_OF]->(pz:PressureZone {id: 'PZ002'})
RETURN c
```

### Analyze Aging Infrastructure

```cypher
// Find all pipes older than 50 years, grouped by material
MATCH (p:Pipe)
WHERE date(p.installDate) < date() - duration('P50Y')
RETURN p.material, count(p) AS count, sum(p.length) AS totalLength
ORDER BY totalLength DESC
```

### Calculate Network Connectivity

```cypher
// Find critical junctions (high connectivity points)
MATCH (j:Junction)
OPTIONAL MATCH (j)-[r]-(c)
WITH j, count(r) AS connections
WHERE connections > 3
RETURN j, connections
ORDER BY connections DESC
```

### Check Redundancy Paths

```cypher
// Find alternate paths between critical points
MATCH (j1:Junction {id: 'JCT001'}), (j2:Junction {id: 'JCT005'})
CALL apoc.algo.allSimplePaths(j1, j2, 'CONNECTED_TO>', 10) YIELD path
RETURN path, length(path) AS pathLength
ORDER BY pathLength
```

### Find Components That Need Maintenance

```cypher
// Find valves due for maintenance
MATCH (v:Valve)
OPTIONAL MATCH (v)-[:HAS_MAINTENANCE]->(m:Maintenance)
WITH v, max(m.date) AS lastMaintenance
WHERE lastMaintenance IS NULL OR date(lastMaintenance) < date() - duration('P1Y')
RETURN v
```

## 8. Next Steps: Integrating with GraphRAG

Now that you have a comprehensive water network model in Neo4j, you're ready to integrate it with GraphRAG for intelligent retrieval. The next guide will cover:

1. How to implement GraphRAG on top of your water network model
2. Creating semantic embeddings for water network components
3. Building multi-hop reasoning capabilities
4. Developing hybrid retrieval combining graph traversal with semantic search

Your water network graph database will serve as the foundation for the entire intelligent water management system, providing the structured knowledge that GraphRAG will leverage for contextual information retrieval.

## 9. Resources and References

### Water Network Modeling
- [AWWA Water Distribution Modeling Guide](https://www.awwa.org/)
- [EPANET Water Network Simulation Software](https://www.epa.gov/water-research/epanet)
- [International Water Association Resources](https://iwa-network.org/)

### Neo4j Resources
- [Neo4j Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/)
- [APOC Library for Neo4j](https://neo4j.com/docs/apoc/current/)
- [Neo4j Spatial Capabilities](https://neo4j.com/docs/cypher-manual/current/functions/spatial/)

### Data Modeling Articles
- [Graph Data Modeling Guidelines](https://neo4j.com/developer/data-modeling/)
- [Time-Series Data in Neo4j](https://neo4j.com/developer/kb/time-series-data-neo4j/)
- [Spatial Data Modeling](https://neo4j.com/developer/spatial/)

By following this guide, you've created a comprehensive graph representation of a water distribution network in Neo4j. This model captures not only the physical components and their connections but also organizational structures, operational states, and historical events. This rich graph model will serve as the foundation for your intelligent water network management system.
