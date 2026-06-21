# Park+ Detailed Architecture Diagram

This file contains the detailed Mermaid.js architecture diagram for the Park+ project. You can copy the code block below and paste it into any Mermaid-compatible viewer (like GitHub, Notion, or Mermaid Live Editor).

```mermaid
graph TD

    %% Data Layer
    subgraph "1. Data Ingestion Layer"
        A1[("Police Violation Data (CSV)")]
    end

    %% Processing & ML Engine Layer
    subgraph "2. Processing & Machine Learning Engine"
        B1["01_clean_data.py<br/>(Parse dates, handle missing coords)"]
        B2["02_hotspot_clustering.py<br/>(Scikit-learn DBSCAN)"]
        B3["02b_geo_enrichment.py<br/>(MapmyIndia POIs & Road Class)"]
        B4["Feature Engineering<br/>(23 features: Lags, Time Encodings)"]
        B5{"XGBoost Regressor<br/>(Train & Save Model)"}
        
        A1 --> B1
        B1 --> B2
        B2 --> B3
        B1 --> B4
        B4 --> B5
    end

    %% Decision & Metrics Layer
    subgraph "3. Decision & Business Logic Layer"
        C1(("Congestion Impact Index (CII)<br/>[Density + Geo-Factors + Severity]"))
        C2(("Enforcement Priority Index (EPI)<br/>[CII + Density + Forecasts]"))
        C3["05c_enforcement_mode.py<br/>(Fixed Camera vs Mobile Patrol)"]
        C4["ROI Simulation Engine<br/>(Diminishing returns heuristic)"]
        
        B3 --> C1
        B5 -- "Predict Next 7 Days" --> C2
        C1 --> C2
        C1 --> C3
        B5 --> C3
        C2 --> C4
    end

    %% Presentation & Frontend Layer
    subgraph "4. Presentation & Frontend Layer"
        D1["06_generate_dashboard_data.py<br/>(Export to JSON/JS)"]
        D2["React Dashboard<br/>(Glassmorphism UI)"]
        D3["Plotly.js<br/>(Interactive Charts & Metrics)"]
        D4["Folium / OSM<br/>(Rendered Map Tiles)"]
        
        C2 --> D1
        C3 --> D1
        C4 --> D1
        D1 --> D2
        D2 --> D3
        B2 --> D4
        D4 -. "Embedded via iframe" .-> D2
    end

    %% Styling Elements
    classDef data fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#fff;
    classDef engine fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;
    classDef decision fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff;
    classDef frontend fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff;

    class A1 data;
    class B1,B2,B3,B4,B5 engine;
    class C1,C2,C3,C4 decision;
    class D1,D2,D3,D4 frontend;
```
