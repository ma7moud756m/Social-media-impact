# 📱 Social Media Impact on Students — AI & ML Prediction Service

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20(98.67%25)-green.svg)](https://xgboost.ai/)

An end-to-end Machine Learning solution designed to assess and predict how social media usage patterns impact student life, mental health, and academic performance. The system classifies outcomes into three categories: **Neutral**, **Beneficial**, or **Negative**.

Includes a **FastAPI production REST API**, an **interactive Streamlit Web Dashboard**, and a **Dockerized deployment**.

---

## 📑 Table of Contents
- [Project Architecture](#-project-architecture)
- [Model Performance Benchmark](#-model-performance-benchmark)
- [Input Features & Schema](#-input-features--schema)
- [Quick Start: Running Locally](#-quick-start-running-locally)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Launch Streamlit Web App](#2-launch-streamlit-web-app-frontend)
  - [3. Launch FastAPI Server](#3-launch-fastapi-server-backend)
- [Docker Deployment](#-docker-deployment)
  - [Using Docker Compose](#option-a-docker-compose-recommended)
  - [Using Docker CLI](#option-b-docker-cli)
  - [Troubleshooting Docker Desktop](#-troubleshooting-docker-on-windows)
- [REST API Endpoints & Usage](#-rest-api-endpoints--usage)
  - [Single Prediction (`POST /predict`)](#single-prediction-post-predict)
  - [Batch Prediction (`POST /predict/batch`)](#batch-prediction-post-predictbatch)

---

## 📁 Project Architecture

```text
├── api/                            # Production FastAPI REST application
│   ├── __init__.py
│   ├── main.py                     # API routes, lifespan & inference handlers
│   └── schemas.py                  # Pydantic v2 validation models
├── dashboard/                      # Interactive Streamlit Web Dashboard
│   ├── __init__.py
│   └── app.py                      # UI components, visual charts & preset profiles
├── models/                         # Trained ML model artifacts
│   └── Social_media_predector.pkl  # scikit-learn + XGBoost pipeline
├── Data/
│   └── Social_media_impact_on_life.csv # Survey dataset
├── Notebook/
│   └── main.ipynb                  # EDA, Feature Engineering & Benchmark training
├── app.py                          # Root launcher proxy for Streamlit
├── Dockerfile                      # Production container image with OpenMP runtime
├── docker-compose.yml              # Multi-service config (FastAPI: 8000 & Streamlit: 8501)
├── .dockerignore                   # Build exclusion rules
├── requirements.txt                # Unified project dependencies
└── README.md                       # Documentation & deployment guide
```

---

## 🏆 Model Performance Benchmark

During experimentation, multiple classification algorithms were trained and evaluated using stratified cross-validation on test data. **XGBoost** achieved the highest accuracy and macro F1-score and was selected for production deployment:

| # | Model | Accuracy | Precision | Recall | F1-Score | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 0 | **Logistic Regression** | 0.9856 | 0.9672 | 0.9451 | 0.9554 | Benchmark |
| 1 | **KNN** | 0.9556 | 0.9255 | 0.8717 | 0.8969 | Benchmark |
| 2 | **KNN GridSearch** | 0.9489 | 0.9033 | 0.8552 | 0.8778 | Benchmark |
| 3 | **Decision Tree** | 0.9711 | 0.9143 | 0.9224 | 0.9183 | Benchmark |
| 4 | **Decision Tree GridSearch** | 0.9733 | 0.9494 | 0.9510 | 0.9500 | Benchmark |
| 5 | **Random Forest** | 0.9778 | 0.9561 | 0.9356 | 0.9451 | Benchmark |
| 6 | **Random Forest GridSearch** | 0.9733 | 0.9494 | 0.9510 | 0.9500 | Benchmark |
| 7 | **XGBoost** | **0.9867** | **0.9664** | **0.9593** | **0.9624** | 🌟 **Production Model** |
| 8 | **XGBoost GridSearch** | 0.9456 | 0.9494 | 0.7695 | 0.8374 | Benchmark |
| 9 | **Voting Classifier** | 0.9833 | 0.9533 | 0.9538 | 0.9534 | Benchmark |

---

## 📊 Input Features & Schema

The model pipeline processes 14 attributes combining demographic, digital habit, and wellness indicators:

| Feature Name | Type | Valid Range / Categories | Description |
|---|---|---|---|
| `Age` | `int` | `15 – 30` | Age of student |
| `Gender` | `str` | `Female`, `Male`, `Non-Binary`, `Prefer not to say` | Gender identity |
| `Academic_Level` | `str` | `Undergraduate`, `High School`, `Postgraduate` | Current academic stage |
| `Primary_Platform` | `str` | `Instagram`, `TikTok`, `YouTube`, `Snapchat`, `LinkedIn`, `Reddit`, `X (Twitter)` | Most used platform |
| `Daily_Usage_Hours` | `float` | `0.0 – 16.0` | Average weekday screen time |
| `Weekend_Extra_Hours` | `float` | `0.0 – 8.0` | Additional hours spent on weekends |
| `Device_Type` | `str` | `Smartphone`, `Tablet`, `Laptop/PC` | Primary device used |
| `Sleep_Duration_Hours`| `float` | `2.0 – 12.0` | Sleep per night in hours |
| `Sleep_Quality_Score` | `int` | `1 – 5` | Self-reported sleep quality scale |
| `Late_Night_Usage` | `int` | `0 (No)` or `1 (Yes)` | Late-night usage indicator |
| `Social_Comparison_Frequency` | `str` | `Never`, `Rarely`, `Sometimes`, `Frequently`, `Always` | Tendency to compare life online |
| `Perceived_Stress_Score` | `float` | `0.0 – 40.0` | Measured stress index |
| `Mental_Health_Index` | `int` | `0 – 100` | Mental health well-being score |
| `Academic_Performance_GPA` | `float` | `0.0 – 4.0` | Current GPA |

### Target Classes:
- `0`: **Neutral** ⚖️ — Balanced or moderate impact.
- `1`: **Beneficial** 🌟 — Positive influence on learning and connectivity.
- `2`: **Negative** ⚠️ — Elevated stress, disrupted sleep, or academic decline.

---

## 🚀 Quick Start: Running Locally

### 1. Install Dependencies
Ensure Python 3.10+ is installed, then install all required packages:
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Web App (Frontend)
Run the interactive dashboard with topic-tailored visuals, quick student presets, and model comparison charts:
```bash
streamlit run app.py
```
*(or explicitly: `streamlit run dashboard/app.py`)*  
👉 Open your browser at: **http://localhost:8501**

### 3. Launch FastAPI Server (Backend)
Run the REST API with automatic reloading:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
👉 Interactive Swagger UI: **http://localhost:8000/docs**  
👉 Alternative ReDoc: **http://localhost:8000/redoc**

---

## 🐳 Docker Deployment

The application is containerized using `python:3.11-slim` with `libgomp1` (required by XGBoost for OpenMP acceleration on Linux).

### Option A: Docker Compose (Dual Services: API + Dashboard)
Runs both the FastAPI backend (`http://localhost:8000`) and the Streamlit dashboard (`http://localhost:8501`) simultaneously:
```bash
docker compose up --build -d
```
To stop all services:
```bash
docker compose down
```

### Option B: Docker CLI
```bash
# Build Docker image
docker build -t social-media-impact-api .

# Run container on port 8000
docker run -d -p 8000:8000 --name social-media-impact-api social-media-impact-api
```

### 💡 Troubleshooting Docker on Windows
If you receive the error:
```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```
**Fix**: Ensure **Docker Desktop** is opened and running from the Windows Start Menu before executing `docker compose`. Wait for the whale icon in the taskbar to turn green.

---

## 📡 REST API Endpoints & Usage

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status, root information, and links |
| `GET` | `/health` | Health check & model readiness status |
| `POST` | `/predict` | Predict impact for a single student |
| `POST` | `/predict/batch` | Predict impact for multiple students in one call |
| `GET` | `/docs` | Interactive Swagger UI testing environment |

---

### Single Prediction (`POST /predict`)

#### cURL Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 20,
    "Gender": "Male",
    "Academic_Level": "Undergraduate",
    "Primary_Platform": "Instagram",
    "Daily_Usage_Hours": 6.5,
    "Weekend_Extra_Hours": 2.0,
    "Device_Type": "Smartphone",
    "Sleep_Duration_Hours": 6.0,
    "Sleep_Quality_Score": 5,
    "Late_Night_Usage": 1,
    "Social_Comparison_Frequency": "Frequently",
    "Perceived_Stress_Score": 8.0,
    "Mental_Health_Index": 45,
    "Academic_Performance_GPA": 2.7
  }'
```

#### JSON Response:
```json
{
  "prediction_code": 0,
  "prediction_label": "Neutral",
  "probabilities": {
    "Neutral": 0.9278,
    "Beneficial": 0.0688,
    "Negative": 0.0035
  }
}
```

---

### Batch Prediction (`POST /predict/batch`)

#### cURL Request:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {
        "Age": 20,
        "Gender": "Male",
        "Academic_Level": "Undergraduate",
        "Primary_Platform": "Instagram",
        "Daily_Usage_Hours": 6.5,
        "Weekend_Extra_Hours": 2.0,
        "Device_Type": "Smartphone",
        "Sleep_Duration_Hours": 6.0,
        "Sleep_Quality_Score": 5,
        "Late_Night_Usage": 1,
        "Social_Comparison_Frequency": "Frequently",
        "Perceived_Stress_Score": 8.0,
        "Mental_Health_Index": 45,
        "Academic_Performance_GPA": 2.7
      },
      {
        "Age": 19,
        "Gender": "Female",
        "Academic_Level": "Undergraduate",
        "Primary_Platform": "TikTok",
        "Daily_Usage_Hours": 8.0,
        "Weekend_Extra_Hours": 3.0,
        "Device_Type": "Smartphone",
        "Sleep_Duration_Hours": 4.5,
        "Sleep_Quality_Score": 1,
        "Late_Night_Usage": 1,
        "Social_Comparison_Frequency": "Always",
        "Perceived_Stress_Score": 32.0,
        "Mental_Health_Index": 35,
        "Academic_Performance_GPA": 2.1
      }
    ]
  }'
```

---

## 🛠️ Tech Stack
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib, Pandas, NumPy
- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **Frontend Dashboard**: Streamlit, Plotly Express & Graph Objects
- **DevOps & Packaging**: Docker, Docker Compose
