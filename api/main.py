from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    LABEL_MAPPING,
    BatchPredictionRequest,
    BatchPredictionResponse,
    SinglePredictionResponse,
    StudentFeatures,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
)
logger = logging.getLogger("social_media_impact_api")

# Global reference for the loaded ML model
model: Optional[Any] = None
MODEL_FILENAME = "Social_media_predector.pkl"


def resolve_model_path() -> Path:
    """Locate the model file across common directories (models/, app/, root, Docker)."""
    candidates = [
        Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME,
        Path(__file__).resolve().parent / "models" / MODEL_FILENAME,
        Path.cwd() / "models" / MODEL_FILENAME,
        Path.cwd() / "app" / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
        Path("/code/models") / MODEL_FILENAME,
        Path("/app/models") / MODEL_FILENAME,
        Path("/app") / MODEL_FILENAME,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not locate '{MODEL_FILENAME}'. Checked: {[str(c) for c in candidates]}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model on startup and release resources on shutdown."""
    global model
    try:
        model_path = resolve_model_path()
        logger.info(f"Loading trained model pipeline from {model_path}...")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        model = None
    yield
    logger.info("Shutting down API service.")


app = FastAPI(
    title="Social Media Impact on Students API",
    description=(
        "Production-ready FastAPI service to predict the overall impact of social media "
        "usage on student academic and mental well-being (Neutral, Beneficial, or Negative)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend or external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def perform_inference(df_input: pd.DataFrame) -> List[SinglePredictionResponse]:
    """Execute model prediction and probability formatting."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable on this server.",
        )

    try:
        predictions = model.predict(df_input)
        probabilities = (
            model.predict_proba(df_input)
            if hasattr(model, "predict_proba")
            else None
        )

        results = []
        for i, pred_code in enumerate(predictions):
            pred_code_int = int(pred_code)
            pred_label = LABEL_MAPPING.get(pred_code_int, "Unknown")

            probs_dict = {}
            if probabilities is not None:
                for class_idx, class_name in LABEL_MAPPING.items():
                    if class_idx < len(probabilities[i]):
                        probs_dict[class_name] = round(float(probabilities[i][class_idx]), 4)

            results.append(
                SinglePredictionResponse(
                    prediction_code=pred_code_int,
                    prediction_label=pred_label,
                    probabilities=probs_dict,
                )
            )
        return results
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint providing health status and documentation link."""
    return {
        "message": "Social Media Impact on Students Prediction API is running.",
        "documentation": "/docs",
        "health": "/health",
        "model_loaded": model is not None,
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint to monitor service and model readiness."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "classes": list(LABEL_MAPPING.values()),
    }


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Inference"],
    summary="Predict impact for a single student",
)
async def predict_single(student: StudentFeatures):
    """Receives student profile and social media usage statistics, returning the predicted impact."""
    df_data = pd.DataFrame([student.model_dump()])
    results = perform_inference(df_data)
    return results[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch prediction for multiple students",
)
async def predict_batch(request: BatchPredictionRequest):
    """Receives multiple student profiles and returns a list of predictions."""
    if not request.students:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The input student list cannot be empty.",
        )
    df_data = pd.DataFrame([s.model_dump() for s in request.students])
    results = perform_inference(df_data)
    return BatchPredictionResponse(
        total_samples=len(results),
        predictions=results,
    )
