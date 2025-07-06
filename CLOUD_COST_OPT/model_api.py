# model_api.py - FastAPI service for traffic prediction and scaling recommendations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import uvicorn
from enum import Enum
from dataclasses import dataclass

from utils.rl_autoscaling_agent import DQNAgent, ServerType, ScalingAction, AutoScalingEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Cloud Cost Optimizer API",
    description="Traffic prediction and auto-scaling recommendations using RL Agent",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "file://",
    "file:///*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_metadata = None
rl_agent = None

SERVER_TYPE_CAPACITIES = {st.value["name"]: st.value["capacity"] for st in ServerType}
SERVER_TYPE_COSTS = {st.value["name"]: st.value["cost"] for st in ServerType}


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Cloud Cost Optimizer API!"}


class TrafficPredictionRequest(BaseModel):
    session_duration: float
    bounce_rate: float
    previous_visits: int
    conversion_rate: float
    traffic_source: str
    timestamp: Optional[str] = None
    current_servers: int = 1


class TrafficPredictionResponse(BaseModel):
    predicted_traffic: float
    recommended_servers: int
    server_type: str
    estimated_cost_per_hour: float
    confidence_score: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rl_agent_loaded: bool
    last_updated: str
    version: str


def load_model_and_config():
    global model, model_metadata, rl_agent

    try:
        model_path = Path("models/best_traffic_model.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("ML Traffic Prediction Model loaded successfully from %s", model_path)
        else:
            logger.error("ML Traffic Prediction Model file not found at %s. API will not function correctly.",
                         model_path)
            return False

        metadata_path = Path("models/model_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded from %s", metadata_path)
        else:
            logger.warning("Model metadata file not found at %s.", metadata_path)

        rl_agent_path = Path("models/autoscaling_agent.json")
        if rl_agent_path.exists():
            rl_agent = DQNAgent(state_size=8, action_size=7)
            rl_agent.load_model(rl_agent_path)
            logger.info("RL Auto-Scaling Agent (Q-table) loaded successfully from %s", rl_agent_path)
        else:
            logger.warning(
                "RL Auto-Scaling Agent (Q-table) file not found at %s. Server recommendations will fall back to simple rules.",
                rl_agent_path)
            rl_agent = None

        return True
    except Exception as e:
        logger.error(f"Error loading models/config: {str(e)}")
        return False


def create_time_features(timestamp_str: str = None) -> Dict:
    dt = datetime.now()
    if timestamp_str:
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Invalid timestamp format '{timestamp_str}'. Using current time.")

    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'is_business_hours': 1 if 9 <= dt.hour <= 17 else 0,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        'day_sin': np.sin(2 * np.pi * dt.day / 31),
        'day_cos': np.cos(2 * np.pi * dt.day / 31),
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12)
    }


def encode_traffic_source(source: str) -> Dict:
    sources = ['Organic', 'Paid', 'Referral', 'Social']
    encoded = {f'traffic_source_{s}': 0 for s in sources}
    if source in sources:
        encoded[f'traffic_source_{source}'] = 1
    else:
        logger.warning(f"Unknown traffic source: {source}. Defaulting to all zeros for encoding.")
    return encoded


def get_rl_server_recommendation(predicted_traffic: float, current_servers: int) -> Tuple[int, str, float]:
    if rl_agent is None:
        logger.warning("RL Agent not loaded. Falling back to simple default recommendation for prediction.")
        return calculate_simple_default_recommendation(predicted_traffic)

    MICRO_CAPACITY = ServerType.MICRO.value["capacity"]
    estimated_total_capacity = current_servers * MICRO_CAPACITY

    if estimated_total_capacity == 0:
        estimated_total_capacity = MICRO_CAPACITY

    estimated_utilization = (predicted_traffic / estimated_total_capacity)
    estimated_utilization = np.clip(estimated_utilization, 0.0, 5.0)

    estimated_response_time = 0.5
    if estimated_utilization <= 0.5:
        estimated_response_time = 0.5
    elif estimated_utilization <= 0.8:
        estimated_response_time = 0.5 + (estimated_utilization - 0.5) * 3.0
    else:
        estimated_response_time = 2.0 + (estimated_utilization - 0.8) * 10.0
    estimated_response_time = np.clip(estimated_response_time, 0.5, 20.0)

    mock_state = {
        "current_traffic": predicted_traffic,
        "total_capacity": estimated_total_capacity,
        "utilization": estimated_utilization,
        "response_time": estimated_response_time,
        "num_servers": current_servers,
        "server_types": [],
        "hourly_cost": 0.0,
        "sla_violation": estimated_response_time > AutoScalingEnvironment().max_response_time
    }

    scaling_action: ScalingAction = rl_agent.choose_action(mock_state)

    recommended_servers = current_servers
    cost_per_hour = 0.0
    representative_server_type_name = ServerType.MICRO.value['name']

    if scaling_action.action_type == "no_action":
        recommended_servers = current_servers
        representative_server_type_name = ServerType.MICRO.value['name']
    elif scaling_action.action_type == "scale_up":
        recommended_servers = current_servers + scaling_action.target_instances
        representative_server_type_name = scaling_action.target_server_type.value['name']
    elif scaling_action.action_type == "scale_down":
        recommended_servers = max(1, current_servers - scaling_action.target_instances)
        representative_server_type_name = ServerType.MICRO.value['name']
    elif scaling_action.action_type == "scale_out":
        recommended_servers = current_servers + scaling_action.target_instances
        representative_server_type_name = scaling_action.target_server_type.value['name']
    elif scaling_action.action_type == "scale_in":
        recommended_servers = max(1, current_servers - scaling_action.target_instances)
        representative_server_type_name = ServerType.MICRO.value['name']

    recommended_servers = max(1, recommended_servers)

    if representative_server_type_name in SERVER_TYPE_COSTS:
        cost_per_hour = recommended_servers * SERVER_TYPE_COSTS[representative_server_type_name]
    else:
        cost_per_hour = recommended_servers * ServerType.MICRO.value['cost']

    logger.info(f"RL Recommendation: Pred. Traffic: {predicted_traffic:.2f}, Current Servers: {current_servers}, "
                f"Mock State: Util={mock_state['utilization']:.2f}, RT={mock_state['response_time']:.2f}, NumS={mock_state['num_servers']}, "
                f"Action: {scaling_action.action_type} (Target Type: {representative_server_type_name}, Target Instances: {scaling_action.target_instances}) -> "
                f"Recommended: {recommended_servers} ({representative_server_type_name}), Cost: ${cost_per_hour:.4f}")

    return recommended_servers, representative_server_type_name, cost_per_hour


def calculate_simple_default_recommendation(predicted_traffic: float) -> Tuple[int, str, float]:
    if predicted_traffic < 500:
        return 1, ServerType.MICRO.value['name'], ServerType.MICRO.value['cost']
    elif predicted_traffic < 1000:
        return 1, ServerType.SMALL.value['name'], ServerType.SMALL.value['cost']
    elif predicted_traffic < 2000:
        return 1, ServerType.MEDIUM.value['name'], ServerType.MEDIUM.value['cost']
    elif predicted_traffic < 4000:
        return 1, ServerType.LARGE.value['name'], ServerType.LARGE.value['cost']
    else:
        return 1, ServerType.XLARGE.value['name'], ServerType.XLARGE.value['cost']


@app.on_event("startup")
async def startup_event():
    success = load_model_and_config()
    if not success or rl_agent is None:
        logger.warning(
            "API initialized, but some components (e.g., RL Agent) might not be loaded correctly. Check logs.")
    else:
        logger.info("API initialized successfully with all components.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None and rl_agent is not None else "degraded",
        model_loaded=model is not None,
        rl_agent_loaded=rl_agent is not None,
        last_updated=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=TrafficPredictionResponse)
async def predict_traffic(request: TrafficPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Traffic Prediction Model not loaded. API is not ready.")

    try:
        features = {
            'session_duration': request.session_duration,
            'bounce_rate': request.bounce_rate,
            'previous_visits': request.previous_visits,
            'conversion_rate': request.conversion_rate
        }

        time_features = create_time_features(request.timestamp)
        features.update(time_features)

        traffic_source_features = encode_traffic_source(request.traffic_source)
        features.update(traffic_source_features)

        lag_features = {
            'Page Views_lag1': 5.0,
            'Page Views_lag24': 5.0,
            'Page Views_lag168': 5.0,
            'Page Views_rolling_mean_24h': 5.0,
            'Page Views_rolling_mean_7d': 5.0,
            'Page Views_rolling_std_24h': 1.0,
            'Page Views_rolling_max_24h': 6.0,
            'Page Views_rolling_min_24h': 4.0,
            'Page Views_trend_24h': 0.1,
            'Page Views_volatility': 0.2
        }
        features.update(lag_features)

        if model_metadata and 'feature_columns' in model_metadata:
            expected_features = model_metadata['feature_columns']
            for col in expected_features:
                if col not in features:
                    features[col] = 0.0

            feature_df = pd.DataFrame([features])[expected_features]
        else:
            feature_df = pd.DataFrame([features])
            logger.warning("Model metadata missing 'feature_columns'. Feature order might be incorrect for prediction.")

        prediction = model.predict(feature_df)[0]
        prediction = max(0.0, prediction)

        confidence = 0.95

        servers, server_type, cost_per_hour = get_rl_server_recommendation(prediction, request.current_servers)

        return TrafficPredictionResponse(
            predicted_traffic=float(prediction),
            recommended_servers=servers,
            server_type=server_type,
            estimated_cost_per_hour=cost_per_hour,
            confidence_score=confidence,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available. Model might not be loaded.")
    return model_metadata


@app.post("/batch_predict")
async def batch_predict(requests: List[TrafficPredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Traffic Prediction Model not loaded. API is not ready.")

    results = []
    for i, request in enumerate(requests):
        try:
            prediction_response = await predict_traffic(request)
            results.append(prediction_response)
        except HTTPException as e:
            logger.error(f"Batch prediction HTTP error for request {i}: {e.detail}")
            results.append({"error": f"Prediction failed for request {i}: {e.detail}", "request_data": request.dict()})
        except Exception as e:
            logger.error(f"Unexpected batch prediction error for request {i}: {str(e)}")
            results.append(
                {"error": f"An unexpected error occurred for request {i}: {str(e)}", "request_data": request.dict()})

    return {"predictions": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)