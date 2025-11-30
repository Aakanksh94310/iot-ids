# src/edge/edge_server.py

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .edge_inference import predict_one

app = FastAPI(
    title="IoT IDS Edge Inference API",
    version="1.0.0",
    description="FastAPI wrapper around RF edge model (8 features) for IoT intrusion detection.",
)


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------


class FlowRequest(BaseModel):
    device_id: Optional[str] = Field(
        default=None,
        description="Logical device identifier (camera_1, thermostat_2, etc.)",
    )
    packet_count: float
    byte_count: float
    flow_duration: float
    avg_packet_size: float
    pkt_rate: float
    byte_rate: float
    tcp_flag_syn: float
    tcp_flag_ack: float


class FlowResponse(BaseModel):
    device_id: Optional[str]
    prediction: Literal[0, 1]
    label: Literal["normal", "attack"]
    timestamp: str
    features: dict


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/predict", response_model=FlowResponse, tags=["inference"])
def predict_flow(flow: FlowRequest):
    """
    Predict whether a single IoT flow is normal (0) or attack (1).
    """
    feature_dict = {
        "packet_count": flow.packet_count,
        "byte_count": flow.byte_count,
        "flow_duration": flow.flow_duration,
        "avg_packet_size": flow.avg_packet_size,
        "pkt_rate": flow.pkt_rate,
        "byte_rate": flow.byte_rate,
        "tcp_flag_syn": flow.tcp_flag_syn,
        "tcp_flag_ack": flow.tcp_flag_ack,
    }

    y_pred = predict_one(feature_dict)
    label = "attack" if y_pred == 1 else "normal"

    return FlowResponse(
        device_id=flow.device_id,
        prediction=y_pred,
        label=label,
        timestamp=datetime.utcnow().isoformat() + "Z",
        features=feature_dict,
    )


if __name__ == "__main__":
    import uvicorn

    # Run on all interfaces so Node-RED (same machine / VM) can call it
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
