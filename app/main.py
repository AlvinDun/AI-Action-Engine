import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict, Any

from app.preprocess import load_vocab, encode
from app.model import TextCNN
from app import actions as act
from app.utils import extract_ip, extract_user, extract_agent

ARTIFACTS = Path("artifacts")
VOCAB_PATH = ARTIFACTS / "vocab.json"
MODEL_PATH = ARTIFACTS / "model.pt"
LABELS_PATH = ARTIFACTS / "labels.json"
CONFIG_PATH = Path("config/actions_map.yaml")

class PredictIn(BaseModel):
    message: str

app = FastAPI(title="AI Action Engine", version="1.0.0")

def load_artifacts():
    vocab = load_vocab(str(VOCAB_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model = TextCNN(vocab_size=len(vocab), embed_dim=64, num_classes=len(labels))
    model.load_state_dict(state)
    model.eval()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        actions_map = yaml.safe_load(f) or {}
    return vocab, labels, model, actions_map

vocab, labels, model, actions_map = load_artifacts()

def predict_label(message: str, max_len: int = 128):
    x = encode(message, vocab, max_len).unsqueeze(0)  # (1, L)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), probs.tolist()

def prepare_action(label: str, message: str) -> Dict[str, Any]:
    mapping = actions_map.get(label, {})
    action = mapping.get("action")
    params_from = mapping.get("params_from")
    params = mapping.get("params", {}) or {}

    if params_from == "ip":
        ip = extract_ip(message)
        if ip: params["ip"] = ip
    elif params_from == "user":
        user = extract_user(message)
        if user: params["user"] = user
    elif params_from == "agent":
        agent = extract_agent(message)
        if agent: params["host"] = agent

    return {"action": action, "parameters": params}

@app.post("/predict")
def predict(payload: PredictIn, execute: bool = Query(False)):
    label, conf, dist = predict_label(payload.message)
    plan = prepare_action(label, payload.message)
    result = None
    if execute and plan.get("action"):
        result = dispatch_action(plan["action"], plan.get("parameters", {}))
    return {
        "label": label,
        "confidence": round(conf, 4),
        "distribution": {labels[i]: round(float(p),4) for i,p in enumerate(dist)},
        "plan": plan,
        "executed": bool(result),
        "result": result
    }

class ActIn(BaseModel):
    action: str
    parameters: Dict[str, Any] = {}

def dispatch_action(action: str, params: Dict[str, Any]):
    if action == "block_ip" and "ip" in params:
        return act.block_ip(params["ip"])
    if action == "disable_user" and "user" in params:
        return act.disable_user(params["user"])
    if action == "quarantine_host" and ("host" in params):
        return act.quarantine_host(params["host"])
    if action == "notify_admin":
        return act.notify_admin(params.get("message","notification"), params.get("channel","log_only"))
    return {"error": "unknown_action_or_missing_parameters", "action": action, "params": params}

@app.post("/act")
def act_endpoint(payload: ActIn):
    return dispatch_action(payload.action, payload.parameters)

@app.get("/health")
def health():
    return {"status": "ok", "artifacts_present": Path(MODEL_PATH).exists()}
