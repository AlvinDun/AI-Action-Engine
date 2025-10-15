# AI Action Engine for SIEM (Wazuh-ready)

A lightweight demo engine that **classifies security log lines** with a CNN (1D) text model and then **executes mapped response actions** (e.g., block IP, disable user, quarantine host, notify admin). Built for classroom demos and extendable for production.

> ⚠️ **Safety first:** Actions are **DRY-RUN by default**. No real firewall/user changes are executed unless you explicitly set `DRY_RUN=false`.

---

## Features
- **CNN for text logs** (PyTorch): Embedding → Conv1d → GlobalMaxPool → Linear.
- Minimal **training pipeline** on CSV logs.
- **FastAPI** microservice with `/predict` and `/act`.
- **Action mapper** (`config/actions_map.yaml`) → `app/actions.py` implementations.
- **Wazuh adapter** (`scripts/wazuhtail.py`) to stream alerts to the engine.
- **Deterministic demo**: comes with a tiny dataset (5 classes) and quick training.

## Quick Start
```bash
# 1) Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Train (writes to artifacts/)
python scripts/train.py --epochs 6 --max-len 128

# 3) Run API
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Predict
```bash
curl -s -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{"message":"sshd: Failed password for invalid user admin from 10.10.10.5 port 59212"}' | jq
```

### Trigger Actions (optional)
You can **let `/predict` auto-run actions** by adding `?execute=true` or call `/act` directly.

```bash
# Auto-run on predict
curl -s -X POST "http://localhost:8080/predict?execute=true"   -H "Content-Type: application/json"   -d '{"message":"sshd: Failed password for invalid user admin from 10.10.10.5 port 59212"}' | jq

# Or call /act
curl -s -X POST http://localhost:8080/act   -H "Content-Type: application/json"   -d '{"action":"block_ip", "parameters":{"ip":"10.10.10.5"}}' | jq
```

> If you see `ERROR: [Errno 98] address already in use`, another process is on the same port. Either kill it (`lsof -i :8080` then `kill <pid>`) or use a different port with `--port 8081`.

---

## Wazuh Integration (optional)
Point the adapter at your Wazuh `alerts.json` and forward entries to the engine:

```bash
# Default file path (adjust as needed)
python scripts/wazuhtail.py   --alerts /var/ossec/logs/alerts/alerts.json   --engine http://localhost:8080   --execute
```

The adapter tries common Wazuh fields (`full_log`, `data.srcip`, `agent.id/name`) and falls back to the raw JSON line. **No root is required** for dry-run mode.

---

## Project Layout
```
ai-action-engine/
├─ app/
│  ├─ __init__.py
│  ├─ main.py                # FastAPI service (predict & act)
│  ├─ model.py               # CNN text classifier
│  ├─ preprocess.py          # tokenization, vocab, tensorize
│  ├─ actions.py             # action implementations (dry-run by default)
│  ├─ utils.py               # extractors for IP/username/etc.
├─ artifacts/                # saved model + vocab after training
├─ config/
│  └─ actions_map.yaml       # class → action mapping
├─ data/
│  ├─ train.csv              # toy dataset (text,label)
│  └─ labels.txt
├─ scripts/
│  ├─ train.py               # training entrypoint
│  └─ wazuhtail.py           # forward Wazuh alerts to engine
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## Classes (demo)
- `benign`
- `ssh_bruteforce`
- `port_scan`
- `malware_dns`
- `priv_escalation`

> You can add more by extending `data/train.csv` and retraining.

---

## Configuration
- **Dry-run mode:** set `DRY_RUN=false` to attempt real actions.
- **Action mapping:** edit `config/actions_map.yaml` to map predicted labels to actions and parameter templates.

Example snippet:
```yaml
ssh_bruteforce:
  action: block_ip
  params_from: ip  # extract first IPv4 from message
malware_dns:
  action: quarantine_host
  params_from: agent  # use agent name/id if present
```

---

## Notes
- This is a teaching/demo codebase. **Review and harden** before any production use.
- Real environments should integrate with IAM/IdP, EDR, SOAR, ticketing (Jira/ServiceNow), and auditable notification channels.
- Test actions in a sandbox first. Keep `DRY_RUN=true` until you’re 100% confident.

---

## License
MIT (see `LICENSE`).
