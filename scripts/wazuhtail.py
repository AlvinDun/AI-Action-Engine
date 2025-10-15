import argparse, json, time, sys, requests, os, re

def pick_message(entry: dict) -> str:
    # Try common Wazuh fields, then fallback
    for key in ("full_log", "predecoder", "syscheck", "data", "location"):
        v = entry.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict) and "message" in v:
            return v["message"]
    return json.dumps(entry, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="/var/ossec/logs/alerts/alerts.json")
    ap.add_argument("--engine", default="http://localhost:8080")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    url = f"{args.engine}/predict"
    if args.execute:
        url += "?execute=true"

    print(f"[wazuhtail] streaming {args.alerts} -> {url}")
    with open(args.alerts, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, 2)  # tail -f
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.25)
                continue
            try:
                entry = json.loads(line)
            except Exception:
                # not JSONL; try to wrap
                entry = {"full_log": line.strip()}
            msg = pick_message(entry)
            try:
                r = requests.post(url, json={"message": msg}, timeout=5)
                print(r.json())
            except Exception as e:
                print(f"[wazuhtail] error posting to engine: {e}", file=sys.stderr)
                time.sleep(1)

if __name__ == "__main__":
    main()
