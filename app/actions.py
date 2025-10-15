import os
from typing import Dict, Any
import subprocess

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"

def _run(cmd: str) -> Dict[str, Any]:
    if DRY_RUN:
        return {"dry_run": True, "cmd": cmd, "result": "skipped"}
    try:
        cp = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return {"dry_run": False, "cmd": cmd, "returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}
    except Exception as e:
        return {"dry_run": False, "cmd": cmd, "error": str(e)}

def block_ip(ip: str) -> Dict[str, Any]:
    # Prefer ufw if available; fallback to iptables; both are demo-only here.
    cmd = f"ufw deny from {ip} || iptables -A INPUT -s {ip} -j DROP"
    return _run(cmd)

def disable_user(user: str) -> Dict[str, Any]:
    # Lock the account; demo-only.
    cmd = f"usermod -L {user}"
    return _run(cmd)

def quarantine_host(host: str) -> Dict[str, Any]:
    # Example: add to a quarantine group or block via firewall alias.
    cmd = f"echo '{host}' >> /tmp/quarantine_list"
    return _run(cmd)

def notify_admin(message: str, channel: str = "log_only") -> Dict[str, Any]:
    # In real life: email/Slack/SIEM ticket; here we just log.
    cmd = f"logger -t ai-action-engine '{message}' || echo '{message}' >> /tmp/ai_action_notifications.log"
    return _run(cmd)
