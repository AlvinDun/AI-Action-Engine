import re
from typing import Optional, Tuple

IPV4_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
USER_RE = re.compile(r'user\s+([a-zA-Z0-9._-]+)', re.IGNORECASE)
AGENT_NAME_RE = re.compile(r'agent(?:\s*name)?[=:]\s*([a-zA-Z0-9._-]+)', re.IGNORECASE)

def extract_ip(text: str) -> Optional[str]:
    m = IPV4_RE.search(text or "")
    return m.group(0) if m else None

def extract_user(text: str) -> Optional[str]:
    m = USER_RE.search(text or "")
    return m.group(1) if m else None

def extract_agent(text: str) -> Optional[str]:
    m = AGENT_NAME_RE.search(text or "")
    if m:
        return m.group(1)
    # fallback heuristics
    for token in (text or "").split():
        if token.lower().startswith(("pc-", "agent-", "workstation")):
            return token
    return None
