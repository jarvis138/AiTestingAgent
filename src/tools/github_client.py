import os
import hmac
import hashlib
import requests
from typing import Dict, Any, List, Optional, Tuple

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO')  # e.g. "owner/repo"
WEBHOOK_SECRET = os.getenv('GITHUB_WEBHOOK_SECRET', '')


def verify_signature(payload: bytes, signature_header: Optional[str]) -> bool:
    """Validate GitHub webhook signature (sha256) against WEBHOOK_SECRET.
    In dev, if no secret is set, allow the request.
    """
    if not WEBHOOK_SECRET:
        return True  # allow in dev/local
    if not signature_header or not signature_header.startswith('sha256='):
        return False
    digest = hmac.new(WEBHOOK_SECRET.encode('utf-8'), payload, hashlib.sha256).hexdigest()
    expected = f"sha256={digest}"
    return hmac.compare_digest(expected, signature_header)


def _headers() -> Dict[str, str]:
    assert GITHUB_TOKEN, 'GITHUB_TOKEN env is required for GitHub API calls'
    return {
        'Authorization': f"Bearer {GITHUB_TOKEN}",
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'ai-testing-agent'
    }


def _split_repo(owner_repo: Optional[str]) -> Tuple[str, str]:
    repo = owner_repo or GITHUB_REPO or ''
    if '/' not in repo:
        raise ValueError('GITHUB_REPO must be set as "owner/repo"')
    owner, name = repo.split('/', 1)
    return owner, name


def get_pr_files(owner_repo: Optional[str], pr_number: int) -> List[Dict[str, Any]]:
    owner, name = _split_repo(owner_repo)
    url = f"https://api.github.com/repos/{owner}/{name}/pulls/{pr_number}/files"
    out: List[Dict[str, Any]] = []
    page = 1
    while True:
        resp = requests.get(url, headers=_headers(), params={'per_page': 100, 'page': page}, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        out.extend(chunk)
        page += 1
    return out


def pr_comment(owner_repo: Optional[str], pr_number: int, body: str) -> Dict[str, Any]:
    owner, name = _split_repo(owner_repo)
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=_headers(), json={'body': body}, timeout=30)
    resp.raise_for_status()
    return resp.json()
