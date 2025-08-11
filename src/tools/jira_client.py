import os
import requests
from typing import Dict, Any, Tuple


def _config() -> Tuple[str, Tuple[str, str]]:
    base = os.getenv('JIRA_BASE_URL')
    email = os.getenv('JIRA_EMAIL')
    token = os.getenv('JIRA_API_TOKEN')
    if not base or not email or not token:
        raise RuntimeError('JIRA not configured. Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN')
    return base.rstrip('/'), (email, token)


def search_issues(jql: str, max_results: int = 20) -> Dict[str, Any]:
    base, auth = _config()
    url = f"{base}/rest/api/3/search"
    resp = requests.post(url, auth=auth, json={
        'jql': jql,
        'maxResults': max_results,
        'fields': ['summary', 'status', 'assignee', 'priority']
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    issues = []
    for it in data.get('issues', []):
        fields = it.get('fields', {})
        issues.append({
            'key': it.get('key'),
            'summary': fields.get('summary'),
            'status': (fields.get('status') or {}).get('name'),
            'assignee': ((fields.get('assignee') or {}).get('displayName')),
            'priority': (fields.get('priority') or {}).get('name')
        })
    return {'issues': issues}


def get_issue(key: str) -> Dict[str, Any]:
    base, auth = _config()
    url = f"{base}/rest/api/3/issue/{key}"
    resp = requests.get(url, auth=auth, params={'expand': 'renderedFields'}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    fields = data.get('fields', {})
    comments = []
    comms = (((fields.get('comment') or {}).get('comments')) or [])
    for c in comms:
        comments.append({
            'author': ((c.get('author') or {}).get('displayName')),
            'body': c.get('body'),
            'created': c.get('created')
        })
    attachments = []
    for a in (fields.get('attachment') or []):
        attachments.append({
            'filename': a.get('filename'),
            'size': a.get('size'),
            'mimeType': a.get('mimeType')
        })
    return {
        'key': data.get('key'),
        'summary': fields.get('summary'),
        'description': fields.get('description') if isinstance(fields.get('description'), str) else None,
        'status': (fields.get('status') or {}).get('name'),
        'assignee': ((fields.get('assignee') or {}).get('displayName')),
        'priority': (fields.get('priority') or {}).get('name'),
        'comments': comments,
        'attachments': attachments
    }


def add_comment(issue_key: str, comment: str) -> Dict[str, Any]:
    base, auth = _config()
    url = f"{base}/rest/api/3/issue/{issue_key}/comment"
    resp = requests.post(url, auth=auth, json={'body': comment}, timeout=30)
    resp.raise_for_status()
    return {'ok': True}
