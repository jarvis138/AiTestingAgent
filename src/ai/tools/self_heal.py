import requests, os
API = os.getenv('FASTAPI_URL','http://localhost:8000')
def self_heal(payload):
    result = requests.post(f'{API}/self_heal', json=payload).json()
    # Optionally create PR for fixes (stub)
    if payload.get('create_pr'):
        result['pr_status'] = 'PR creation logic not implemented yet.'
    return result
