import requests, os
API = os.getenv('FASTAPI_URL','http://localhost:8000')
def generate(source):
    return requests.post(f'{API}/generate_test', json={'source': source}).json()
