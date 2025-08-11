import requests, os
API = os.getenv('FASTAPI_URL','http://localhost:8000')
def run_tests(tests):
    return requests.post(f'{API}/run_tests', json={'tests': tests}).json()
