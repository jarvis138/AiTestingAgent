import requests, os, json
API = os.getenv('FASTAPI_URL','http://localhost:8000')

def run_playwright_tests():
    # Call the orchestrator /run_tests endpoint which runs npm test in the playwright container (demo)
    resp = requests.post(f'{API}/run_tests', json={'tests':['example.spec.ts']}, timeout=600)
    return resp.json()

if __name__ == '__main__':
    print(run_playwright_tests())
