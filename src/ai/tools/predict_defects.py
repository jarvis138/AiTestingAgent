import requests, os
API = os.getenv('FASTAPI_URL','http://localhost:8000')
def predict(file, features):
    return requests.post(f'{API}/predict', json={'repo':'demo','file':file,'features':features}).json()
