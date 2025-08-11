from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import subprocess, uuid, os, json, shutil
import pickle
import traceback

app = FastAPI(title='AI Testing Agent - Orchestrator')

# Allow local dev UI and general usage; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analytics endpoint for dashboard charts
@app.get('/analytics')
def analytics():
    mem_path = os.getenv('AGENT_MEMORY_PATH', 'data/memory.json')
    if not os.path.exists(mem_path):
        return {'runs': [], 'analytics': {}}
    with open(mem_path, 'r') as f:
        mem = json.load(f)
    return {
        'runs': mem.get('runs', []),
        'analytics': mem.get('analytics', {})
    }

MODEL_PATH = os.getenv('MODEL_PATH', 'src/models/defect_model.pkl')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# Load model if available
model = None
model_meta = None
try:
    with open(MODEL_PATH, 'rb') as f:
        loaded = pickle.load(f)
        if isinstance(loaded, dict) and 'model' in loaded:
            model_meta = loaded
            model = loaded['model']
            feature_cols = loaded.get('feature_columns', [])
            print(f'Loaded model with feature columns: {feature_cols[:10] if len(feature_cols) > 10 else feature_cols}')
        else:
            model = loaded
            print('Loaded legacy model from', MODEL_PATH)
except Exception as e:
    print('No model loaded:', e)

class PredictRequest(BaseModel):
    repo: str
    file: str
    features: Dict = {}

class PredictResponse(BaseModel):
    risk: float
    explanation: Dict = {}

class GenerateTestRequest(BaseModel):
    source: Dict

class RunTestRequest(BaseModel):
    tests: List[str]

@app.get('/')
def root():
    return {"status":"ok", "message":"AI Testing Agent Orchestrator"}

# Simple health/status endpoint
@app.get('/health')
def health():
    mem_path = os.getenv('AGENT_MEMORY_PATH', 'data/memory.json')
    mem_runs = 0
    try:
        if os.path.exists(mem_path):
            with open(mem_path, 'r') as f:
                mem = json.load(f)
                mem_runs = len(mem.get('runs', []))
    except Exception:
        pass
    return {
        'status': 'ok',
        'model_loaded': bool(model is not None),
        'feature_columns': len(model_meta.get('feature_columns', [])) if isinstance(model_meta, dict) else 0,
        'memory_runs': mem_runs
    }

# Defect predictor using loaded model or heuristic fallback
@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # If model available and features provided, use it
        if model is not None and req.features:
            # If model_meta present, align features
            if model_meta is not None and 'feature_columns' in model_meta:
                cols = model_meta['feature_columns']
                X = [req.features.get(c, 0) for c in cols]
                # Handle different model types
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba([X])[0][1])
                else:
                    prob = float(model.predict([X])[0])
                explanation = {"source":"trained_model", "features": dict(zip(cols, X)), "model_type": model_meta.get('model_type', 'unknown')}
                return {"risk": prob, "explanation": explanation}
            else:
                X = [req.features.get(k,0) for k in ['loc','complexity','churn','num_devs']]
                prob = float(model.predict_proba([X])[0][1])
                explanation = {"source":"legacy_model", "features": dict(zip(['loc','complexity','churn','num_devs'], X))}
                return {"risk": prob, "explanation": explanation}
        # fallback heuristic
        risk = min(0.95, (len(req.file) % 10) / 10 + 0.1)
        explanation = {"heuristic":"length_mod_10", "file_length": len(req.file)}
        return {"risk": risk, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate test: if OPENAI_KEY present, call OpenAI completion, else return template
@app.post('/generate_test')
def generate_test(req: GenerateTestRequest):
    try:
        if OPENAI_KEY:
            # Use OpenAI API (legacy 0.28 client) to generate a Playwright test
            import openai
            openai.api_key = OPENAI_KEY
            prompt = (
                "Write a Playwright test in TypeScript that exercises the user flow described below.\n"
                "Return only the code block with no extra commentary.\n\n"
                f"Flow: {req.source}\n"
            )
            resp = openai.ChatCompletion.create(
                model=os.getenv('LLM_MODEL', 'gpt-4'),
                messages=[{'role':'user','content':prompt}],
                max_tokens=800,
                temperature=0.1
            )
            code = resp['choices'][0]['message']['content']
            test_id = f"gen-{uuid.uuid4().hex[:8]}"
            return {"test_id": test_id, "framework":"playwright", "code": code}
        else:
            # fallback: return a simple example
            test_id = f"gen-{uuid.uuid4().hex[:8]}"
            code = """import { test, expect } from '@playwright/test';

test('generated test - example', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example Domain/);
});
"""
            return {"test_id": test_id, "framework":"playwright", "code": code}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Run tests locally in the playwright folder using npx, with safe fallbacks
@app.post('/run_tests')
def run_tests(req: RunTestRequest):
    try:
        # Ensure we have npx available
        npx = shutil.which('npx') or shutil.which('npx.cmd')
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../playwright'))
        if not os.path.isdir(cwd):
            cwd = os.path.abspath(os.path.join(os.getcwd(), 'playwright'))
        if not npx:
            return {"error": "npx not found. Install Node.js or run tests via the playwright Docker service."}
        cmd = [npx, 'playwright', 'test', *req.tests, '--reporter=list'] if req.tests else [npx, 'playwright', 'test', '--reporter=list']
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}

@app.post('/agent')
def agent_run(payload: Dict):
    """Run the LangChain automation agent with JSON payload: { 'goal': str, 'context': dict }"""
    try:
        from src.ai.agent import run_agent
        import os
        goal = payload.get('goal','')
        context = payload.get('context',{})
        result = run_agent(goal, context)
        # If agent returns test code, save to tests/generated/
        if isinstance(result, dict) and 'code' in result:
            test_code = result['code']
            test_name = result.get('test_id', f"gen_{os.urandom(4).hex()}")
            gen_dir = os.path.join(os.path.dirname(__file__), '../../tests/generated')
            os.makedirs(gen_dir, exist_ok=True)
            test_path = os.path.join(gen_dir, f"{test_name}.spec.ts")
            with open(test_path, 'w') as f:
                f.write(test_code)
            # Optionally run the test
            try:
                import subprocess
                run_result = subprocess.run([
                    'npx', 'playwright', 'test', test_path, '--reporter=list'
                ], capture_output=True, text=True, timeout=300)
                result['run_output'] = run_result.stdout
            except Exception as e:
                result['run_error'] = str(e)
        # Optionally commit to GitHub via PR (stub)
        if isinstance(result, dict) and 'code' in result and payload.get('commit_to_github'):
            try:
                # TODO: Implement GitHub PR creation
                result['pr_status'] = 'PR creation logic not implemented yet.'
            except Exception as e:
                result['pr_error'] = str(e)
        # Save to memory
        try:
            from src.ai.memory.execution_memory import append_run
            append_run({'goal': goal, 'context': context, 'result': result})
        except Exception:
            pass
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}

# Simple self-heal endpoint: suggest alternate selectors when selector not found
@app.post('/self_heal')
def self_heal(payload: Dict):
    try:
        error = payload.get('error','')
        dom = payload.get('dom','')
        suggestions = []
        if 'selector' in error.lower() or 'no node found' in error.lower():
            # naive suggestion: try using text-based selector if label present
            if 'login' in dom.lower():
                suggestions.append({"selector": "text=Login", "reason":"text fallback contains 'Login'"})
            suggestions.append({"selector": "css=.btn-primary", "reason":"common primary button class fallback"})
        return {"fixed": bool(suggestions), "suggestions": suggestions}
    except Exception as e:
        return {"error": str(e)}

@app.post('/explain')
def explain(req: PredictRequest):
    try:
        # require model_meta with feature_columns
        if model_meta is None or 'feature_columns' not in model_meta:
            raise Exception('Explainability requires a model trained with feature columns. Train KC1 model first.')
        from src.models.explain_shap import explain_instance
        return explain_instance(req.features)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
