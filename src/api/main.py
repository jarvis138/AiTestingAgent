from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
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
    self_heal: Optional[bool] = False

class JiraSearchRequest(BaseModel):
    jql: str
    max_results: Optional[int] = 20

class JiraLinkRunRequest(BaseModel):
    issue_key: str
    result: Dict

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
        # If the source is a Jira issue, parse to structured context
        source = req.source or {}
        prompt_context = ""
        if source.get('type') == 'jira':
            try:
                from src.tools.jira_parser import parse_issue
                issue = source.get('issue') or {}
                ctx = parse_issue(issue.get('summary'), issue.get('description'))
                prompt_context = (
                    "User Story: " + (ctx.get('user_story') or '') + "\n" +
                    ("Acceptance Criteria:\n- " + "\n- ".join(ctx.get('acceptance_criteria') or []) + "\n" if ctx.get('acceptance_criteria') else "") +
                    ("Gherkin Steps:\n" + "\n".join(ctx.get('gherkin') or []) + "\n" if ctx.get('gherkin') else "") +
                    ("Test Goals:\n- " + "\n- ".join(ctx.get('test_goals') or []) + "\n" if ctx.get('test_goals') else "")
                )
            except Exception:
                prompt_context = ''
        if OPENAI_KEY:
            # Use OpenAI API (legacy 0.28 client) to generate a Playwright test
            import openai
            openai.api_key = OPENAI_KEY
            prompt = (
                "Write a Playwright test in TypeScript for the following requirements.\n"
                "Return only the code block with no extra commentary.\n\n"
                + ("Context:\n" + prompt_context + "\n\n" if prompt_context else "")
                + f"Input: {req.source}\n"
            )
            resp = openai.ChatCompletion.create(
                model=os.getenv('LLM_MODEL', 'gpt-4'),
                messages=[{'role':'user','content':prompt}],
                max_tokens=800,
                temperature=0.1
            )
            code = resp['choices'][0]['message']['content']
            test_id = f"gen-{uuid.uuid4().hex[:8]}"
            # persist into Playwright tests/generated
            gen_dir = os.path.join(_playwright_cwd(), 'tests', 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            test_path = os.path.join(gen_dir, f"{test_id}.spec.ts")
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(code)
            rel = os.path.relpath(test_path, _playwright_cwd())
            return {"test_id": test_id, "framework":"playwright", "code": code, "path": rel}
        else:
            # fallback: return a simple example
            test_id = f"gen-{uuid.uuid4().hex[:8]}"
            code = """import { test, expect } from '@playwright/test';

test('generated test - example', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example Domain/);
});
"""
            gen_dir = os.path.join(_playwright_cwd(), 'tests', 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            test_path = os.path.join(gen_dir, f"{test_id}.spec.ts")
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(code)
            rel = os.path.relpath(test_path, _playwright_cwd())
            return {"test_id": test_id, "framework":"playwright", "code": code, "path": rel}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def _playwright_cwd():
    """Resolve absolute path to the Playwright project directory."""
    cand = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../playwright'))
    if os.path.isdir(cand):
        return cand
    return os.path.abspath(os.path.join(os.getcwd(), 'playwright'))

def _run_playwright(tests: List[str]):
    npx = shutil.which('npx') or shutil.which('npx.cmd')
    cwd = _playwright_cwd()
    if not npx:
        return None, cwd, {"error": "npx not found. Install Node.js or run tests via the playwright Docker service."}
    cmd = [npx, 'playwright', 'test', *tests, '--reporter=list'] if tests else [npx, 'playwright', 'test', '--reporter=list']
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    return result, cwd, None

def _extract_selector_from_output(text: str) -> Optional[str]:
    if not text:
        return None
    # naive patterns commonly seen in Playwright errors
    for line in text.splitlines():
        l = line.strip().lower()
        if 'selector' in l and ('not found' in l or 'strict mode' in l or 'waiting for' in l):
            # try to get quoted selector from the original line
            for quote in ("'", '"', '`'):
                parts = line.split(quote)
                if len(parts) >= 3:
                    sel = parts[1]
                    if len(sel) < 200:
                        return sel
    return None

def _apply_selector_heal(file_path: str, old_selector: str, new_selector: str) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            src = f.read()
        if old_selector not in src:
            return False
        src2 = src.replace(old_selector, new_selector)
        if src2 == src:
            return False
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(src2)
        return True
    except Exception:
        return False

# Run tests locally in the playwright folder using npx, with optional self-heal
@app.post('/run_tests')
def run_tests(req: RunTestRequest):
    try:
        result, cwd, err = _run_playwright(req.tests)
        if err:
            return err
        response = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        try:
            from src.ai.memory.execution_memory import append_run
            append_run({'action': 'run_tests', 'tests': req.tests, 'result': {'run_output': result.stdout, 'returncode': result.returncode}})
        except Exception:
            pass
        if result.returncode != 0 and req.self_heal and req.tests:
            # attempt self-heal for first test file
            failed_selector = _extract_selector_from_output(result.stderr or result.stdout)
            if failed_selector:
                # ask our self_heal endpoint for suggestions
                try:
                    heal_payload = {"error": result.stderr or result.stdout, "dom": ""}
                    from fastapi.testclient import TestClient
                    client = TestClient(app)
                    heal_resp = client.post('/self_heal', json=heal_payload)
                    suggestions = []
                    if heal_resp.status_code == 200:
                        suggestions = heal_resp.json().get('suggestions', [])
                except Exception:
                    suggestions = []
                if suggestions:
                    new_sel = suggestions[0]['selector']
                    # Resolve absolute path for the test file relative to cwd
                    test_path = req.tests[0]
                    abs_test_path = test_path if os.path.isabs(test_path) else os.path.join(cwd, test_path)
                    if _apply_selector_heal(abs_test_path, failed_selector, new_sel):
                        # re-run
                        result2, _, _ = _run_playwright([test_path])
                        response['self_heal'] = {
                            'attempted': True,
                            'old_selector': failed_selector,
                            'new_selector': new_sel,
                            'returncode': result2.returncode,
                            'stdout': result2.stdout,
                            'stderr': result2.stderr
                        }
                        try:
                            from src.ai.memory.execution_memory import append_run
                            append_run({'action': 'self_heal', 'tests': [test_path], 'result': {'run_output': result2.stdout, 'returncode': result2.returncode}})
                        except Exception:
                            pass
                    else:
                        response['self_heal'] = {'attempted': True, 'reason': 'patch_failed'}
                else:
                    response['self_heal'] = {'attempted': True, 'reason': 'no_suggestions'}
        return response
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
        # If agent returns test code, save under Playwright project tests/generated/
        if isinstance(result, dict) and 'code' in result:
            test_code = result['code']
            test_name = result.get('test_id', f"gen_{os.urandom(4).hex()}")
            gen_dir = os.path.join(_playwright_cwd(), 'tests', 'generated')
            os.makedirs(gen_dir, exist_ok=True)
            test_path = os.path.join(gen_dir, f"{test_name}.spec.ts")
            with open(test_path, 'w') as f:
                f.write(test_code)
            # Optionally run the test
            try:
                run_result, _, _ = _run_playwright([os.path.relpath(test_path, _playwright_cwd())])
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

@app.get('/tests_generated')
def list_generated_tests():
    """List AI-generated tests under Playwright tests/generated directory."""
    gen_dir = os.path.join(_playwright_cwd(), 'tests', 'generated')
    files = []
    if os.path.isdir(gen_dir):
        for root, _, filenames in os.walk(gen_dir):
            for n in filenames:
                if n.endswith('.spec.ts'):
                    files.append(os.path.relpath(os.path.join(root, n), _playwright_cwd()))
    return {"files": sorted(files)}

# JIRA integration endpoints
@app.post('/jira/search')
def jira_search(req: JiraSearchRequest):
    try:
        from src.tools.jira_client import search_issues
        return search_issues(req.jql, req.max_results or 20)
    except Exception as e:
        return {"error": str(e)}

@app.get('/jira/issue/{key}')
def jira_issue(key: str):
    try:
        from src.tools.jira_client import get_issue
        return get_issue(key)
    except Exception as e:
        return {"error": str(e)}

@app.post('/jira/link_test_run')
def jira_link_test_run(req: JiraLinkRunRequest):
    """Attach a test run result as a comment to a JIRA issue."""
    try:
        from src.tools.jira_client import add_comment
        summary = req.result.get('summary') or ''
        rc = req.result.get('returncode')
        comment = f"Automated test run:\nReturn code: {rc}\nSummary: {summary}\n```${req.result.get('stdout','')[:2000]}```"
        return add_comment(req.issue_key, comment)
    except Exception as e:
        return {"error": str(e)}

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
