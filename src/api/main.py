from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import PlainTextResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

# stdlib imports
import os
import json
import uuid
import shutil
import subprocess
import asyncio
import traceback
import pickle
import logging

# persistence helpers
from src.api.persistence import init_db, log_run, add_generated_test, recent_runs, add_artifact

logging.basicConfig(level=logging.INFO)

# Settings and globals
class Settings:
    api_key: Optional[str] = os.getenv('API_KEY')


settings = Settings()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# Optional model load
model = None
model_meta: Dict = {}
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'defect_model.pkl')
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    # optional meta
    meta_path = os.path.splitext(model_path)[0] + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            model_meta = json.load(f)
except Exception:
    model = None
    model_meta = {}

# App init
app = FastAPI(title='AI Testing Agent Orchestrator')

# CORS (allow local dev by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Ensure data/artifacts exists and mount for static serving
os.makedirs(os.path.join('data', 'artifacts'), exist_ok=True)
app.mount('/artifacts', StaticFiles(directory=os.path.join('data', 'artifacts')), name='artifacts')

# Initialize SQLite schema
try:
    init_db()
except Exception:
    pass


def _wants_text(format: Optional[str], request: Optional[Request]) -> bool:
    if format == 'txt':
        return True
    try:
        if request and 'text/plain' in (request.headers.get('accept') or ''):
            return True
    except Exception:
        pass
    return False

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

class GitHubWebhookPayload(BaseModel):
    """Raw payload for GitHub webhook (we'll also use the request body directly)."""
    pass

class RunSummary(BaseModel):
    id: Optional[int] = None
    action: str
    tests: List[str] = []
    returncode: int = 0
    stdout: Optional[str] = None
    stderr: Optional[str] = None

@app.get('/')
def root():
    return {"status":"ok", "message":"AI Testing Agent Orchestrator"}

# Simple health/status endpoint
@app.get('/health')
def health(format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    mem_path = os.getenv('AGENT_MEMORY_PATH', 'data/memory.json')
    mem_runs = 0
    try:
        if os.path.exists(mem_path):
            with open(mem_path, 'r') as f:
                mem = json.load(f)
                mem_runs = len(mem.get('runs', []))
    except Exception:
        pass
    if _wants_text(format, request):
        loaded = 'yes' if model is not None else 'no'
        feat_count = (len(model_meta.get('feature_columns', [])) if isinstance(model_meta, dict) else 0)
        return PlainTextResponse(f"Health: OK. Model loaded: {loaded}. Feature columns: {feat_count}. Logged runs: {mem_runs}.")
    else:
        return {
            'status': 'ok',
            'model_loaded': bool(model is not None),
            'feature_columns': len(model_meta.get('feature_columns', [])) if isinstance(model_meta, dict) else 0,
            'memory_runs': mem_runs
        }

# Defect predictor using loaded model or heuristic fallback
@app.post('/predict')
def predict(req: PredictRequest, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
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
                if _wants_text(format, request):
                    return PlainTextResponse(f"Predicted defect risk: {prob:.2%}. Based on trained model using features {list(explanation.get('features', {}).keys())}.")
                return {"risk": prob, "explanation": explanation}
            else:
                X = [req.features.get(k,0) for k in ['loc','complexity','churn','num_devs']]
                prob = float(model.predict_proba([X])[0][1])
                explanation = {"source":"legacy_model", "features": dict(zip(['loc','complexity','churn','num_devs'], X))}
                if _wants_text(format, request):
                    return PlainTextResponse(f"Predicted defect risk: {prob:.2%} (legacy model).")
                return {"risk": prob, "explanation": explanation}
        # fallback heuristic
        risk = min(0.95, (len(req.file) % 10) / 10 + 0.1)
        explanation = {"heuristic":"length_mod_10", "file_length": len(req.file)}
        if _wants_text(format, request):
            return PlainTextResponse(f"Heuristic risk estimate: {risk:.2%} (based on file path length mod 10).")
        return {"risk": risk, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate test: if OPENAI_KEY present, call OpenAI completion, else return template
@app.post('/generate_test')
def generate_test(req: GenerateTestRequest, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
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
            try:
                add_generated_test(test_id, rel)
            except Exception:
                pass
            if _wants_text(format, request):
                return PlainTextResponse(f"Generated Playwright test {test_id} and saved to {rel}. The test was created from the provided context.")
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
            try:
                add_generated_test(test_id, rel)
            except Exception:
                pass
            if _wants_text(format, request):
                return PlainTextResponse(f"Generated example Playwright test {test_id} and saved to {rel}.")
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

def _require_api_key(x_api_key: Optional[str]) -> None:
    # Enforce only when API_KEY is configured
    if getattr(settings, 'api_key', None):
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail='Invalid or missing API key')

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

def _save_artifacts(stdout: str, stderr: str) -> Dict[str, str]:
    """Persist minimal artifacts (stdout/stderr) to data/artifacts and return paths."""
    base = os.path.join('data', 'artifacts')
    os.makedirs(base, exist_ok=True)
    rid = uuid.uuid4().hex[:8]
    paths = {}
    try:
        out_name = f"run_{rid}.out.txt"
        err_name = f"run_{rid}.err.txt"
        out_p = os.path.join(base, out_name)
        err_p = os.path.join(base, err_name)
        with open(out_p, 'w', encoding='utf-8') as f:
            f.write(stdout or '')
        with open(err_p, 'w', encoding='utf-8') as f:
            f.write(stderr or '')
        paths = {
            'stdout_path': out_p,
            'stderr_path': err_p,
            'stdout_url': f"/artifacts/{out_name}",
            'stderr_url': f"/artifacts/{err_name}",
        }
    except Exception:
        pass
    return paths

def _run_tests_core(tests: List[str], self_heal: bool = False) -> Dict:
    """Core implementation for running tests with optional self-heal, returns JSON dict."""
    result, cwd, err = _run_playwright(tests)
    if err:
        return err
    response = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    # Save basic artifacts
    art_paths = _save_artifacts(result.stdout, result.stderr)
    response.update(art_paths)
    try:
        from src.ai.memory.execution_memory import append_run
        append_run({'action': 'run_tests', 'tests': tests, 'result': {'run_output': result.stdout, 'returncode': result.returncode}})
    except Exception:
        pass
    try:
        log_run('run_tests', tests, result.returncode, result.stdout, result.stderr)
        # best-effort link artifacts to last run id (simple: query recent id)
        try:
            from src.api.persistence import _conn
            con = _conn()
            try:
                cur = con.execute('SELECT MAX(id) FROM runs')
                rid = cur.fetchone()[0]
            finally:
                con.close()
            if rid and art_paths:
                for k, p in art_paths.items():
                    try:
                        add_artifact(int(rid), k, p)
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass
    if result.returncode != 0 and self_heal and tests:
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
                test_path = tests[0]
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
                    sh_art = _save_artifacts(result2.stdout, result2.stderr)
                    response['self_heal'].update(sh_art)
                    # persist artifacts link
                    try:
                        from src.api.persistence import _conn
                        con = _conn()
                        try:
                            cur = con.execute('SELECT MAX(id) FROM runs')
                            rid = cur.fetchone()[0]
                        finally:
                            con.close()
                        if rid and sh_art:
                            for k, p in sh_art.items():
                                try:
                                    add_artifact(int(rid), f"self_heal_{k}", p)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        from src.ai.memory.execution_memory import append_run
                        append_run({'action': 'self_heal', 'tests': [test_path], 'result': {'run_output': result2.stdout, 'returncode': result2.returncode}})
                    except Exception:
                        pass
                    try:
                        log_run('self_heal', [test_path], result2.returncode, result2.stdout, result2.stderr)
                    except Exception:
                        pass
                else:
                    response['self_heal'] = {'attempted': True, 'reason': 'patch_failed'}
            else:
                response['self_heal'] = {'attempted': True, 'reason': 'no_suggestions'}
    return response

# Run tests locally in the playwright folder using npx, with optional self-heal
@app.post('/run_tests')
def run_tests(req: RunTestRequest, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    try:
        res = _run_tests_core(req.tests or [], bool(req.self_heal))
        if _wants_text(format, request):
            status = 'passed' if res.get('returncode') == 0 else 'failed'
            base = f"Executed Playwright tests ({', '.join(req.tests) if req.tests else 'all'}) -> {status}."
            if isinstance(res.get('self_heal'), dict):
                sh = res['self_heal']
                if sh.get('attempted') and 'new_selector' in sh:
                    base += f" Self-heal applied: '{sh.get('old_selector')}' -> '{sh.get('new_selector')}'."
                elif sh.get('attempted'):
                    base += f" Self-heal attempted: {sh.get('reason','n/a')}."
            tail = (res.get('stdout') or '')
            tail = tail[-400:] if len(tail) > 400 else tail
            return PlainTextResponse(base + ("\n--- Output (tail) ---\n" + tail if tail else ""))
        return res
    except Exception as e:
        if _wants_text(format, request):
            return PlainTextResponse(f"Error running tests: {str(e)}")
        return {"error": str(e)}

@app.get('/runs')
def get_runs(limit: int = 20, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    try:
        rows = recent_runs(limit)
        if _wants_text(format, request):
            if not rows:
                return PlainTextResponse('No runs logged yet.')
            lines = []
            for r in rows:
                lines.append(f"#{r['id']} {r['created_at']} {r['action']} rc={r['returncode']} tests={len(r['tests'])} artifacts={r['artifacts']}")
            return PlainTextResponse("Recent runs:\n" + "\n".join(lines))
        return {'runs': rows}
    except Exception as e:
        if _wants_text(format, request):
            return PlainTextResponse(f"Error listing runs: {str(e)}")
        return {'error': str(e)}
@app.post('/webhooks/github')
async def github_webhook(request: Request, x_hub_signature_256: Optional[str] = Header(default=None)):
    """Handle GitHub PR events to trigger plan: predict -> generate -> run -> comment."""
    raw = await request.body()
    try:
        from src.tools.github_client import verify_signature, get_pr_files, pr_comment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub client missing: {e}")
    if not verify_signature(raw, x_hub_signature_256):
        raise HTTPException(status_code=401, detail='Invalid signature')
    event = request.headers.get('x-github-event', '')
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    # Only handle pull_request events
    if event != 'pull_request':
        return {'status': 'ignored', 'reason': f'event {event} not supported'}
    action = payload.get('action')
    if action not in {'opened', 'synchronize', 'reopened'}:
        return {'status': 'ignored', 'reason': f'action {action}'}
    pr = payload.get('pull_request', {})
    pr_number = pr.get('number') or payload.get('number')
    repo_full = payload.get('repository', {}).get('full_name')
    if not pr_number:
        return {'status': 'ignored', 'reason': 'no pr number'}
    # Get changed files
    files = []
    try:
        files = get_pr_files(repo_full, int(pr_number))
    except Exception:
        pass
    changed_paths = [f.get('filename') for f in files if isinstance(f, dict) and f.get('filename')]
    # Simple plan: generate and run a basic smoke test for each frontend path change
    generated_ids: List[str] = []
    for path in changed_paths[:3]:  # limit
        if not (str(path).endswith('.tsx') or str(path).endswith('.ts') or str(path).endswith('.js') or str(path).endswith('.jsx')):
            continue
        try:
            gen = generate_test(GenerateTestRequest(source={'type':'repo_change','file': path}))
            if isinstance(gen, dict) and gen.get('path'):
                generated_ids.append(gen['path'])
        except Exception:
            continue
    # Run tests (generated ones only)
    run_res = None
    if generated_ids:
        try:
            run_res = _run_tests_core(generated_ids, self_heal=True)
        except Exception as e:
            run_res = {'error': str(e)}
    # Comment back on PR
    try:
        summary = "No tests generated."
        if generated_ids:
            if isinstance(run_res, dict) and 'returncode' in run_res:
                status = 'passed' if run_res.get('returncode') == 0 else 'failed'
                out_tail = (run_res.get('stdout','') or '')
                out_tail = out_tail[-400:] if len(out_tail) > 400 else out_tail
                summary = f"AI Test Agent: generated {len(generated_ids)} test(s) and {status}.\n\nOutput (tail):\n```${out_tail}```"
            else:
                summary = f"AI Test Agent: generated {len(generated_ids)} test(s). Run result unavailable."
        pr_comment(repo_full, int(pr_number), summary)
    except Exception:
        pass
    return {'status': 'ok', 'generated': generated_ids, 'run': run_res}

@app.post('/agent')
def agent_run(payload: Dict, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
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
                rel = os.path.relpath(test_path, _playwright_cwd())
                run_result, _, _ = _run_playwright([rel])
                result['run_output'] = run_result.stdout
                try:
                    add_generated_test(test_name, rel)
                    log_run('agent_run', [rel], run_result.returncode, run_result.stdout, run_result.stderr)
                except Exception:
                    pass
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
        if _wants_text(format, request):
            if isinstance(result, dict):
                if 'code' in result:
                    msg = 'Agent generated a test and saved it.'
                    if 'test_id' in result:
                        msg = f"Agent generated test {result['test_id']} and saved it."
                    if 'run_output' in result:
                        msg += " The test was executed; see dashboard for full logs."
                    return PlainTextResponse(msg)
                if 'message' in result:
                    return PlainTextResponse(str(result['message']))
            return PlainTextResponse('Agent run completed.')
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}

@app.get('/tests_generated')
def list_generated_tests(format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    """List AI-generated tests under Playwright tests/generated directory."""
    gen_dir = os.path.join(_playwright_cwd(), 'tests', 'generated')
    files = []
    if os.path.isdir(gen_dir):
        for root, _, filenames in os.walk(gen_dir):
            for n in filenames:
                if n.endswith('.spec.ts'):
                    files.append(os.path.relpath(os.path.join(root, n), _playwright_cwd()))
    if _wants_text(format, request):
        if not files:
            return PlainTextResponse('No AI-generated tests found.')
        return PlainTextResponse("Generated tests:\n" + "\n".join(sorted(files)))
    return {"files": sorted(files)}

# JIRA integration endpoints
@app.post('/jira/search')
def jira_search(req: JiraSearchRequest, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    try:
        from src.tools.jira_client import search_issues
        data = search_issues(req.jql, req.max_results or 20)
        if _wants_text(format, request):
            issues = data.get('issues', []) if isinstance(data, dict) else []
            lines = [f"{i.get('key')}: {i.get('fields',{}).get('summary','(no summary)')}" for i in issues]
            return PlainTextResponse("Jira search results:\n" + ("\n".join(lines) if lines else 'No issues found.'))
        return data
    except Exception as e:
        if _wants_text(format, request):
            return PlainTextResponse(f"Jira search error: {str(e)}")
        return {"error": str(e)}

@app.get('/jira/issue/{key}')
def jira_issue(key: str, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    try:
        from src.tools.jira_client import get_issue
        data = get_issue(key)
        if _wants_text(format, request):
            if isinstance(data, dict):
                summary = data.get('fields',{}).get('summary','')
                status = data.get('fields',{}).get('status',{}).get('name','')
                return PlainTextResponse(f"{key}: {summary} (status: {status})")
        return data
    except Exception as e:
        if _wants_text(format, request):
            return PlainTextResponse(f"Jira fetch error: {str(e)}")
        return {"error": str(e)}

@app.post('/jira/link_test_run')
def jira_link_test_run(req: JiraLinkRunRequest, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    """Attach a test run result as a comment to a JIRA issue."""
    try:
        from src.tools.jira_client import add_comment
        summary = req.result.get('summary') or ''
        rc = req.result.get('returncode')
        comment = f"Automated test run:\nReturn code: {rc}\nSummary: {summary}\n```${req.result.get('stdout','')[:2000]}```"
        res = add_comment(req.issue_key, comment)
        if _wants_text(format, request):
            return PlainTextResponse(f"Attached test run details to Jira issue {req.issue_key}.")
        return res
    except Exception as e:
        if _wants_text(format, request):
            return PlainTextResponse(f"Jira link error: {str(e)}")
        return {"error": str(e)}

# Simple self-heal endpoint: suggest alternate selectors when selector not found
@app.post('/self_heal')
def self_heal(payload: Dict, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    try:
        error = payload.get('error','')
        dom = payload.get('dom','')
        suggestions = []
        if 'selector' in error.lower() or 'no node found' in error.lower():
            # naive suggestion: try using text-based selector if label present
            if 'login' in dom.lower():
                suggestions.append({"selector": "text=Login", "reason":"text fallback contains 'Login'"})
            suggestions.append({"selector": "css=.btn-primary", "reason":"common primary button class fallback"})
        if _wants_text(format, request):
            if suggestions:
                first = suggestions[0]
                msg = f"Suggest trying selector '{first['selector']}' ({first['reason']})."
            else:
                msg = 'No self-heal suggestions available for this error.'
            return PlainTextResponse(msg)
        return {"fixed": bool(suggestions), "suggestions": suggestions}
    except Exception as e:
        if _wants_text(format, request):
            return PlainTextResponse(f"Self-heal error: {str(e)}")
        return {"error": str(e)}

@app.post('/explain')
def explain(req: PredictRequest, format: Optional[str] = None, request: Request = None, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    try:
        # require model_meta with feature_columns
        if model_meta is None or 'feature_columns' not in model_meta:
            raise Exception('Explainability requires a model trained with feature columns. Train KC1 model first.')
        from src.models.explain_shap import explain_instance
        data = explain_instance(req.features)
        if _wants_text(format, request):
            vals = data.get('shap_values') if isinstance(data, dict) else None
            if isinstance(vals, dict):
                # Sort top contributions
                items = sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                parts = [f"{k}: {v:+.4f}" for k, v in items]
                return PlainTextResponse("Top feature contributions (SHAP): " + ", ".join(parts))
        return data
    except Exception as e:
        traceback.print_exc()
        if _wants_text(format, request):
            return PlainTextResponse(f"Explainability error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket('/ws/run')
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        # Parse query args
        params = ws.query_params if hasattr(ws, 'query_params') else {}
        tests_str = params.get('tests') if hasattr(params, 'get') else None
        tests = [t.strip() for t in (tests_str or '').split(',') if t.strip()] if tests_str else []
        npx = shutil.which('npx') or shutil.which('npx.cmd')
        if not npx:
            await ws.send_text('npx not found. Install Node.js or use the Playwright container.')
            await ws.close()
            return
        cwd = _playwright_cwd()
        cmd = [npx, 'playwright', 'test', *tests, '--reporter=list'] if tests else [npx, 'playwright', 'test', '--reporter=list']
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await ws.send_text(f"Started: {' '.join(cmd)} (cwd={cwd})")
        assert proc.stdout is not None
        async for line in proc.stdout:
            try:
                await ws.send_text(line.decode(errors='ignore').rstrip())
            except WebSocketDisconnect:
                break
        rc = await proc.wait()
        await ws.send_text(f"[done] returncode={rc}")
        await ws.close()
    except Exception as e:
        try:
            await ws.send_text(f"[error] {str(e)}")
        finally:
            await ws.close()
