# AI Testing Agent – Agentic Automation Testing Platform

Single-system overview (one line)

A cloud-hosted Agent Orchestrator (LangChain/LLM) coordinates a Defect Predictor, an AI Test Generator, a Test Runner, and a Self-Healing module — all triggered by PRs/commits and producing PR comments, Jira tickets, test runs, and a monitoring dashboard.

What's included (current):
- FastAPI orchestrator with:
  - `/predict` - defect prediction (loads a simple RandomForest model if trained)
  - `/generate_test` - uses OpenAI (if `OPENAI_API_KEY` provided) to generate Playwright tests
   - `/run_tests` - runs Playwright tests locally via npx (or use the Playwright container)
   - `/health` - quick status check (model loaded, memory stats)
  - `/self_heal` - basic heuristic suggestions for locator fixes
- Natural-language responses: add `?format=txt` or `Accept: text/plain` on most endpoints
- Optional API key auth: set `API_KEY` and send header `X-API-Key`
- Minimal persistence: SQLite logging of runs and generated tests (`data/agent.db`)
- `/runs` endpoint to list recent runs (with basic artifact counts)
- WebSocket live logs (MVP): `ws://localhost:8000/ws/run?tests=tests/example.spec.ts`
- Playwright Node.js project that can run generated or example tests
- React dashboard (dashboard/) for visualizing analytics and invoking endpoints
 - JIRA integration (optional): search issues, load details, and generate tests from issues
   - Jira descriptions and Acceptance Criteria are parsed into structured context for higher-quality generation.
- Docker Compose to run services locally
- Simple model training script to create a demo defect predictor

Planned/Production features (roadmap):
- GitHub PR webhook/Actions trigger and PR comment integration
- Ephemeral container test runs with artifacts (JUnit, screenshots, traces, DOM snapshot) and S3 upload
- Self-healing v2 (DOM-aware + LLM proposals with verification) and auto-PRs for fixes
- Observability: Prometheus metrics + Grafana dashboards (flakiness, risk heatmap)
- MLOps: retrain pipeline (ETL to S3, scheduled retrain, model registry/deploy)

## Quickstart (local, Docker)
1. Install Docker & Docker Compose.
2. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` if you want LLM-driven test generation.
3. Optional: Build the model for prediction/explainability
   - Install ML deps locally: `py -m pip install -r requirements-ml.txt`
   - Or inside Docker: `docker-compose run --rm orchestrator pip install -r requirements-ml.txt`
   - Then run: `docker-compose run --rm orchestrator python src/models/train_defect_model.py`
   - This will create `src/models/defect_model.pkl`.
4. Start services:
   - `docker-compose up --build`
5. Visit:
   - FastAPI: http://localhost:8000 (health: /health)
   - React Dashboard: cd dashboard && npm start (http://localhost:3000)
   - JIRA page: http://localhost:3000/jira (requires JIRA env config)

Optional security (local):
- To enforce simple API key auth, set `API_KEY` in your environment, then include header `X-API-Key: <value>` in requests.

Live logs (MVP):
- Connect a WebSocket client to `ws://localhost:8000/ws/run?tests=tests/example.spec.ts` to stream Playwright output during a run.

## JIRA Integration (optional)
Set these environment variables to enable JIRA (Cloud):

- JIRA_BASE_URL=https://your-domain.atlassian.net
- JIRA_EMAIL=you@example.com
- JIRA_API_TOKEN=your_api_token

Endpoints:
- POST /jira/search { jql, max_results? }
- GET /jira/issue/{key}
- POST /jira/link_test_run { issue_key, result }


## Notes
- This is a starter. For production, you should:
  - Use secure secrets management (AWS Secrets Manager / Vault)
  - Deploy model to SageMaker or a model server
  - Harden the OpenAI usage (prompt templates, safety, rate-limits)
  - Implement thorough sandboxing for running untrusted PR code
   - Add GitHub PR webhooks/Actions, PR comments, and artifact uploads (e.g., S3)
   - Add metrics (Prometheus) and dashboards (Grafana)
   - Move analytics from memory.json to DB-backed summaries



## Feature Extraction & Explainability

### Extracting features from a git repo
Use the feature extractor to compute LOC, complexity, churn, and number of developers per file:

```bash
docker-compose run --rm orchestrator python -c "from src.tools.feature_extractor import extract_basic_metrics; import json; print(json.dumps(extract_basic_metrics('/path/to/repo'), indent=2))"
```

### Explain a prediction with SHAP
After training the KC1 model (which saves `src/models/defect_model.pkl`), you can call `/explain` with the same features used in `/predict` to get SHAP values per feature.



## LangChain Agent

A LangChain-based agent has been added at `src/ai/agent.py`. It exposes a `/agent` endpoint in FastAPI to run free-form goals. It uses tools that call existing orchestrator endpoints (generate_test, predict, run_tests, self_heal). Configure `OPENAI_API_KEY` to enable LLM-driven planning.


## Core components (design)
- Event Trigger: GitHub webhook/Actions triggers on PR open/update. (Planned)
- Agent Orchestrator: LangChain/LLM plans steps and calls tools; stores per-PR state. (Basic agent present; planner roadmap)
- Defect Prediction Service: REST service returning risk + SHAP; can be backed by SageMaker. (Local model + SHAP supported)
- Test Script Generator: LLM reads PR diff/Jira AC/DOM/OpenAPI to produce Playwright tests and metadata. (Jira-enriched prompts implemented)
- Test Runner / Sandbox: Runs in ephemeral containers, captures artifacts, returns results. (Local npx runner; Docker service available; artifacts roadmap)
- Self-Healing Module: Heuristics + optional LLM; validate against DOM snapshot; propose patches/PRs. (Heuristic v1 implemented)
- Integrator / Notifier: PR comments, Jira issues, Slack alerts, dashboard updates. (Jira endpoints present; PR/Slack roadmap)
- Dashboard & Observability: React dashboard now; Prometheus/Grafana planned. (Roadmap)
- Retrain Pipeline: ETL to S3 and scheduled retrain; model registry/deploy. (Roadmap)

## High-level data flow
1. PR → GitHub webhook → Agent Orchestrator (plans).
2. Orchestrator → Defect Predictor for changed files.
3. If risk high or missing tests → Test Script Generator.
4. Generated tests → validate (lint/dry-run) → push to sandbox or open PR.
5. Run tests → on failure → Self-Healing → if validated, open PR/suggest fix.
6. Post results/artifacts to PR & Jira; update dashboard.
7. Save labels/results to storage (S3/DB) for retraining.

## GitHub Webhook (PR) [MVP]

Configure environment:

```
GITHUB_WEBHOOK_SECRET=your_secret
GITHUB_TOKEN=ghp_xxx
GITHUB_REPO=owner/repo
```

Add a webhook in your GitHub repo pointing to POST /webhooks/github with content type application/json and the same secret. On PR opened/updated, the agent will attempt to generate simple smoke tests for changed TS/JS files and run them, then comment a short summary on the PR.

## Project structure: current vs. production
- Current: FastAPI app (`src/api/main.py`), tools (`src/tools`), agent (`src/ai`), models (`src/models`), Playwright project (`playwright/`), dashboard (`dashboard/`).
- Production target: split API routes/schemas, add plugins (Slack/Jira/GitHub PR), monitoring, infra (Terraform/Helm), CI/CD workflows, and MLOps pipeline. See docs/architecture.md and roadmap.md.
