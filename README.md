# AI Testing Agent - Starter (Mixed Tech) - Ready Edition

This starter repository provides a more complete local development environment for an
**agentic AI automation testing tool** combining Python and Node.js Playwright.

What's included:
- FastAPI orchestrator with:
  - `/predict` - defect prediction (loads a simple RandomForest model if trained)
  - `/generate_test` - uses OpenAI (if `OPENAI_API_KEY` provided) to generate Playwright tests
   - `/run_tests` - runs Playwright tests locally via npx (or use the Playwright container)
   - `/health` - quick status check (model loaded, memory stats)
  - `/self_heal` - basic heuristic suggestions for locator fixes
- Playwright Node.js project that can run generated or example tests
- React dashboard (dashboard/) for visualizing analytics and invoking endpoints
 - JIRA integration (optional): search issues, load details, and generate tests from issues
- Docker Compose to run services locally
- Simple model training script to create a demo defect predictor

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
