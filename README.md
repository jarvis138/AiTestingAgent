# AI Testing Agent - Starter (Mixed Tech) - Ready Edition

This starter repository provides a more complete local development environment for an
**agentic AI automation testing tool** combining Python and Node.js Playwright.

What's included:
- FastAPI orchestrator with:
  - `/predict` - defect prediction (loads a simple RandomForest model if trained)
  - `/generate_test` - uses OpenAI (if `OPENAI_API_KEY` provided) to generate Playwright tests
  - `/run_tests` - runs Playwright tests via the Node container
  - `/self_heal` - basic heuristic suggestions for locator fixes
- Playwright Node.js project that can run generated or example tests
- Streamlit dashboard for demoing predictions and test generation
- Docker Compose to run services locally
- Simple model training script to create a demo defect predictor

## Quickstart (local, Docker)
1. Install Docker & Docker Compose.
2. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` if you want LLM-driven test generation.
3. Build the model (optional but recommended):
   - Run: `docker-compose run --rm orchestrator python src/models/train_defect_model.py`
   - This will create `src/models/model.pkl` inside the container filesystem.
4. Start services:
   - `docker-compose up --build`
5. Visit:
   - FastAPI: http://localhost:8000
   - Streamlit Dashboard: http://localhost:8501

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
