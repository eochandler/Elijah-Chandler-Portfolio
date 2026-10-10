# Turbofan RUL MLOps Project: Copilot Build Spec
How to use this doc: save it in your repo as `docs/PROJECT_SPEC.md`. In Copilot Chat (agent mode if available), attach it with `#file:docs/PROJECT_SPEC.md` and paste one phase prompt from the last section at a time. Review and run each phase before starting the next.

## 1. Goal
Build a small, production-style MLOps project that predicts Remaining Useful Life (RUL) of turbofan engines from sensor data using the NASA C-MAPSS dataset.

The finished repo must demonstrate:

- Reproducible data prep and training
- Experiment tracking and a model registry with MLflow
- A FastAPI prediction service with validated inputs
- A Docker image that builds and runs anywhere
- Automated tests and CI
- A free-tier deployment with a public URL
- A README that tells the story to a hiring manager
Scope rule: start with subset FD001 only. Extending to FD002 to FD004 is a stretch goal, listed in section 12.

## 2. Tech Stack and Constraints

- Python 3.11
- pandas, NumPy, scikit-learn, LightGBM (primary model)
- MLflow (tracking and model registry, local file or SQLite backend)
- FastAPI, Uvicorn, Pydantic v2
- pytest, httpx (API tests), ruff (lint and format)
- Docker (multi-stage build), docker compose
- GitHub Actions for CI
- Optional stretch: PyTorch 1D-CNN or LSTM. Keep PyTorch out of the serving image unless the deep model is the one being served.
Constraints:

- No credit card or paid service required for any core step
- Everything runs on macOS (including Apple Silicon) and Linux
- Pin dependency versions in `requirements.txt` (training) and `requirements-serve.txt` (serving only, smaller)
- Use type hints, docstrings on public functions, and no hard-coded absolute paths
- All configuration (paths, hyperparameters, thresholds) lives in `configs/config.yaml`, not scattered in code
- Set random seeds for reproducibility

## 3. Repository Structure

```
turbofan-rul/
├── .github/workflows/ci.yml
├── configs/config.yaml
├── data/
│   ├── raw/            # user places CMAPSS files here (git-ignored)
│   └── processed/      # generated (git-ignored)
├── docs/PROJECT_SPEC.md
├── models/             # exported serving model (small; may be committed)
├── notebooks/01_eda.ipynb
├── src/turbofan/
│   ├── __init__.py
│   ├── config.py       # loads config.yaml
│   ├── data.py         # load raw files, add RUL labels
│   ├── features.py     # feature engineering (shared by training AND serving)
│   ├── metrics.py      # RMSE and NASA score
│   ├── train.py        # training entry point, logs to MLflow
│   ├── evaluate.py     # evaluation on the official test set
│   └── export.py       # export registered model to models/
├── app/
│   ├── main.py         # FastAPI app
│   ├── schemas.py      # Pydantic request/response models
│   └── model_loader.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── Makefile
├── requirements.txt
├── requirements-serve.txt
└── README.md
```
Important design rule: `features.py` is the single source of truth for feature engineering and is imported by both `train.py` and the API. Training/serving skew is a classic MLOps bug, and avoiding it is part of what this project demonstrates.

## 4. Dataset: NASA C-MAPSS
The user downloads the data manually (do NOT write code that auto-downloads from a URL). Place the files in `data/raw/`. The README must explain where to get it: search for "NASA C-MAPSS Turbofan Engine Degradation Simulation" on the NASA Prognostics Data Repository (a Kaggle mirror also exists).

Files used for FD001:

- `train_FD001.txt`: 100 engines run to failure
- `test_FD001.txt`: 100 engines, trajectories cut off before failure
- `RUL_FD001.txt`: true RUL at the last recorded cycle of each test engine
Format: space-separated, no header, 26 columns:

1. `unit` (engine id)
2. `cycle` (time step)
3. `op_setting_1`, `op_setting_2`, `op_setting_3`
4. `sensor_1` to `sensor_21`
Handle trailing whitespace in the files (it creates empty extra columns if read naively).

Labels:

- For training engines, `RUL = max_cycle_of_that_engine - cycle`
- Clip the training RUL at 125 (piecewise linear target). Make the clip value a config option. Explain in a code comment that early-life degradation is not observable, so a capped target trains better.
- For test engines, evaluate ONLY on the last cycle of each engine against `RUL_FD001.txt`.

## 5. Data Preparation and Features
Cleaning:

- Drop sensors that are constant or near-constant in FD001: sensors 1, 5, 6, 10, 16, 18, 19. Compute this from the training data (variance threshold) and save the kept-column list to config or a JSON artifact, so serving uses exactly the same columns.
- Drop the operating settings for FD001 (single operating condition), but keep the code path configurable.
- Fit any scaler on training data only, and persist it as an artifact.
Feature engineering (per engine, using only past and present cycles, never future):

- Raw kept sensors
- Rolling mean and rolling std over windows 5, 10, and 30
- Difference from the engine's first-N-cycle baseline (N configurable, default 10)
- Cycle number
Split rules (leakage prevention):

- Split train and validation BY ENGINE (GroupKFold or GroupShuffleSplit on `unit`), never by random rows
- Never let rolling windows cross engine boundaries

## 6. Modeling
Baselines first, then improve. Log every run to MLflow.

1. Naive baseline: predict the mean training RUL
2. Ridge regression on engineered features
3. LightGBM regressor (primary model), with a small random or Optuna-free grid search over `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`
Stretch: PyTorch 1D-CNN or LSTM on sliding windows of 30 cycles. If trained, compare against LightGBM in MLflow.

Metrics (implement in `metrics.py` with unit tests):

- RMSE
- NASA scoring function. With `d = predicted - true`: score = sum over engines of `exp(-d/13) - 1` when `d < 0` (early prediction) and `exp(d/10) - 1` when `d >= 0` (late prediction). Late predictions are penalized more because a missed failure costs more than an early warning. Lower is better.
For reference, published results on FD001 vary roughly from 11 to 20 RMSE depending on the method. A solid tree-based baseline landing in the mid-teens is a good outcome. Do not over-tune to chase a leaderboard, and do not tune on the test set.

Also produce:

- A plot of predicted vs true RUL for the test engines (saved as an MLflow artifact)
- A feature importance plot for LightGBM
- A short note on the false-alarm versus missed-failure tradeoff, written in the README

## 7. MLflow Tracking and Registry

- Use a local backend: `sqlite:///mlflow.db` with an `mlruns/` artifact folder (both git-ignored)
- One experiment named `turbofan-rul-fd001`
- Each run logs: all hyperparameters, config values (clip, windows, dropped sensors), data subset name, git commit hash if available, validation RMSE, test RMSE, test NASA score, plots, and the fitted model with an input example and signature
- Register the best model as `turbofan-rul` and assign the alias `champion`
- `export.py` loads the `champion` model and writes the serving artifacts into `models/` (model file, scaler, kept-columns JSON, feature config)
- Provide `make mlflow-ui` to start the UI on port 5000

## 8. FastAPI Service
Endpoints:

- `GET /health` returns `{"status": "ok"}`
- `GET /model-info` returns model name, version, training date, FD001 test RMSE, expected input columns
- `POST /predict` accepts the recent cycle history for ONE engine and returns the predicted RUL
Request body (Pydantic schema):

```
{
  "engine_id": "string, optional",
  "cycles": [
    {"cycle": 1, "sensor_2": 641.8, "sensor_3": 1589.7, "...": "..."}
  ]
}
```
Rules:

- Require at least 30 cycles of history (config value). Return HTTP 422 with a clear message otherwise.
- Response: `predicted_rul` (float, cycles), `alert` (bool, true when predicted RUL is below a configurable threshold, default 30), `model_version`
- Apply the same `features.py` pipeline as training
- Load the model once at startup (lifespan handler), not per request
- Add a `GET /docs` friendly example payload using Pydantic `json_schema_extra`
- Structured logging of each request (engine_id, number of cycles, prediction, latency) without logging full payloads
- Include a sample request file `examples/sample_request.json` generated from a real test engine

## 9. Docker
`Dockerfile` requirements:

- Multi-stage build on `python:3.11-slim`
- Install only `requirements-serve.txt` in the final image (no MLflow server, no training libraries, no PyTorch for the LightGBM model)
- Copy only `app/`, `src/turbofan/` (feature code), and `models/`
- Run as a non-root user
- `HEALTHCHECK` that calls `/health`
- Read the port from an environment variable `PORT`, default 8000
- Add `.dockerignore` excluding data, notebooks, mlruns, mlflow.db, tests, git files, and virtual environments
`docker-compose.yml`:

- Service `api`: the serving image
- Service `mlflow`: runs the MLflow tracking UI with a mounted volume (for local demos)
Apple Silicon note (put in the README): when building for a cloud host that runs x86 servers, use `docker buildx build --platform linux/amd64 ...`.

Target image size: under about 500 MB. Report the final size in the README.

## 10. Testing and CI
Tests (pytest):

- `test_data.py`: loader returns 26 columns, RUL label is correct and clipped, no NaNs
- `test_features.py`: rolling features never use future cycles; output columns match the saved column list
- `test_metrics.py`: NASA score and RMSE against hand-computed values, including asymmetry (late costs more than early)
- `test_api.py`: health check, valid prediction, too-few-cycles returns 422, missing sensor returns 422
CI (`.github/workflows/ci.yml`):

- On push and pull request: set up Python 3.11, install dependencies, run ruff, run pytest
- Build the Docker image to verify it builds
- On pushes to `main`: push the image to GitHub Container Registry (GHCR) using the built-in `GITHUB_TOKEN`
- Tests must not require the raw NASA data. Use a tiny synthetic fixture generated in the test setup.

## 11. Deployment (free tier)
Primary target: Hugging Face Spaces using the Docker SDK (free CPU tier, no credit card).

- Add the Space configuration front matter to a Space-specific README with `sdk: docker` and `app_port` set to match the container port (or set the container to port 7860)
- Provide `docs/DEPLOY.md` with step-by-step instructions
Alternatives to document: Render (free web service, sleeps when idle) and running the GHCR image locally. Note in `docs/DEPLOY.md` that free-tier terms change, so check each provider's current page, and that some providers require a card on file.

After deployment, add the public URL and a `curl` example using `examples/sample_request.json` to the README.

## 12. Stretch Goals (only after everything above works)

- Extend to FD002 to FD004: for multi-condition subsets, cluster the three operating settings (KMeans, 6 clusters) and standardize sensors within each cluster
- Add a PyTorch 1D-CNN or LSTM and compare it in MLflow
- Add a simple drift check: compare incoming feature means against training means and expose `GET /drift`
- Add Prometheus-style metrics endpoint or request counters
- Add a Streamlit or Gradio demo page that plots the predicted RUL for a sample engine
- Add a model card (`docs/MODEL_CARD.md`) covering intended use, data, limits, and failure modes

## 13. README Requirements
The README is part of the portfolio. It must contain:

1. One-paragraph problem statement and why predictive maintenance matters
2. Architecture diagram (a Mermaid diagram is fine): data to features to training to MLflow registry to export to Docker to FastAPI to deployment
3. Results table: model, validation RMSE, test RMSE, NASA score
4. Plot of predicted vs true RUL
5. Quickstart: commands to set up, train, run MLflow UI, run the API, build and run Docker, and call the endpoint
6. Design decisions: RUL clipping, engine-level splits, shared feature code, why LightGBM is served instead of a deep model
7. Limitations: simulated data, single fault mode in FD001, no real sensor drift, and what would change for production
8. Link to the live demo

## 14. Definition of Done

- `make train` reproduces the results from raw data with fixed seeds
- MLflow shows at least 5 comparable runs and a registered `champion` model
- `docker build` and `docker run -p 8000:8000` serve a working `/predict`
- `pytest` and `ruff` pass locally and in CI
- Image is under about 500 MB and runs as non-root
- Public deployment responds to the sample request
- README complete with results and architecture diagram
- No data files, secrets, or MLflow databases committed

## 15. Phase Prompts for Copilot
Paste one at a time. After each, run the code and fix errors before moving on.

**Phase 1: scaffold**

> Using `#file:docs/PROJECT_SPEC.md`, create the repository structure from section 3, a `.gitignore`, `requirements.txt` and `requirements-serve.txt` with pinned versions, a `Makefile` with targets `setup`, `train`, `evaluate`, `export`, `serve`, `test`, `lint`, `mlflow-ui`, `docker-build`, and `docker-run`, and `configs/config.yaml` with the values described in the spec. Do not implement logic yet.
**Phase 2: data and features**

> Implement `src/turbofan/data.py`, `features.py`, and `metrics.py` following sections 4, 5, and 6, with the tests in `tests/test_data.py`, `test_features.py`, and `test_metrics.py`. Use a small synthetic fixture so tests do not need the raw data.
**Phase 3: training with MLflow**

> Implement `train.py` and `evaluate.py` per sections 6 and 7: naive baseline, Ridge, and LightGBM with a small search, engine-grouped validation, full MLflow logging, plots as artifacts, and model registration with the `champion` alias.
**Phase 4: export and API**

> Implement `export.py` and the FastAPI app in `app/` per section 8, reusing `features.py`. Add `tests/test_api.py` and generate `examples/sample_request.json` from a real test engine.
**Phase 5: Docker**

> Write the `Dockerfile`, `.dockerignore`, and `docker-compose.yml` per section 9. Keep the image small, run as non-root, and add the healthcheck.
**Phase 6: CI and deployment**

> Write `.github/workflows/ci.yml` per section 10 and `docs/DEPLOY.md` plus the Hugging Face Space configuration per section 11.
**Phase 7: README**

> Write the README per section 13 using the actual results from MLflow. Leave placeholders only for the live demo URL.
**Tips for working with Copilot on this**

- Run `make test` after every phase, and ask Copilot to fix failures before continuing
- Ask it to explain any code you do not fully understand. You should be able to defend every design choice in an interview.
- Check that rolling features cannot see future cycles and that the split is by engine. These are the two mistakes interviewers probe, and AI-generated code gets them wrong more often than you would expect.
- Commit after each phase with a clear message so the git history shows steady progress
