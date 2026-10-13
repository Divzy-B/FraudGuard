# Operations guide

## Local workflow

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m fraudguard.train --rows 12000 --seed 2026 --output artifacts/fraudguard.joblib
python -m uvicorn fraudguard.api:app --reload
```

Browse `http://127.0.0.1:8000/docs` for the interactive API schema. The
default model location is `artifacts/fraudguard.joblib`; override it with
`FRAUDGUARD_MODEL_PATH` when serving a known artifact.

## Container workflow

```bash
docker compose up --build
curl http://localhost:8000/health
```

The image trains a synthetic demo model during build and runs as a non-root
user. Keep production model artifacts out of images and inject them through a
controlled model-delivery process instead.

## Reproducibility

The generator accepts `--seed`; training parameters and the threshold are
stored in the artifact. To compare runs, record the commit SHA, Python and
package versions, generator seed, training-row count, review capacity, and
artifact metrics. Do not compare models by accuracy alone.
