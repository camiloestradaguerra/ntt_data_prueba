PYTHON := python
RAW_DATA := data/01_raw/coffee_db.parquet
PROCESSED_DATA := data/02_processed/data_long.parquet
FEATURES_DATA := data/03_features/data_features.parquet
FEATURE_ARTIFACTS := data/03_features/artifacts
MODEL_PATH := data/04_models/lgbm_model.joblib
MODEL_ARTIFACTS := data/04_models/artifacts
EVAL_DIR := data/05_evaluation

.PHONY: all setup etl features train evaluate run-api run-agent test-agent tunnel clean

all: setup etl features train evaluate

setup:
	@$(PYTHON) -c "import os; [os.makedirs(d, exist_ok=True) for d in ['data/01_raw', 'data/02_processed', 'data/03_features/artifacts', 'data/04_models/artifacts', 'data/05_evaluation']]"

etl:
	$(PYTHON) src/pipelines/1_data_etl/main.py --input_path $(RAW_DATA) --output_path $(PROCESSED_DATA)

features:
	$(PYTHON) src/pipelines/2_feature_engineering/main.py --input_path $(PROCESSED_DATA) --output_path $(FEATURES_DATA) --artifacts_dir $(FEATURE_ARTIFACTS)

train:
	$(PYTHON) src/pipelines/3_training/main.py --input_path $(FEATURES_DATA) --input_artifacts_dir $(FEATURE_ARTIFACTS) --output_model_path $(MODEL_PATH) --artifacts_dir $(MODEL_ARTIFACTS) --n_trials 15

evaluate:
	$(PYTHON) src/pipelines/4_evaluation/main.py --input_path $(FEATURES_DATA) --model_path $(MODEL_PATH) --input_artifacts_dir $(FEATURE_ARTIFACTS) --output_dir $(EVAL_DIR)

# Comandos de ejecución para desarrollo local y agente
run-api:
	uvicorn src.api.main:app --reload --port 8000

run-agent:
	$(PYTHON) -m src.agent.cli_chat

test-agent:
	$(PYTHON) -c "from src.agent.bot import analyze_market; print('✅ Módulo de agente e InferenceClient importados correctamente')"

tunnel:
	ngrok http 8000

clean:
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) if p.is_dir() else p.unlink() for folder in pathlib.Path('data').glob('*') if folder.name != '01_raw' for p in folder.glob('*')]"