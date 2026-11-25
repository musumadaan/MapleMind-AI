PY ?= python
export PYTHONPATH := src

install:        ## core deps (offline demo)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .
ingest:         ## build vector index
	$(PY) -m maplemind.cli ingest
demo:           ## scripted demo scenarios
	bash scripts/demo.sh
chat:
	$(PY) -m maplemind.cli chat --debug
checkin:
	$(PY) -m maplemind.cli chat --workflow static --debug
test:
	$(PY) -m pytest -q tests
eval:           ## end-to-end harness + context comparison + coverage sweep
	$(PY) eval/harness.py --label default
	$(PY) eval/compare_context.py
	$(PY) eval/harness.py --sweep retrieval.min_query_coverage=0,0.34,0.5 --label cov
sft-data:
	$(PY) finetune/build_sft_dataset.py --mentalchat 400
train:          ## needs GPU + finetune/requirements-train.txt
	$(PY) finetune/train_lora.py --config finetune/configs/qwen25_1p5b_qlora.yaml
matrix:         ## base vs fine-tuned vs alternative, with/without RAG
	$(PY) eval/model_matrix.py --models base_qwen25_1.5b ft_qwen25_1.5b alt_llama32_3b
api:
	uvicorn maplemind.api:app --reload

robustness:     ## five-axis safety suite
	$(PY) eval/robustness.py
fixtures:       ## record real pipeline responses for the web demo mode
	$(PY) web/scripts/make_fixtures.py
web-install:
	cd web/client && npm install && cd ../server && npm install
web-build:
	cd web/client && npm run build
web:            ## Node gateway + built React app on :8787
	cd web/server && npm start
sft-smoke:      ## CPU, no downloads: tiny Qwen2 + LoRA training + adapter evaluation (pipeline verification)
	$(PY) finetune/make_tiny_base.py
	$(PY) finetune/train_lora.py --base-model finetune/outputs/tiny-qwen2-base --output-dir finetune/outputs/tiny-qwen2-lora-balanced --epochs 30 --lr 3e-3
	$(PY) eval/model_matrix.py --models mock tiny_base tiny_ft_bal
