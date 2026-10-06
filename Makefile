# emo-sis automation with uv and npm

.PHONY: help setup sync test-infer serve-ai serve-backend migrate clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Initialize Python venv with uv and install backend packages"
	@echo "  make sync           - Synchronize Python environment with uv"
	@echo "  make serve-ai       - Start FastAPI Multimodal Fusion service (port 8000)"
	@echo "  make serve-backend  - Start Node/Express + Socket.IO backend (port 5000)"
	@echo "  make migrate        - Run MySQL schema migrations"
	@echo "  make test-audio     - Test audio emotion inference on sample.wav"
	@echo "  make test-facial    - Test facial expression inference on sample.jpg"

setup:
	uv venv --python 3.12 .venv
	uv sync --all-extras
	cd backend && npm install

sync:
	uv sync --all-extras

serve-ai:
	uv run uvicorn ai.fusion.service:app --reload --port 8000

serve-backend:
	npm --prefix backend run dev

migrate:
	npm --prefix backend run migrate

test-audio:
	uv run python -m ai.audio.infer sample_test.wav

test-facial:
	uv run python -m ai.facial.infer sample_test.jpg

clean:
	rm -rf .pytest_cache .coverage htmlcov
