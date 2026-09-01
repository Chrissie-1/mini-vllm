.DEFAULT_GOAL := help
.PHONY: help venv install proto proto-py proto-go test test-worker test-gateway \
        lint bench bench-kv bench-throughput run-worker run-http run-gateway \
        up down logs ps clean

PY      ?= python
VENV    ?= venv
BIN     := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN     := $(VENV)/Scripts
endif

# torch spawns one OpenMP thread per core by default, which on a small machine
# costs more in contention than it gains. Every target inherits this.
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export TORCH_THREADS ?= 1
export HF_HUB_DISABLE_PROGRESS_BARS ?= 1

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: venv ## Install worker dependencies (including dev tools)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r worker/requirements-dev.txt

proto: proto-py proto-go ## Regenerate all gRPC bindings

proto-py: ## Regenerate the Python bindings
	$(BIN)/python -m grpc_tools.protoc -I proto \
		--python_out=worker/server/pb \
		--grpc_python_out=worker/server/pb \
		--pyi_out=worker/server/pb \
		proto/inference.proto
	@# protoc emits a flat import; rewrite it to a package-relative one.
	sed -i 's/^import inference_pb2 as/from . import inference_pb2 as/' \
		worker/server/pb/inference_pb2_grpc.py

proto-go: ## Regenerate the Go bindings (needs protoc + protoc-gen-go)
	protoc -I proto \
		--go_out=gateway/internal/pb --go_opt=paths=source_relative \
		--go-grpc_out=gateway/internal/pb --go-grpc_opt=paths=source_relative \
		proto/inference.proto

test: test-worker test-gateway ## Run every test

test-worker: ## Run the Python test suite
	cd worker && ../$(BIN)/python -m pytest -q

test-gateway: ## Run the Go test suite
	cd gateway && go test -race ./...

lint: ## Vet the Go code
	cd gateway && go vet ./...

bench: bench-kv bench-throughput ## Run every benchmark

bench-kv: ## Cached vs uncached decoding
	$(BIN)/python bench/bench_kv_cache.py --tokens 16,32,64,128

bench-throughput: ## Throughput across batch-size limits
	$(BIN)/python bench/bench_throughput.py --concurrency 16 --tokens 32 --batches 1,2,4,8,16

run-worker: ## Run the gRPC worker locally
	cd worker && ../$(BIN)/python -m server.worker

run-http: ## Run the FastAPI worker locally (no gateway needed)
	cd worker && ../$(BIN)/python -m uvicorn server.api:app --host 0.0.0.0 --port 8000

run-gateway: ## Run the Go gateway locally against a worker on :50051
	cd gateway && go run ./cmd/server -workers localhost:50051

up: ## Start the full stack (gateway, 2 workers, Prometheus, Grafana)
	docker compose up -d --build
	@echo "gateway    http://localhost:8080"
	@echo "prometheus http://localhost:9090"
	@echo "grafana    http://localhost:3000 (admin/admin)"

down: ## Stop the stack and remove volumes
	docker compose down -v

logs: ## Follow stack logs
	docker compose logs -f

ps: ## Show stack status
	docker compose ps

clean: ## Remove build and test artefacts
	rm -rf .pytest_cache worker/.pytest_cache worker/**/__pycache__ \
		worker/__pycache__ gateway/internal/pb/*.pb.go
