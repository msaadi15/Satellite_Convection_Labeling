.PHONY: install test lint evaluate realtime serve docker
install: ; pip install -r requirements-dev.txt && pip install -e . --no-deps
test: ; pytest -q
lint: ; ruff check src tests
evaluate: ; convseg evaluate --limit 200 --min-dice 0.5 --out-csv outputs/eval.csv
realtime: ; convseg realtime
serve: ; convseg serve
docker: ; docker build -t convseg .
