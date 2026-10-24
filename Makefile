.PHONY: install data data-check test eval eval-dev eval-val clean
install:      ; pip install -r requirements.txt
# PMR-Bench at its pinned revision, then the dev_add / val / test carve
data:         ; python tools/fetch_pmr.py && python tools/split_extended.py
data-check:   ; python tools/fetch_pmr.py --check
test:         ; python -m pytest tests/ -q
eval:         ; python run_eval.py --data data/ext_test.csv
eval-dev:     ; python run_eval.py --data data/pmr_dev.csv data/ext_dev_add.csv --pmr-fallback ext_dev_add
eval-val:     ; python run_eval.py --data data/ext_val.csv
clean:        ; rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
