.PHONY: install data test all clean run-sanity

PYTHON ?= python3
PIP ?= pip3

install:
	$(PIP) install -r requirements.txt

data:
	PYTHONPATH=src $(PYTHON) -m faultlens.generator --generate-split --count 300 --out-dir data/sample_scenarios

test:
	PYTHONPATH=src pytest -v tests/

sanity:
	PYTHONPATH=src $(PYTHON) -m faultlens.diagnoser --sanity-check

all: install data test
