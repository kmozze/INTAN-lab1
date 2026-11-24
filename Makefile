PYTHON ?= python3
TECTONIC ?= /opt/homebrew/bin/tectonic
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: setup run report clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

run:
	$(VENV_PYTHON) src/main.py

report:
	mkdir -p report/build
	cd report && $(TECTONIC) --synctex --keep-logs --outdir build main.tex

clean:
	mkdir -p report/build
	find report/build -type f -delete
