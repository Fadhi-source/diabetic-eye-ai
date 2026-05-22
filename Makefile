.PHONY: lint test smoke verify

lint:
	ruff check . --fix

format:
	ruff format .

test:
	python -m pytest tests/ -v --tb=short

smoke:
	python -m training.train --smoke_test

verify: lint test smoke
