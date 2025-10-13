# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help docs

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif

# Includes environment variables from the .env file
include .env

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

# Set the default repository URL to be used for setting up a remote machine
REPO ?= https://github.com/viggo-gascou/rsp.git

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


install: ## Install dependencies
	@echo "Installing the project..."
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment
	@$(MAKE) --quiet install-pre-commit
	@echo "Installed the project."

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
			echo "Installed uv."; \
		else \
			echo "Updating uv..."; \
			uv self update || true; \
	fi

install-dependencies:
	@uv python install 3.11
	@uv sync --no-dev --python 3.11

setup-environment:
	@uv run python src/scripts/setup_env.py

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

setup-remote: ## Setup environment on remote machine
	@if [ -z "$(REMOTE)" ]; then \
		echo "Error: REMOTE variable not set. Usage: make setup-remote REMOTE=user@host"; \
		exit 1; \
	fi
	@echo "Setting up remote machine: $(REMOTE)"
	@echo "Cloning repository: $(REPO)"
	@ssh $(REMOTE) "git clone $(REPO) ~/project 2>/dev/null || (cd ~/project && git pull)"
	@echo "Copying .env file..."
	@scp .env $(REMOTE):~/project/.env
	@echo "Running installation..."
	@ssh $(REMOTE) "cd ~/project && make install"
	@echo "Setup complete!"
