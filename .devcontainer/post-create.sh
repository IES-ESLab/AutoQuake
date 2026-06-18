#!/bin/bash
set -e

echo "🚀 Setting up AutoQuake development environment..."

# Ensure conda is properly initialized
source /opt/conda/etc/profile.d/conda.sh

# Activate the AutoQuake environment
conda activate AutoQuake_v0
echo "📦 Active conda environment: $CONDA_DEFAULT_ENV"

# Dev-only tooling (kept out of the conda env so it stays lean).
# Ruff replaces black + isort + flake8; pre-commit runs it on every commit.
echo "🔧 Installing development tools (ruff, pre-commit)..."
pip install --no-cache-dir ruff pre-commit

# Initialize git submodules (EQNet, GaMMA) so the full pipeline can import them.
if [ -f ".gitmodules" ]; then
    echo "🔗 Initializing git submodules..."
    git submodule update --init --recursive
fi

# Install pre-commit hooks for code quality (commit + push stages).
echo "🎯 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

# Register a Jupyter kernel for the environment.
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=AutoQuake_v0 --display-name="AutoQuake v0.1"

echo "✨ AutoQuake development environment ready!"
echo ""
echo "   Run the test suite with:  pytest tests/unit -m \"not integration\""
echo "   Run the full pipeline with:  python predict.py <config.json>"
