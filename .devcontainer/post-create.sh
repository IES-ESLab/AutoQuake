#!/bin/bash

echo "🚀 Setting up AutoQuake development environment..."

# Ensure conda is properly initialized
source /opt/conda/etc/profile.d/conda.sh

# Activate the AutoQuake environment
conda activate AutoQuake_v0

# Verify environment activation
echo "📦 Active conda environment: $CONDA_DEFAULT_ENV"

# Install development dependencies (ipykernel and jupyterlab already in env-codespace.yml)
echo "🔧 Installing development tools..."
pip install --no-cache-dir \
    black \
    isort \
    flake8 \
    pre-commit

# Install the AutoQuake package in development mode
echo "📥 Installing AutoQuake in development mode..."
pip install -e .

# Initialize git submodules if they exist
if [ -f ".gitmodules" ]; then
    echo "🔗 Initializing git submodules..."
    git submodule update --init --recursive
fi

# Set up pre-commit hooks for code quality
echo "🎯 Setting up pre-commit hooks..."
pre-commit install

# Create Jupyter kernel for the environment
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=AutoQuake_v0 --display-name="AutoQuake v0.1"

# Create commonly used directories
mkdir -p data
mkdir -p outputs
mkdir -p logs

# Set up git configuration helper
echo "🔧 Setting up git configuration..."
echo '#!/bin/bash
echo "Quick setup for git configuration:"
echo "git config --global user.name \"Your Name\""
echo "git config --global user.email \"your.email@example.com\""
' > setup-git.sh
chmod +x setup-git.sh

echo "✨ AutoQuake development environment setup completed!"
echo ""
echo "🚀 Ready for AutoQuake development!"
