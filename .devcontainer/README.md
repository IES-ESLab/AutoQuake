# AutoQuake Development Container

This devcontainer provides a standardized development environment for the AutoQuake project, ensuring all contributors work in a consistent setup regardless of their local machine configuration.

## 🚀 Quick Start

### Prerequisites
- [Docker](https://www.docker.com/get-started)
- [VS Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting Started
1. Clone the AutoQuake repository
2. Open the project in VS Code
3. When prompted, click "Reopen in Container" or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
4. Wait for the container to build and setup (first time takes ~5-10 minutes)
5. Start developing! 🎉

## 📦 What's Included

### Environment
- **Python 3.10.13** with conda/mamba package management
- **AutoQuake conda environment** built from `env-codespace.yml` (CPU-only PyTorch)
- **Scientific computing stack**: NumPy, Pandas, SciPy, Matplotlib, ObsPy
- **Machine Learning**: PyTorch (CPU), scikit-learn, ONNX Runtime
- **Seismology / geospatial tools**: ObsPy, PyGMT, Cartopy, pyproj
- **Development tools**: Ruff, pre-commit

### VS Code Extensions
- Python development suite (Python, Pylance, Jupyter)
- Linting & formatting with Ruff
- Git integration (GitLens, GitHub tools)
- Documentation tools (Markdown, autodocstring)

### Development Tools
- **Ruff** for linting and formatting (replaces black + isort + flake8)
- **Pre-commit hooks** for code quality
- **Jupyter Lab** ready to use

## 🛠️ Development Workflow

### Environment Activation
The conda environment `AutoQuake_v0` is activated automatically in new terminals.
To activate it manually:
```bash
conda activate AutoQuake_v0
```

### Development Tools
```bash
ruff check .                   # Lint
ruff format .                  # Format (single quotes, 88 cols — see pyproject.toml)
pytest tests/unit -m "not integration"   # Run the fast unit tests
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser  # Start Jupyter Lab
```

### Testing Your Setup
```bash
conda activate AutoQuake_v0    # Ensure environment is active
python -c "import autoquake; print('AutoQuake loaded successfully')"  # Test import
```

## 📁 Project Structure

**GitHub Codespaces:** The repository is automatically mounted to `/workspaces/AutoQuake`
**Local Dev Containers:** The repository is mounted to your local project directory

```
AutoQuake/
├── .devcontainer/           # Development container configuration
│   ├── devcontainer.json    # VS Code dev container settings
│   ├── Dockerfile          # Container image definition
│   ├── post-create.sh      # Setup script
│   └── README.md           # This file
├── autoquake/              # Main Python package
├── data/                   # Data directory (created automatically)
├── outputs/                # Output directory (created automatically)
├── tests/                  # Test directory (created automatically)
├── env.yml                 # Conda environment definition
└── pyproject.toml          # Project configuration
```

## 🌐 Real-time Data Access

This environment is prepared for real-time seismic data processing with:
- Network libraries for data center access
- Streaming data processing capabilities
- Efficient memory management for continuous processing

## 🔧 Customization

### Adding Dependencies
1. Add the package to `env.yml` (local/CI) **and** `env-codespace.yml` (Codespaces)
2. Rebuild the container ("Dev Containers: Rebuild Container")

### VS Code Settings
Modify `.devcontainer/devcontainer.json` to customize:
- VS Code extensions
- Python interpreter settings
- Code formatting preferences
- Port forwarding

### System Dependencies
Add system packages in the Dockerfile under the apt-get install section.

## 🚨 Troubleshooting

### Container Won't Start
- Ensure Docker is running
- Try rebuilding: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

### Python Environment Issues
```bash
conda activate AutoQuake_v0   # Ensure environment is active
conda env list                # Confirm AutoQuake_v0 exists
```

### Import Errors
The project runs from the repository root (pytest adds it to `sys.path` via
`pyproject.toml`). If `import autoquake` fails, make sure your working directory
is the repo root and the conda environment is active.

### Port Conflicts
The container forwards ports 8888, 8080, and 5000. Modify `devcontainer.json` if needed.

## 🤝 Contributing

1. The environment automatically sets up pre-commit hooks (commit + push)
2. Code is automatically formatted on save by Ruff
3. Run tests before committing: `pytest tests/unit -m "not integration"`
4. Follow the existing code style (Ruff: single quotes, 88-column lines)

## 📝 Notes for Real-time Development

When implementing real-time features:
- Use the `fsspec` library for flexible data access
- Consider `obspy.clients` for seismic data center access
- The environment includes network tools for data streaming
- Memory-efficient processing tools are pre-installed

## 🆘 Getting Help

- Check the main AutoQuake README for project-specific information
- Run `conda env list` and `conda list` to inspect the environment
- Use the VS Code integrated terminal with all tools pre-configured

Happy coding! 🎉
