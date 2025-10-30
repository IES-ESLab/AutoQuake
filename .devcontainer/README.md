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
- **AutoQuake conda environment** with all dependencies from `env.yml`
- **Scientific computing stack**: NumPy, Pandas, SciPy, Matplotlib, ObsPy
- **Machine Learning**: PyTorch, scikit-learn
- **Seismology tools**: ObsPy, PyGMT, Pyrocko
- **Development tools**: Black, isort, flake8, pre-commit

### VS Code Extensions
- Python development suite (Python, Pylance, Jupyter)
- Code formatting and linting (Black, Ruff, flake8)
- Git integration (GitLens, GitHub tools)
- Documentation tools (Markdown, autodocstring)
- Scientific computing support

### Development Tools
- **Pre-commit hooks** for code quality
- **Jupyter Lab** ready to use
- **Code formatting** with Black and isort
- **Linting** with flake8 and Ruff

## 🛠️ Development Workflow

### Environment Activation
The conda environment `AutoQuake_v0` is automatically activated. You can also use:
```bash
aq  # Quick alias to activate environment
```

### Development Tools
```bash
# You can create your own aliases if needed, such as:
conda activate AutoQuake_v0    # Activate environment
black . && isort .             # Format code
flake8 .                       # Check code quality
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root  # Start Jupyter
```

### Jupyter Development
```bash
aqjupyter   # Start Jupyter Lab on port 8888
```

### Testing Your Setup
```bash
conda activate AutoQuake_v0    # Ensure environment is active
python -c "import autoquake; print('AutoQuake loaded successfully')"  # Test AutoQuake import
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
1. Update `env.yml` for conda packages
2. Update `requirements.txt` for pip packages
3. Rebuild the container

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
conda activate AutoQuake_v0  # Ensure environment is active
python dev-utils.py          # Check environment status
```

### Import Errors
```bash
pip install -e .  # Reinstall AutoQuake in development mode
```

### Port Conflicts
The container forwards ports 8888, 8080, and 5000. Modify `devcontainer.json` if needed.

## 🤝 Contributing

1. The environment automatically sets up pre-commit hooks
2. Code is automatically formatted on save
3. Run tests before committing: `aqtest`
4. Follow the existing code style (Black formatting)

## 📝 Notes for Real-time Development

When implementing real-time features:
- Use the `fsspec` library for flexible data access
- Consider `obspy.clients` for seismic data center access
- The environment includes network tools for data streaming
- Memory-efficient processing tools are pre-installed

## 🆘 Getting Help

- Check the main AutoQuake README for project-specific information
- Run `python dev-utils.py` for environment diagnostics
- Use the VS Code integrated terminal with all tools pre-configured

Happy coding! 🎉
