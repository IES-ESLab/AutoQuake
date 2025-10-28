# AutoQuake

> **AutoQuake: An Automated, All-in-One Solution for Earthquake Catalog Generation.**: An object-oriented package that integrates each step of generating an earthquake catalog. The package connects each step as an object, making it easy to manage and streamline the process.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Installation](#installation)
- [Dataset Structure (both hdf5 for DAS and SAC for Seismometer)](#dataset-structure-both-hdf5-for-das-and-sac-for-seismometer)
- [Usage](#usage)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Submodules and External Dependencies](#submodules-and-external-dependencies)
- [References](#references)
- [Contact](#contact)

---

## Getting Started
To get started with this project, choose the scenario that best describes your needs:

### Scenario 1: Preliminary Users (Use Only)
![For Preliminary users](images/for_normal_user.png)
If you only want to use the package without contributing to the codebase, follow these simple steps:

1. **Clone the repository and init the submodules (both HTTPS or SSH is fine)**:
   ```bash
   git clone --branch main git@github.com:IES-ESLab/AutoQuake.git
   cd AutoQuake
   ./init_submodules.sh # use sh to execute the script if powershell.
   ```
2. **Install dependecies**: Follow the installation instructions in the [Installation](#installation) section.       
**Use the Package**: Follow the instructions in the Usage section to start generating earthquake catalogs.       
**Report Issues**: If you encounter any problems or have suggestions for improvement, please [open or report an issue](https://github.com/IES-ESLab/AutoQuake/issues). Describe the problem in detail to help us address it quickly.        
**Pull Updates**: Regularly pull updates from the original repository to keep your local copy synchronized and avoid conflicts.
   ```bash
   git pull origin main
   ```
### Scenario 2: Contributors (Recommended. Develop and Improve)
![For contributors](images/for_contributor.png)
If you’re interested in contributing to the development of this project, follow these steps:

1. **Fork the repository**: Visit the GitHub page of the repository and click on "Fork" to create your own copy.
2. **Clone your forked repository and init the submodules (both HTTPS or SSH is fine)**:
   ```bash
   git clone --branch dev https://github.com/your-username/AutoQuake.git # clone the dev branch
   cd AutoQuake
   ./init_submodules.sh # use sh to execute the script if powershell.
   ```
3. **Install dependecies**: Follow the installation instructions in the [Installation](#installation) section.
4. **Adding remote for original repository**: Connect your local repository to the original (upstream) repository to fetch updates.    
   ```bash
   git remote add upstream https://github.com/IES-ESLab/AutoQuake.git
   ```
5. **Make your changes**:
   ```bash
   git add .
   ```
6. **commit** them:
   ```bash
   git commit -m "Description of your changes"
   ```
7. **Push your changes** to your forked repository when you're ready to share your work:
   ```bash
   git push origin feature/your-feature-name
   ```
**Pull Updates**: Regularly pull updates from the original repository to keep your local copy synchronized and avoid conflicts.
   ```bash
   git pull upstream main
   ```
**Open a Pull Request**: Once your changes are pushed, go to the original repository on GitHub and open a Pull Request from your branch to the dev branch of the main repository.
Read to our [Contributing](#contributing) section for more details on how to contribute to this project.       
**Report Issues**: You can help resolving the issues.    

---

## Development Environment

### 🚀 DevContainer Setup (Recommended for Contributors)
We provide a complete development container that ensures all contributors work in a consistent environment:

#### Quick Start
1. **Prerequisites**: Install [Docker](https://www.docker.com/get-started), [VS Code](https://code.visualstudio.com/), and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Clone and open**:
   ```bash
   git clone --branch dev https://github.com/your-username/AutoQuake.git
   cd AutoQuake
   code .  # Open in VS Code
   ```

3. **Start developing**: When prompted, click "Reopen in Container" or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

#### What's Included
- ✅ Python 3.10 with pre-configured conda environment
- ✅ All scientific dependencies (NumPy, ObsPy, PyTorch, etc.)
- ✅ VS Code extensions for Python, Jupyter, and seismology
- ✅ Code formatting tools (Black, isort, Ruff)
- ✅ Testing framework (pytest) with coverage
- ✅ Pre-commit hooks for code quality
- ✅ Ready for real-time seismic data processing

#### Development Commands
```bash
aq          # Activate AutoQuake environment
aqtest      # Run tests with pytest
aqformat    # Format code with Black and isort
aqcheck     # Check code quality
aqjupyter   # Start Jupyter Lab
```

📖 **Detailed Guide**: See [`.devcontainer/README.md`](.devcontainer/README.md) for complete documentation.

---

## Installation
This package supports both `requirements.txt` and `env.yml` for dependency management. Choose one of the following installation methods:    

### 1. Using Conda (`env.yml`)
If you’re using Conda, create a new environment directly from the provided `env.yml` file:
   ```bash
   conda env create -f env.yml
   conda activate AutoQuake_v0
   ```
   
### 2. Using pip (`requirements.txt`)
To install dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configurations with `pyproject.toml`
The `pyproject.toml` file contains configuration settings, including code linting rules (using `ruff`) and project metadata. To apply these settings, make sure to install `ruff`:
   ```bash
   pip install ruff
   ```
## Dataset Structure (Different data format in different data directory)
**Important**: For the package to function correctly, the dataset must follow a specific directory structure. Ensure your data is organized as follows:    
For seismometer:    
   ```bash
   /dataset_parent_dir/
    ├── \*YYYYMMDD\*/         # data folder should contain the Year-Month-Day format for glob. 
    │   └── waveform.SAC      # Waveform data in SAC format
   ```
For DAS:    
   ```bash
   /dataset_parent_dir/
    ├── \*YYYYMMDD\*/          
    │   └── MiDAS_20240402_86100_86400.h5      # Waveform data in h5 format
   ```
<!-- ## Usage
This package includes a Sphinx-generated documentation site and a Jupyter Notebook for a step-by-step guideline.     
###  Documentation
The documentation includes:
- **Installation and Setup Guide**: Step-by-step instructions for setting up and using the package.
- **API Reference**: Detailed explanations of each class, method, and function.
- **Workflow Examples**: Usage examples demonstrating how to generate an earthquake catalog from start to finish.

To build the Sphinx documentation locally, run:
```bash
cd docs
make html
```
The documentation will be generated in `docs/_build/html` and can be opened in a browser.
## Contributing
We encourage developers to contribute to this project. Here’s a quick guide:

1. **Fork the repository** and **clone your fork** (recommended to work with the dev branch).
2. **Create a new branch** for your feature or bug fix.
3. **Make changes** and **commit** them.
4. **Push to your branch** and **open a Pull Request** to the `dev` branch. -->
## Roadmap
Here are the planned future improvements for this project:

-  **Interactive Visualization**: Add interactive tools for visualizing and analyzing earthquake catalogs.
-  **Enhanced Validation**: Improved data validation checks to ensure datasets are correctly formatted and complete.
-  **Performance Optimization**: Memory and performance enhancements to better handle large datasets.
-  **Additional Data Format Support**: Add compatibility for other seismic data formats.
-  **Refactoring Fortrans into Python**: Replace existing Fortran code with Python implementations for better maintainability and compatibility.
-  **Integration with Existing Tools**: Integrate with popular seismic data processing tools and platforms.

The roadmap can adapt based on user feedback and new technology developments. If you’re interested in contributing to any of these features, check the [Contributing](#contributing) section.
## License
This project is licensed under the MIT License, which allows for reuse, modification, and distribution with minimal restrictions. See the [LICENSE](./LICENSE) file for the full MIT License text.

### Submodule Licenses
This repository includes third-party submodules with their own licensing terms:
1. **GaMMA** - MIT License: Permissive license allowing free use, modification, and distribution.
2. **EQNet** - Academic and Commercial License:
   - This submodule is available for academic and research use only. For commercial use, a separate license is required. Contact the authors for more information.
   - Users must provide proper attribution in any publications resulting from its use.

Please review each submodule’s `LICENSE` file for detailed terms.
## Submodules and External Dependencies
This project relies on the following external repositories as submodules:
1. [**EQNet**](https://github.com/IES-ESLab/EQNet): A forked repository contains the PhaseNet, PhaseNet-DAS.
2. [**GaMMA**](https://github.com/IES-ESLab/GaMMA): A forked repository contains the Gaussian Mixture Model for Earthquake Detection and Location.
## References
- Huang, Hsin-Hua, Wu, Y.-M., Song, X., Chang, C.-H., Lee, S.-J., Chang, T.-M., & Hsieh, H.-H. (2014). [**Joint Vp and Vs tomography of Taiwan: Implications for subduction-collision orogeny.**](https://www.sciencedirect.com/science/article/pii/S0012821X14000995?via%3Dihub) Earth Planet Science Letters, 392, 177–191. https://doi.org/10.1016/j.epsl.2014.02.026
- Hsu, Yu-Fang & Huang, Hsin-Hua & Huang, Mong-Han & Tsai, Victor & Chuang, Ray & Feng, Kuan-Fu & Lin, S.‐H. (2020). [**Evidence for Fluid Migration During the 2016 Meinong, Taiwan, Aftershock Sequence.**](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JB019994) Journal of Geophysical Research: Solid Earth. 125. 10.1029/2020JB019994.
- Wu, Yih-Min et al. (2008). [**Focal-Mechanism Determination in Taiwan by Genetic Algorithm.**](https://pubs.geoscienceworld.org/ssa/bssa/article/98/2/651/350113/Focal-Mechanism-Determination-in-Taiwan-by-Genetic). Bulletin of the Seismological Society of America 2008;; 98 (2): 651–661. doi: https://doi.org/10.1785/0120070115
- Zhu, Weiqiang et al. (2021). [**Earthquake Phase Association using a Bayesian Gaussian Mixture Model.**](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023249) Journal of Geophysical Research: Solid Earth, 127, e2021JB023249. https://doi.org/10.1029/2021JB023249
- Zhu, Weiqiang, and Gregory C. Beroza. (2018). [**PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method.**](https://academic.oup.com/gji/article/216/1/261/5129142) Geophysical Journal International, Volume 216, Issue 1, January 2019, Pages 261–273, https://doi.org/10.1093/gji/ggy423
- Zhu, Weiqiang et al. (2023). [**Seismic arrival-time picking on distributed acoustic sensing data using semi-supervised learning.**](https://www.nature.com/articles/s41467-023-43355-3) Nat Commun 14, 8192. https://doi.org/10.1038/s41467-023-43355-3
- Zhao, M., Xiao, Z., Zhang, M., Yang, Y., Tang, L., & Chen, S. (2023). [**DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion.**](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1103914/full) Frontiers in Earth Science, 11, 335.https://doi.org/10.3389/feart.2023.1103914
## Contact
If you have any questions or suggestions, feel free to reach out:
- **Email**: [patrick.yang880612@gmail.com](mailto:patrick.yang880612@gmail.com)
- **GitHub**: [@Pamicoding](https://github.com/Pamicoding)

