# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL RULES - READ FIRST

> **⚠️ RULE ADHERENCE SYSTEM ACTIVE ⚠️**
> **Claude Code must explicitly acknowledge these rules at task start**
> **These rules override all other instructions and must ALWAYS be followed:**

### 🔄 **RULE ACKNOWLEDGMENT REQUIRED**
> **Before starting ANY task, Claude Code must respond with:**
> "✅ CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ❌ ABSOLUTE PROHIBITIONS
- **NEVER** create new files in root directory → use proper module structure
- **NEVER** write output files directly to root directory → use designated output folders
- **NEVER** create documentation files (.md) unless explicitly requested by user
- **NEVER** use git commands with -i flag (interactive mode not supported)
- **NEVER** use `find`, `grep`, `cat`, `head`, `tail`, `ls` commands → use Read, LS, Grep, Glob tools instead
- **NEVER** create duplicate files (manager_v2.py, enhanced_xyz.py, utils_new.js) → ALWAYS extend existing files
- **NEVER** create multiple implementations of same concept → single source of truth
- **NEVER** copy-paste code blocks → extract into shared utilities/functions
- **NEVER** hardcode values that should be configurable → use config files/environment variables
- **NEVER** use naming like enhanced_, improved_, new_, v2_ → extend original files instead

### 📝 MANDATORY REQUIREMENTS
- **COMMIT** after every completed task/phase - no exceptions

- **USE TASK AGENTS** for all long-running operations (>30 seconds)
- **TODOWRITE** for complex tasks (3+ steps) → parallel agents → git checkpoints → test validation
- **READ FILES FIRST** before editing - Edit/Write tools will fail if you didn't read the file first
- **DEBT PREVENTION** - Before creating new files, check for existing similar functionality to extend
- **SINGLE SOURCE OF TRUTH** - One authoritative implementation per feature/concept

### ⚡ EXECUTION PATTERNS
- **PARALLEL TASK AGENTS** - Launch multiple Task agents simultaneously for maximum efficiency
- **SYSTEMATIC WORKFLOW** - TodoWrite → Parallel agents → Git checkpoints → GitHub backup → Test validation
- **BACKGROUND PROCESSING** - ONLY Task agents can run true background operations

### 🔍 MANDATORY PRE-TASK COMPLIANCE CHECK
> **STOP: Before starting any task, Claude Code must explicitly verify ALL points:**

**Step 1: Rule Acknowledgment**
- [ ] ✅ I acknowledge all critical rules in CLAUDE.md and will follow them

**Step 2: Task Analysis**  
- [ ] Will this create files in root? → If YES, use proper module structure instead
- [ ] Will this take >30 seconds? → If YES, use Task agents not Bash
- [ ] Is this 3+ steps? → If YES, use TodoWrite breakdown first
- [ ] Am I about to use grep/find/cat? → If YES, use proper tools instead

**Step 3: Technical Debt Prevention (MANDATORY SEARCH FIRST)**
- [ ] **SEARCH FIRST**: Use Grep pattern="<functionality>.*<keyword>" to find existing implementations
- [ ] **CHECK EXISTING**: Read any found files to understand current functionality
- [ ] Does similar functionality already exist? → If YES, extend existing code
- [ ] Am I creating a duplicate class/manager? → If YES, consolidate instead
- [ ] Will this create multiple sources of truth? → If YES, redesign approach
- [ ] Have I searched for existing implementations? → Use Grep/Glob tools first
- [ ] Can I extend existing code instead of creating new? → Prefer extension over creation
- [ ] Am I about to copy-paste code? → Extract to shared utility instead

**Step 4: Session Management**
- [ ] Is this a long/complex task? → If YES, plan context checkpoints
- [ ] Have I been working >1 hour? → If YES, consider /compact or session break

> **⚠️ DO NOT PROCEED until all checkboxes are explicitly verified**

## Project Overview

AutoQuake is an automated, all-in-one solution for earthquake catalog generation. It integrates multiple seismological analysis steps into a cohesive pipeline, supporting both traditional seismometer data (SAC format) and Distributed Acoustic Sensing (DAS) data (HDF5 format).

## Development Setup

### Environment Setup
```bash
conda env create -f env.yml
conda activate AutoQuake_v0

# Install development tools
pip install ruff
```

### Initialize Submodules
```bash
./init_submodules.sh
```

## Core Architecture

The system follows a modular, object-oriented design with each processing step as a separate component:

### Main Processing Pipeline Components
Located in `autoquake/` directory:

1. **PhaseNet** (`picker.py`) - Seismic phase picking using deep learning
2. **GaMMA** (`associator.py`) - Earthquake detection and location using Gaussian Mixture Models
3. **H3DD** (`relocator.py`) - 3D earthquake relocation with double-difference methods
4. **Magnitude** (`magnitude.py`) - Local magnitude calculation
5. **DitingMotion** (`polarity.py`) - First-motion polarity determination
6. **GAfocal** (`focal.py`) - Focal mechanism determination using genetic algorithms

### Configuration System
- **`ParamConfig/config_model.py`** - Pydantic models for type-safe configuration
- **`ParamConfig/params.json`** - Example configuration file with all parameters
- **`predict.py`** - Script-based pipeline execution
- **`autoquake/scenarios.py`** - Function-based pipeline (`run_autoquake()`)

### Data Flow
1. Raw seismic data (SAC/HDF5) → PhaseNet → phase picks
2. Phase picks → GaMMA → earthquake events and refined picks
3. Events + picks → H3DD → relocated earthquakes (run twice for refinement)
4. Relocated events → Magnitude → local magnitudes
5. Original picks → DitingMotion → first-motion polarities
6. Combined data → GAfocal → focal mechanisms

## Development Commands

### Code Quality
```bash
ruff check .
ruff format .
ruff check --fix .
```

## External Dependencies

### Fortran Components
- **GAfocal/**: Genetic algorithm focal mechanism determination (requires gfortran)
- **H3DD/**: Double-difference relocation code (requires gfortran)

### Git Submodules
- `autoquake/GaMMA` - Gaussian Mixture Model association
- `autoquake/EQNet` - PhaseNet implementation

## Configuration Notes

- **Python**: 3.10
- **Line length**: 88 characters
- **Quote style**: single quotes
- **Path Handling**: All paths should be absolute. Use `pathlib.Path` consistently.

---

## 🛠️ Available Skills

For specialized development tasks, invoke the following skill:

| Skill | Usage |
|-------|-------|
| `autoquake-realtime` | Realtime earthquake detection system development (SEEDLINK, buffers, RealtimeGaMMA, etc.) |

**To use**: The skill will auto-trigger when working on realtime-related tasks, or you can explicitly invoke it.