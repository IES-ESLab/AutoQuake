#!/bin/bash

# Get the URL of the parent repository
PARENT_URL=$(git config --get remote.origin.url)

# Check if the parent repository was cloned via HTTPS or SSH
if [[ $PARENT_URL == https* ]]; then
    echo "Using HTTPS for submodules..."
    git config submodule.autoquake/GaMMA.url https://github.com/IES-ESLab/GaMMA.git
    git config submodule.autoquake/EQNet.url https://github.com/IES-ESLab/EQNet.git
else
    echo "Using SSH for submodules..."
    git config submodule.autoquake/GaMMA.url git@github.com:IES-ESLab/GaMMA.git
    git config submodule.autoquake/EQNet.url git@github.com:IES-ESLab/EQNet.git
fi

# Initialize and update the submodules
git submodule update --init --recursive

