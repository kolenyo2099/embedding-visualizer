#!/bin/bash

# Semantic Embedding Explorer - Setup and Run Script for macOS
# This script creates a virtual environment, installs dependencies, and starts the app

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="venv"
APP_NAME="Semantic Embedding Explorer"
PYTHON_MIN_VERSION="3.8"

echo -e "${BLUE}🎯 $APP_NAME - Setup and Run Script${NC}"
echo "=================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python $PYTHON_VERSION is installed. Python $REQUIRED_VERSION or higher is required.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Check if virtual environment exists
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}📁 Virtual environment '$VENV_NAME' already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Removing existing virtual environment...${NC}"
        rm -rf "$VENV_NAME"
    else
        echo -e "${BLUE}📂 Using existing virtual environment${NC}"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
    python3 -m venv "$VENV_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${BLUE}🚀 Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found in current directory${NC}"
    exit 1
fi

# Install requirements
echo -e "${BLUE}📦 Installing requirements...${NC}"
echo "This may take a few minutes, especially for PyTorch..."

# Install requirements with progress indicator
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All requirements installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install requirements${NC}"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ app.py not found in current directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All setup complete!${NC}"
echo ""
echo -e "${BLUE}🚀 Starting $APP_NAME...${NC}"
echo -e "${YELLOW}The app will open in your default browser at http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the Streamlit app
streamlit run app.py

# Deactivate virtual environment when the app is stopped
echo -e "${BLUE}👋 Deactivating virtual environment...${NC}"
deactivate
