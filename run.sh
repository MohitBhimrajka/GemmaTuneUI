#!/bin/bash
set -e  # Exit on error

# ANSI color codes for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "====================================================="
echo "   GemmaTuneUI - Easy Gemma Fine-Tuning Setup        "
echo "====================================================="
echo -e "${NC}"

# Create virtual environment directory if it doesn't exist
VENV_DIR="venv"

# Check if Python 3.9+ is installed
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.9 or newer.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9 or newer is required (found $PYTHON_VERSION)${NC}"
    echo "Please install a newer version of Python and try again."
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Check if we're on macOS
IS_MACOS=false
IS_APPLE_SILICON=false
if [[ "$OSTYPE" == "darwin"* ]]; then
    IS_MACOS=true
    # Check if we're on Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        IS_APPLE_SILICON=true
    fi
fi

# GPU check varies by platform
if [ "$IS_MACOS" = true ]; then
    echo -e "${YELLOW}Mac system detected${NC}"
    
    if [ "$IS_APPLE_SILICON" = true ]; then
        echo -e "${YELLOW}Apple Silicon (M1/M2/M3) detected${NC}"
        echo -e "${YELLOW}Setting up for Mac testing mode${NC}"
        echo -e "${YELLOW}Note: Full GPU acceleration for model training is optimized for NVIDIA GPUs${NC}"
        
        # Set environment variable to use MPS if available (Metal GPU acceleration)
        echo -e "${YELLOW}Enabling Metal Performance Shaders (MPS) if available${NC}"
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        
        # Offer a clear warning about limitations
        echo -e "${YELLOW}⚠ WARNING: You're in Mac testing mode. You'll be able to:${NC}"
        echo -e "${YELLOW}  ✓ Explore the full UI and workflow${NC}"
        echo -e "${YELLOW}  ✓ Test data loading and preprocessing${NC}"
        echo -e "${YELLOW}  ✓ Configure training parameters${NC}"
        echo -e "${YELLOW}  ✗ Full model training will likely fail or be extremely slow${NC}"
        echo -e "${YELLOW}  ✗ For production use, an NVIDIA GPU is recommended${NC}"
        
        read -p "Continue in Mac testing mode? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${YELLOW}Intel Mac detected${NC}"
        echo -e "${YELLOW}⚠ Warning: This application is optimized for NVIDIA GPUs${NC}"
        echo -e "${YELLOW}⚠ You can test the UI, but model training will likely fail${NC}"
        
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    # Non-Mac system, check for CUDA as before
    echo -e "${YELLOW}Checking for NVIDIA GPU and CUDA...${NC}"
    if command -v nvidia-smi &>/dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}⚠ NVIDIA GPU not detected or nvidia-smi not in PATH${NC}"
        echo -e "${YELLOW}⚠ This application requires an NVIDIA GPU with CUDA support for fine-tuning${NC}"
        echo -e "${YELLOW}⚠ You can still continue, but fine-tuning will fail without GPU acceleration${NC}"
        # Give user option to continue
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Set up virtual environment
echo -e "${YELLOW}Setting up virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment in ./$VENV_DIR"
    $PYTHON -m venv $VENV_DIR
    FRESH_INSTALL=true
else
    echo "Using existing virtual environment in ./$VENV_DIR"
    FRESH_INSTALL=false
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source $VENV_DIR/Scripts/activate
else
    # Unix-like systems (Linux, macOS)
    source $VENV_DIR/bin/activate
fi

# Install or update dependencies
if [ "$FRESH_INSTALL" = true ]; then
    echo -e "${YELLOW}Installing dependencies (this may take a while)...${NC}"
    # Upgrade pip
    pip install --upgrade pip
    # Install requirements
    pip install -r requirements.txt
    
    # On Apple Silicon, add additional packages that may help
    if [ "$IS_APPLE_SILICON" = true ]; then
        echo -e "${YELLOW}Installing Mac-specific optimizations...${NC}"
        pip install --upgrade torch torchvision
    fi
else
    echo -e "${YELLOW}Checking for missing dependencies...${NC}"
    pip install -r requirements.txt
    
    # On Apple Silicon, ensure optimized packages
    if [ "$IS_APPLE_SILICON" = true ]; then
        echo -e "${YELLOW}Updating Mac-specific optimizations...${NC}"
        pip install --upgrade torch torchvision
    fi
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Set environment variables for Mac
if [ "$IS_MACOS" = true ]; then
    echo -e "${YELLOW}Setting up Mac environment...${NC}"
    # Enable MPS (Metal Performance Shaders) if applicable
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    if [ "$IS_APPLE_SILICON" = true ]; then
        export USE_MPS=1
    fi
    # Pass this to the app so it knows we're in Mac testing mode
    export MAC_TESTING_MODE=1
fi

# Run the app
echo -e "${BLUE}Starting GemmaTuneUI...${NC}"
echo -e "${BLUE}The application will open in your web browser${NC}"
streamlit run app.py

# Deactivate virtual environment when done
deactivate 