#!/bin/bash
#
# PLATINUM STANDARD - Sentinel Health Co-Pilot Environment Setup
#
# This script sets up a consistent and isolated Python virtual environment,
# installs all required dependencies, and prepares the project for first run.
# It is designed to be idempotent and safe for repeated execution.

# --- Script Configuration & Safety ---
set -e
set -o pipefail

# --- Constants ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
PYTHON_EXEC="${SENTINEL_PYTHON_EXEC:-python3}"
MIN_PYTHON_VERSION_MAJOR=3
MIN_PYTHON_VERSION_MINOR=9

# --- Helper Functions for Logging ---
log_info() { printf "[INFO] %s\n" "$1"; }
log_warn() { printf "[WARN] %s\n" "$1"; }
log_error() { printf "[ERROR] %s\n" "$1" >&2; }
exit_on_error() {
    log_error "$1"
    log_error "Setup aborted."
    exit "${2:-1}"
}

# ==============================================================================
# 1. PRE-REQUISITE CHECKS
# ==============================================================================
log_info "Starting Sentinel Co-Pilot environment setup..."
log_info "Project Root: ${PROJECT_ROOT}"
log_info "Python Interpreter: ${PYTHON_EXEC}"

# Check for Python
if ! command -v "${PYTHON_EXEC}" &>/dev/null; then
    exit_on_error "Python interpreter '${PYTHON_EXEC}' not found. Please install Python ${MIN_PYTHON_VERSION_MAJOR}.${MIN_PYTHON_VERSION_MINOR}+ and ensure it is in your PATH."
fi

# Check Python version
PYTHON_VERSION=$("${PYTHON_EXEC}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$(printf '%s\n' "${MIN_PYTHON_VERSION_MAJOR}.${MIN_PYTHON_VERSION_MINOR}" "$PYTHON_VERSION" | sort -V | head -n1)" != "${MIN_PYTHON_VERSION_MAJOR}.${MIN_PYTHON_VERSION_MINOR}" ]]; then
     exit_on_error "Python version ${PYTHON_VERSION} is not supported. Please use Python ${MIN_PYTHON_VERSION_MAJOR}.${MIN_PYTHON_VERSION_MINOR} or newer."
fi
log_info "Found compatible Python version: $PYTHON_VERSION"

# Check for venv module
if ! "${PYTHON_EXEC}" -m venv -h &>/dev/null; then
    log_warn "Python 'venv' module not found. It may need to be installed separately."
    log_warn "e.g., on Debian/Ubuntu: sudo apt-get install python3-venv"
    log_warn "e.g., on RHEL/CentOS:  sudo yum install python3-devel"
    exit_on_error "'venv' module is required to create an isolated environment."
fi

# ==============================================================================
# 2. SYSTEM DEPENDENCY GUIDANCE
# ==============================================================================
log_info "------------------------------------------------------------------"
log_info "Providing guidance on required system-level dependencies."
log_info "Some Python packages (like Prophet and scikit-learn) require system"
log_info "libraries for compilation. Please ensure they are installed."
log_info "Example commands for your OS:"

if [[ "$(uname)" == "Darwin" ]]; then
    log_info "  macOS (using Homebrew):"
    log_info "    brew install libomp"
elif grep -qE "(Debian|Ubuntu|Mint)" /etc/os-release &>/dev/null; then
    log_info "  Debian/Ubuntu/Mint:"
    log_info "    sudo apt-get update && sudo apt-get install -y build-essential python3-dev"
elif grep -qE "(CentOS|RHEL|Fedora|Rocky)" /etc/os-release &>/dev/null; then
    log_info "  RHEL/CentOS/Fedora/Rocky:"
    log_info "    sudo yum groupinstall -y 'Development Tools' && sudo yum install -y python3-devel"
else
    log_warn "Could not detect your OS. You may need to install a C++ compiler"
    log_warn "(e.g., gcc, build-essential) and Python development headers manually."
fi
log_info "------------------------------------------------------------------"
echo

# ==============================================================================
# 3. VIRTUAL ENVIRONMENT & DEPENDENCY INSTALLATION
# ==============================================================================

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating Python virtual environment at: ${VENV_DIR}"
    "${PYTHON_EXEC}" -m venv "${VENV_DIR}"
else
    log_info "Virtual environment already exists. Skipping creation."
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log_info "Virtual environment activated."
log_info "Using Python interpreter: $(which python)"

# Upgrade pip and install packages
log_info "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    exit_on_error "Requirements file not found at: ${REQUIREMENTS_FILE}"
fi
log_info "Installing Python dependencies from requirements.txt..."
python -m pip install -r "${REQUIREMENTS_FILE}"

log_info "All Python dependencies installed successfully."

# ==============================================================================
# 4. ENVIRONMENT FILE SETUP
# ==============================================================================
ENV_EXAMPLE_FILE="${PROJECT_ROOT}/.env.example"
ENV_FILE="${PROJECT_ROOT}/.env"

if [ -f "$ENV_EXAMPLE_FILE" ] && [ ! -f "$ENV_FILE" ]; then
    log_warn "'.env.example' found. Copying to '.env' for local configuration."
    cp "${ENV_EXAMPLE_FILE}" "${ENV_FILE}"
    log_info "Successfully created '.env' file."
    log_warn "ACTION REQUIRED: Edit the new '.env' file to add your secrets (e.g., SENTINEL_MAPBOX_TOKEN)."
elif [ ! -f "$ENV_EXAMPLE_FILE" ]; then
    log_info "No '.env.example' file found. Skipping .env creation."
else
    log_info "'.env' file already exists. Skipping creation."
fi

# ==============================================================================
# 5. COMPLETION
# ==============================================================================
printf "\n"
log_info "============================================================"
log_info "✅ Sentinel Co-Pilot Environment Setup Complete!"
log_info "============================================================"
echo
log_info "To activate this environment in your terminal, run:"
log_info "  source ${VENV_DIR}/bin/activate"
echo
log_info "To run the Streamlit application, execute:"
log_info "  streamlit run app.py"
echo
log_info "To deactivate the virtual environment when you are done:"
log_info "  deactivate"
echo

exit 0
