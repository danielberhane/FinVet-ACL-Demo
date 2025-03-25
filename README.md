# Financial Misinformation Detection System (FinVet)

A system for detecting misinformation in financial claims using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques.

## Quick Start for ACL Reviewers

To quickly evaluate this tool, follow these steps:

1. **Installation**:
   ```bash
   # Install pyarrow first to avoid build issues
   pip install pyarrow==11.0.0
   
   # Clone and install the package
   git clone https://github.com/danielberhane/FinVet-ACL-Demo.git
   cd FinVet-ACL-Demo
   pip install .
   ```

   Alternatively, install directly from GitHub:
   ```bash
   pip install pyarrow==11.0.0
   pip install git+https://github.com/danielberhane/FinVet-ACL-Demo.git
   ```

2. **Download Required Data Files**:
   - Go to the [Releases page](https://github.com/danielberhane/FinVet-ACL-Demo/releases/latest)
   - Download `metadata.pkl` and `faiss_index.bin`
   - Place these files in the following directory:
     ```bash
     # On macOS/Linux
     ~/.financial-misinfo/
     
     # On Windows
     C:\Users\YourUsername\.financial-misinfo\
     ```
   - Alternatively, the system will attempt to download these files automatically on first run

3. **Configuration**:
   ```bash
   financial-misinfo config --hf-token YOUR_HF_TOKEN --google-api-key YOUR_GOOGLE_API_KEY
   ```
   
   You'll need:
   - HuggingFace API token: Get one at https://huggingface.co/settings/tokens
   - Google API key with Fact Check API enabled: Create at https://console.cloud.google.com/

4. **Launch the Web Interface**:
   ```bash
   financial-misinfo ui
   ```
   This will open a browser window with an interactive interface.

5. **Example Claims to Try**:
   - "Apple's stock price doubled in 2023"
   - "Tesla achieved profitability for the first time in 2020"
   - "Amazon acquired Whole Foods for $13 billion"

## Overview

This tool helps users verify financial claims by combining:
* Google Fact Check API integration
* RAG (Retrieval-Augmented Generation) technology
* Multiple LLM verification pathways
* A voting mechanism to determine the final verdict

## Installation

### Prerequisites

* Python 3.8 or higher (Python 3.10 recommended)
* HuggingFace API token: https://huggingface.co/settings/tokens
* Google API key: https://console.cloud.google.com/ with Fact Check API enabled

### Installation Steps

1. **Install pyarrow first** (important to avoid dependency issues):
   ```bash
   pip install pyarrow==11.0.0
   ```

2. **Install the package**:

   From GitHub:
   ```bash
   pip install git+https://github.com/danielberhane/FinVet-ACL-Demo.git
   ```

   Or after cloning the repository:
   ```bash
   git clone https://github.com/danielberhane/FinVet-ACL-Demo.git
   cd FinVet-ACL-Demo
   pip install .
   ```

### Installation Notes

If you encounter dependency issues:

1. **PyArrow Issues**: PyArrow must be installed separately first to avoid build errors
   ```bash
   pip install pyarrow==11.0.0
   ```

2. **Alternative Installation Methods**:
   - With UI dependencies: `pip install ".[ui]"`
   - For development: `pip install -e ".[dev]"`
   - Using conda: `conda env create -f environment.yml`

3. **Python Version**: Make sure you're using Python 3.8, 3.9, or 3.10

## Configuration

Before using the system, you need to configure your API keys:

```bash
financial-misinfo config --hf-token YOUR_HF_TOKEN --google-api-key YOUR_GOOGLE_API_KEY
```

Alternatively, you can enter your API keys in the web interface under "API Credentials".

## Usage

### Command Line Interface

Verify a financial claim:
```bash
financial-misinfo verify "Tesla's stock price doubled in 2023"
```

### Web Interface

Launch the Streamlit-based web UI:
```bash
financial-misinfo ui
```

This will open a browser window with a user-friendly interface for verifying financial claims.

## System Architecture

The system uses multiple verification pathways:
* RAG Pipeline A: Uses primary LLM with retrieved context
* RAG Pipeline B: Uses secondary LLM with retrieved context
* Fact Check: Uses Google Fact Check API with LLM fallback

Results from all pathways are combined through a voting mechanism to determine the final verdict.

## Data Files

This package uses pre-built index files for efficient operation:
- `faiss_index.bin`: Vector embeddings for the knowledge base
- `metadata.pkl`: Metadata associated with the knowledge base

### Automatic Download

On first run, the system will attempt to automatically download these files from the GitHub release. This requires an internet connection.

### Manual Installation

If the automatic download fails, you can manually download the files from:
https://github.com/danielberhane/FinVet-ACL-Demo/releases/latest

After downloading, place them in:
```bash
# On macOS/Linux
~/.financial-misinfo/

# On Windows
C:\Users\YourUsername\.financial-misinfo\
```

## License

MIT License

## Contact

Daniel Berhane Araya - dberhan4@gmu.edu