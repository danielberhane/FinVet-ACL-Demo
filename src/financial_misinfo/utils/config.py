"""Configuration management for the financial misinformation system."""

import os
import json
import importlib.resources
from pathlib import Path
from typing import Dict, Any, Optional

# Set user's base directory 
BASE_DIR = Path(os.path.expanduser("~/.financial-misinfo"))
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_VECTOR_STORE_PATH = BASE_DIR / "faiss_index.bin"
DEFAULT_METADATA_PATH = BASE_DIR / "metadata.pkl"

# Try to find the package's data directory
def get_package_data_dir():
    """Try to find the package's data directory"""
    try:
        # For Python 3.9+
        import importlib.resources
        try:
            with importlib.resources.files("financial_misinfo.data") as data_dir:
                return data_dir
        except Exception:
            pass
    except Exception:
        pass
    
    # Fallback method
    try:
        import financial_misinfo
        path = Path(os.path.dirname(financial_misinfo.__file__)) / "data"
        if path.exists():
            return path
    except Exception:
        return None

def ensure_base_dir():
    """Ensure base directory exists"""
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR

def get_vector_store_path():
    """Get path to vector store, checking package location first"""
    package_dir = get_package_data_dir()
    if package_dir:
        vector_path = package_dir / "faiss_index.bin"
        if vector_path.exists():
            return vector_path
    
    # Fall back to user directory
    ensure_base_dir()
    return DEFAULT_VECTOR_STORE_PATH

def get_metadata_path():
    """Get path to metadata, checking package location first"""
    package_dir = get_package_data_dir()
    if package_dir:
        metadata_path = package_dir / "metadata.pkl"
        if metadata_path.exists():
            return metadata_path
    
    # Fall back to user directory
    ensure_base_dir()
    return DEFAULT_METADATA_PATH

def get_default_config() -> Dict[str, Any]:
    """Get default configuration with appropriate fallbacks."""
    return {
        "vector_store_path": str(get_vector_store_path()),
        "metadata_path": str(get_metadata_path()),
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "embedding_model": "all-MiniLM-L6-v2",
        "models": {
            "primary_llm": "meta-llama/Llama-3.3-70B-Instruct",
            "secondary_llm": "mistralai/Mixtral-8x7B-Instruct-v0.1"
        },
        "max_context_items": 5
    }

def load_config(config_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """Load configuration from a JSON file with fallback to default values."""
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    # Start with default configuration
    config = get_default_config()
    
    # Override with file config if available
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            if verbose:
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Error loading config from {config_path}: {e}")
            print("Using default configuration")
    else:
        if verbose:
            print(f"Config file {config_path} not found. Using default configuration.")
        # Create default config file if directory exists
        if config_path.parent.exists():
            save_config(config, config_path)
    
    return config

def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """Save configuration to a JSON file."""
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")

def expand_path(path: str) -> str:
    """Expand user home directory and resolve path."""
    return str(Path(path).expanduser().resolve())

def copy_package_data_to_user_dir():
    """
    Copy pre-built index files from package data to user directory
    if they don't exist in user directory but do exist in package.
    """
    package_dir = get_package_data_dir()
    if not package_dir:
        return False
    
    # Ensure base directory exists
    ensure_base_dir()
    
    copied = False
    # Check and copy vector store if needed
    package_vector = package_dir / "faiss_index.bin"
    if package_vector.exists() and not DEFAULT_VECTOR_STORE_PATH.exists():
        import shutil
        try:
            shutil.copy2(package_vector, DEFAULT_VECTOR_STORE_PATH)
            print(f"Copied vector store from package to {DEFAULT_VECTOR_STORE_PATH}")
            copied = True
        except Exception as e:
            print(f"Error copying vector store: {e}")
    
    # Check and copy metadata if needed
    package_metadata = package_dir / "metadata.pkl"
    if package_metadata.exists() and not DEFAULT_METADATA_PATH.exists():
        import shutil
        try:
            shutil.copy2(package_metadata, DEFAULT_METADATA_PATH)
            print(f"Copied metadata from package to {DEFAULT_METADATA_PATH}")
            copied = True
        except Exception as e:
            print(f"Error copying metadata: {e}")
    
    return copied


def ensure_data_files():
    """Check for data files and download them if they don't exist"""
    vector_path = get_vector_store_path()
    metadata_path = get_metadata_path()
    
    # Check if files exist
    files_exist = vector_path.exists() and metadata_path.exists()
    if files_exist:
        return True
    
    # Create directories if needed
    ensure_base_dir()
    
    print("Downloading required data files...")
    try:
        import requests
        from tqdm import tqdm
        
        # Update this line in your ensure_data_files function:
        release_url = "https://github.com/danielberhane/FinVet-ACL-Demo/releases/download/v0.1.4"
        
        # Download vector store if needed
        if not vector_path.exists():
            vector_url = f"{release_url}/faiss_index.bin"
            print(f"Downloading vector store from {vector_url}...")
            
            response = requests.get(vector_url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(vector_path, 'wb') as f, tqdm(
                    desc="Vector store",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            
            print("Vector store download complete.")
        
        # Download metadata if needed
        if not metadata_path.exists():
            metadata_url = f"{release_url}/metadata.pkl"
            print(f"Downloading metadata from {metadata_url}...")
            
            response = requests.get(metadata_url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(metadata_path, 'wb') as f, tqdm(
                    desc="Metadata",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            
            print("Metadata download complete.")
        
        print("All required data files are now available.")
        return True
        
    except Exception as e:
        print(f"Error downloading data files: {e}")
        print("\nPlease download these files manually from:")
        # Change this line to point to your new repository
        print("https://github.com/danielberhane/FinVet-ACL-Demo/releases/latest")
        print(f"And place them in: {DEFAULT_VECTOR_STORE_PATH.parent}")
        return False