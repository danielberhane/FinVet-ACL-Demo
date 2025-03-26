import os
import sys
import warnings

# Disable warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Fix for PyTorch/Streamlit file watcher issue
try:
    import streamlit.watcher.local_sources_watcher as watcher
    original_extract_paths = watcher.extract_paths
    
    def patched_extract_paths(module):
        if module.__name__.startswith('torch.'):
            # Skip path extraction for torch modules
            return []
        return original_extract_paths(module)
    
    # Patch the function
    watcher.extract_paths = patched_extract_paths
except Exception:
    # If we can't patch, continue anyway
    pass

# Fix for torch JIT issues
try:
    import torch
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
except Exception:
    pass

# Now continue with the rest of your imports
import streamlit as st
import requests
import subprocess
import tempfile
import json
import traceback


# Configure page
st.set_page_config(
    page_title="FinVet | Financial Misinfo Detector",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import package
try:
    from financial_misinfo.utils.config import load_config, save_config
    package_imported = True
except ImportError as e:
    package_imported = False
    st.error(f"Could not import financial_misinfo package: {e}")
    st.info("Attempting to install package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        st.warning("Package installed, but you might need to refresh the page.")
        try:
            from financial_misinfo.utils.config import load_config, save_config
            package_imported = True
            st.success("Package imported successfully after installation!")
        except ImportError as e2:
            st.error(f"Still could not import package after installation: {e2}")
    except Exception as install_e:
        st.error(f"Failed to install package: {install_e}")



# Display debug information at the top for troubleshooting
with st.expander("Debug Information", expanded=False):
    st.write("### System Paths")
    st.code("\n".join(sys.path))
    
    st.write("### Working Directory")
    st.code(os.getcwd())
    
    st.write("### Current Directory Contents")
    try:
        st.code("\n".join(os.listdir(".")))
    except Exception as e:
        st.code(f"Error listing directory: {e}")
    
    st.write("### Src Directory Contents")
    try:
        if os.path.exists("src"):
            st.code("\n".join(os.listdir("src")))
        else:
            st.warning("src directory not found")
    except Exception as e:
        st.code(f"Error listing src directory: {e}")
    
    st.write("### Package Status")
    if package_imported:
        st.success("financial_misinfo package imported successfully")
    else:
        st.error("financial_misinfo package import failed")

# Simple download function
def download_index_files():
    """Download index files from GitHub release"""
    # Create directory for index files
    misinfo_dir = os.path.join(os.path.expanduser('~'), '.financial-misinfo')
    os.makedirs(misinfo_dir, exist_ok=True)
    
    # File paths
    vector_path = os.path.join(misinfo_dir, 'faiss_index.bin')
    metadata_path = os.path.join(misinfo_dir, 'metadata.pkl')

    # Skip if already downloaded
    if os.path.exists(vector_path) and os.path.exists(metadata_path):
        st.success(f"Index files already exist in {misinfo_dir}")
        st.write(f"Vector store: {os.path.getsize(vector_path)} bytes")
        st.write(f"Metadata: {os.path.getsize(metadata_path)} bytes")
        return True
    
    # GitHub release URL 
    release_url = "https://github.com/danielberhane/FinVet-ACL-Demo/releases/download/v0.1.4/"
    
    try:
        # Download vector store
        if not os.path.exists(vector_path):
            st.info(f"Downloading vector store index to {vector_path}...")
            response = requests.get(f"{release_url}faiss_index.bin")
            response.raise_for_status()
            with open(vector_path, 'wb') as f:
                f.write(response.content)
            st.success(f"Vector store downloaded! Size: {os.path.getsize(vector_path)} bytes")
        
        # Download metadata
        if not os.path.exists(metadata_path):
            st.info(f"Downloading metadata index to {metadata_path}...")
            response = requests.get(f"{release_url}metadata.pkl")
            response.raise_for_status()
            with open(metadata_path, 'wb') as f:
                f.write(response.content)
            st.success(f"Metadata downloaded! Size: {os.path.getsize(metadata_path)} bytes")
        
        return True
    
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        st.code(traceback.format_exc())
        return False
    


# Alternative verification that doesn't use CLI
def direct_verify(claim, hf_token, google_key):
    """Use direct API approach instead of CLI"""
    st.warning("Using direct verification method (no CLI)")
    try:
        # Try to import and use the system directly
        from financial_misinfo.system import FinancialMisinfoSystem
        
        # Set up config with credentials
        config = load_config(verbose=False)
        config['hf_token'] = hf_token
        config['google_api_key'] = google_key
        
        # Create system
        system = FinancialMisinfoSystem(config=config)
        
        # Verify claim
        result = system.verify(claim)
        
        return {
            "success": True,
            "result": result,
            "message": "Verification completed via direct API"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Simple CLI finder
def find_cli_tool():
    """Find financial-misinfo CLI tool"""
    import shutil
    
    # Check common locations
    paths = [
        shutil.which("financial-misinfo"),
        os.path.join(sys.prefix, 'bin', 'financial-misinfo'),
        os.path.join(os.path.expanduser('~'), '.local', 'bin', 'financial-misinfo'),
        "/app/.local/bin/financial-misinfo",  # Streamlit Cloud path
    ]
    
    # Log the paths being checked
    st.write("Looking for CLI tool in these locations:")
    for path in paths:
        if path:
            exists = os.path.exists(path)
            st.write(f"- {path}: {'✅ Found' if exists else '❌ Not found'}")

# Return first existing path
    for path in paths:
        if path and os.path.exists(path):
            return path
    
    # Install CLI if needed
    st.warning("CLI not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        st.success("Installation complete. Trying again...")
        
        # Check again
        cli_path = shutil.which("financial-misinfo")
        if cli_path and os.path.exists(cli_path):
            return cli_path
    except Exception as e:
        st.error(f"Installation failed: {e}")
    
    # Fall back to module approach 
    st.warning("CLI tool not found. Will try to use module approach.")
    return [sys.executable, "-m", "financial_misinfo"]

def main():
    # Title
    st.title("FinVet: Financial Misinformation Detector")
    
    # Initialize session state
    if 'claim_text' not in st.session_state:
        st.session_state.claim_text = ""
    
    # Download index files
    if not download_index_files():
        st.error("Failed to download required index files.")
        st.stop()
    
    # Load config and API keys
    try:
        config = load_config(verbose=False)
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        config = {}
    
    # Try to get API keys from Streamlit secrets
    hf_token = ""
    google_key = ""
    if hasattr(st, 'secrets'):
        hf_token = st.secrets.get('hf_token', config.get('hf_token', ''))
        google_key = st.secrets.get('google_api_key', config.get('google_api_key', ''))
    
    # Layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Verify Financial Claim")
        
        # API keys
        with st.expander("API Credentials"):
            hf_token = st.text_input("HuggingFace Token", value=hf_token, type="password")
            google_key = st.text_input("Google API Key", value=google_key, type="password")
        
        # Example claims
        with st.expander("Try an example claim"):
            examples = [
                "Apple's stock price doubled in 2023",
                "Tesla achieved profitability for the first time in 2020",
                "Amazon acquired Whole Foods for $13 billion"
            ]
            
            col1_ex, col2_ex, col3_ex = st.columns(3)
            with col1_ex:
                if st.button("Example 1"):
                    st.session_state.claim_text = examples[0]
                    st.rerun()
            with col2_ex:
                if st.button("Example 2"):
                    st.session_state.claim_text = examples[1]
                    st.rerun()
            with col3_ex:
                if st.button("Example 3"):
                    st.session_state.claim_text = examples[2]
                    st.rerun()
        
        # Claim input
        claim = st.text_area("Enter claim:", 
                            value=st.session_state.claim_text,
                            placeholder="Example: Tesla's stock price doubled in 2023")
        
        # Save to session state
        if claim != st.session_state.claim_text:
            st.session_state.claim_text = claim
        
        # Verify button
        verify_disabled = not (claim and hf_token and google_key)
        verify = st.button("Verify Claim", disabled=verify_disabled, type="primary")
    
    with col2:
        st.subheader("Verification Results")
        
        if verify and claim:
            with st.spinner("Analyzing claim..."):
                try:
                    # First try to find the CLI tool
                    cli_tool = find_cli_tool()
                    
                    # Create temp config file with credentials
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
                        config_data = {
                            "hf_token": hf_token,
                            "google_api_key": google_key,
                            **{k: v for k, v in config.items() 
                            if k not in ["hf_token", "google_api_key"]}
                        }
                        json.dump(config_data, temp_config)
                        config_path = temp_config.name
                    
                    # Try two approaches - first CLI/module, then direct
                    success = False
                    
                    # 1. Try CLI/module approach
                    try:
                        if isinstance(cli_tool, list):
                            st.warning("Using module approach. This might fail if the module can't be imported.")
                            cmd = cli_tool + ["--config", config_path, "verify", claim]
                        else:
                            st.success(f"Using CLI tool found at: {cli_tool}")
                            cmd = [cli_tool, "--config", config_path, "verify", claim]
                        
                        # Display the command being run
                        st.code(" ".join(cmd))
                        
                        # Try with a shorter timeout first
                        process = subprocess.Popen(
                            cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        st.info("Waiting for verification result...")
                        stdout, stderr = process.communicate(timeout=60)
                        
                        # Check if we got a result
                        if process.returncode == 0 and not stderr:
                            success = True
                            st.success("Verification complete!")
                            
                            # Display results based on output
                            if "true" in stdout.lower():
                                st.success("Claim appears to be TRUE")
                            elif "false" in stdout.lower():
                                st.error("Claim appears to be FALSE")
                            else:
                                st.warning("Not enough information to verify")
                            
                            # Show raw output
                            with st.expander("Raw Output"):
                                st.code(stdout)
                        else:
                            st.error("CLI/module verification failed")
                            if stderr:
                                st.code(stderr)
                    
                    except subprocess.TimeoutExpired:
                        process.kill()
                        st.warning("Verification timed out. This might be due to slow processing.")
                        success = False
                    except Exception as e:
                        st.error(f"Error with CLI/module approach: {e}")
                        success = False
                    
                    # 2. If CLI/module didn't work, try direct approach
                    if not success:
                        st.warning("Trying direct verification approach...")
                        
                        try:
                            # Try to directly import and use the system
                            from financial_misinfo.system import FinancialMisinfoSystem
                            
                            # Show progress
                            st.info("Initializing verification system...")
                            
                            # Create system with provided config
                            system = FinancialMisinfoSystem(config_data)
                            
                            # Verify claim
                            st.info(f"Verifying claim: {claim}")
                            result = system.verify(claim)
                            
                            # Show result
                            st.success("Verification complete!")
                            st.json(result)
                            
                            if "label" in result:
                                if result["label"].lower() == "true":
                                    st.success("Claim appears to be TRUE")
                                elif result["label"].lower() == "false":
                                    st.error("Claim appears to be FALSE")
                                else:
                                    st.warning("Not enough information to verify")
                            
                            if "evidence" in result:
                                st.subheader("Evidence")
                                st.write(result["evidence"])
                        except Exception as direct_error:
                            st.error(f"Direct verification failed: {direct_error}")
                            st.error(traceback.format_exc())
                except Exception as e:
                    st.error(f"Verification error: {str(e)}")
                    st.code(traceback.format_exc())
                finally:
                    # Clean up temp file
                    try:
                        if 'config_path' in locals():
                            os.unlink(config_path)
                    except Exception:
                        pass
    
    # Footer
    st.markdown("---")
    st.caption("FinVet: Financial Misinformation Detection System v0.1.4")

# Run the main function
if __name__ == "__main__":
    main()