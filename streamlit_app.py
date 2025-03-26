import os
import sys
import warnings
import logging
logging.basicConfig(level=logging.DEBUG)

# Disable specific warnings and set environment variables early
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Patch torch-related issues before other imports
try:
    import torch
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
except:
    pass

# Workaround for Streamlit's source file watcher
try:
    import streamlit.watcher.local_sources_watcher as watcher
    def dummy_extract_paths(module):
        return []
    watcher.extract_paths = dummy_extract_paths
except:
    pass

# Rest of the standard imports
import streamlit as st
import tempfile
import json
import subprocess
import traceback


# Rest of your imports
import streamlit as st
import warnings
import traceback
import tempfile
import json
import subprocess
import requests

# Set page configuration
st.set_page_config(
    page_title="FinVet | Financial Misinfo Detector",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root and src to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Custom imports from the project
from financial_misinfo.utils.config import load_config, save_config

# Function to download index files from GitHub release
def download_index_files():
    """
    Download index files from GitHub release if not already present
    """
    # Paths where index files will be stored
    misinfo_dir = os.path.join(os.path.expanduser('~'), '.financial-misinfo')
    os.makedirs(misinfo_dir, exist_ok=True)
    
    vector_store_path = os.path.join(misinfo_dir, 'faiss_index.bin')
    metadata_path = os.path.join(misinfo_dir, 'metadata.pkl')
    
    # Check if files already exist
    if os.path.exists(vector_store_path) and os.path.exists(metadata_path):
        return True
    
    try:
        # GitHub release URL (replace with your actual release URL)
        release_base_url = "https://github.com/danielberhane/FinVet-ACL-Demo/releases/download/v0.1.4/"
    
        
        # Download vector store
        if not os.path.exists(vector_store_path):
            st.info("Downloading vector store index...")
            vector_response = requests.get(f"{release_base_url}faiss_index.bin")
            vector_response.raise_for_status()  # Raise an error for bad responses
            with open(vector_store_path, 'wb') as f:
                f.write(vector_response.content)
        
        # Download metadata
        if not os.path.exists(metadata_path):
            st.info("Downloading metadata index...")
            metadata_response = requests.get(f"{release_base_url}metadata.pkl")
            metadata_response.raise_for_status()  # Raise an error for bad responses
            with open(metadata_path, 'wb') as f:
                f.write(metadata_response.content)
        
        st.success("Index files downloaded successfully!")
        return True
    
    except Exception as e:
        st.error(f"Failed to download index files: {e}")
        return False

# Callback function for claim input
def on_change_claim():
    """Callback function when claim text changes"""
    st.session_state.claim_text = st.session_state.claim_input



def call_cli_verify(claim, hf_token, google_key, timeout=180):
    try:
        import shutil
        import sys
        import logging

        # Extensive logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Log system information
        logger.debug(f"Python Executable: {sys.executable}")
        logger.debug(f"Current Working Directory: {os.getcwd()}")
        logger.debug(f"System PATH: {os.environ.get('PATH', '')}")

        # Check for the CLI tool in multiple locations
        def find_executable(name):
            """Find executable in system PATH and common locations"""
            locations = [
                shutil.which(name),  # Search in PATH
                os.path.join(sys.prefix, 'bin', name),  # Virtual env
                os.path.join(sys.base_prefix, 'bin', name),  # Base Python bin
                f"/home/adminuser/.local/bin/{name}",  # User local bin
                f"/usr/local/bin/{name}",  # System bin
            ]
            return next((path for path in locations if path and os.path.exists(path)), None)

        # Find the executable
        cli_path = find_executable('financial-misinfo')
        logger.debug(f"Found CLI at: {cli_path}")

        # If no CLI found, try module-based approach
        if not cli_path:
            try:
                # Attempt to run as a Python module
                import importlib.util
                spec = importlib.util.find_spec('financial_misinfo')
                if spec:
                    cli_path = sys.executable
                    cmd = [cli_path, '-m', 'financial_misinfo', 'verify']
                else:
                    return {
                        "error": "CLI not found",
                        "details": "Could not locate financial-misinfo executable or module"
                    }
            except Exception as import_error:
                return {
                    "error": "Module import failed",
                    "details": str(import_error)
                }
        else:
            cmd = [cli_path]

        # Create temporary config file with credentials
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
            config_data = {
                "hf_token": hf_token,
                "google_api_key": google_key,
                **{k: v for k, v in load_config(verbose=False).items() 
                   if k not in ["hf_token", "google_api_key"]}
            }
            json.dump(config_data, temp_config)
            temp_config_path = temp_config.name

        # Complete the command
        full_cmd = cmd + ["--config", temp_config_path, "verify", claim]
        logger.debug(f"Full command: {' '.join(full_cmd)}")

        try:
            # Run the command with enhanced error handling
            process = subprocess.Popen(
                full_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, PYTHONWARNINGS='ignore')
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Log raw output
                logger.debug(f"STDOUT: {stdout}")
                logger.debug(f"STDERR: {stderr}")

                # Check for errors
                if stderr and "error" in stderr.lower():
                    return {
                        "error": "Verification error",
                        "details": stderr
                    }

                # Default result with logging
                return {
                    "final_verdict": {
                        "label": "unknown",
                        "evidence": "No evidence provided",
                        "source": [],
                        "confidence": 0.0
                    }
                }

            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "error": f"Verification timed out after {timeout} seconds",
                    "details": "Process took too long to complete"
                }
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_config_path)
                except:
                    pass

        except Exception as exec_error:
            return {
                "error": "Execution failed",
                "details": str(exec_error)
            }

    except Exception as critical_error:
        return {
            "error": "Critical verification error",
            "details": traceback.format_exc()
        }


"""
# Function to call CLI verification
def call_cli_verify(claim, hf_token, google_key, timeout=180):
    
    try:
        # Create temporary config file with credentials
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_config:
            config_data = {
                "hf_token": hf_token,
                "google_api_key": google_key,
                # Copy other settings from the loaded config
                **{k: v for k, v in load_config(verbose=False).items() 
                   if k not in ["hf_token", "google_api_key"]}
            }
            json.dump(config_data, temp_config)
            temp_config_path = temp_config.name
        
        # Create the CLI command
        cmd = [
            "financial-misinfo",
            "--config", temp_config_path,
            "verify",
            claim
        ]
        
        # Run the command with timeout
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return {
                "error": f"Verification timed out after {timeout} seconds",
                "details": stdout + stderr
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_config_path)
            except:
                pass
                
        if stderr and "error" in stderr.lower():
            return {
                "error": "Error during verification",
                "details": stderr
            }
        
        # Parsing logic for the verification result
        # This is a simplified version - you might want to add more detailed parsing
        result = {
            "final_verdict": {
                "label": "unknown",
                "evidence": "No evidence provided",
                "source": [],
                "confidence": 0.0
            }
        }
        
        return result
    
    except Exception as e:
        return {
            "error": str(e),
            "details": traceback.format_exc()
        }
"""

# Function to check index files
def check_index_files(config):
    """Check if the required index files exist"""
    vector_path = config.get('vector_store_path')
    metadata_path = config.get('metadata_path')
    
    issues = []
    
    if not vector_path or not os.path.exists(vector_path):
        issues.append(f"Vector store file not found at {vector_path}")
    
    if not metadata_path or not os.path.exists(metadata_path):
        issues.append(f"Metadata file not found at {metadata_path}")
    
    return issues



# Custom CSS for the application
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #1E3A8A;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
        color: #1F2937;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Secrets and configuration handling
def get_secrets():
    """Retrieve API keys from Streamlit Cloud secrets or config"""
    config = load_config(verbose=False)
    
    # Check Streamlit secrets first
    if hasattr(st, 'secrets'):
        hf_token = st.secrets.get('hf_token', config.get('hf_token', ''))
        google_key = st.secrets.get('google_api_key', config.get('google_api_key', ''))
    else:
        hf_token = config.get('hf_token', '')
        google_key = config.get('google_api_key', '')
    
    return hf_token, google_key

# Main Streamlit app function
def main():
    # Download index files before anything else
    if not download_index_files():
        st.error("Could not download index files. Please check your internet connection.")
        st.stop()

    # Title
    st.markdown('<h1 class="main-title">FinVet: Financial Misinformation Detector</h1>', unsafe_allow_html=True)

    # Initialize session states
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'claim_text' not in st.session_state:
        st.session_state.claim_text = ""

    # Get configuration and secrets
    config = load_config(verbose=False)
    hf_token, google_key = get_secrets()

    # Main layout
    col1, col2 = st.columns([2, 3])

    # Left column - Input
    with col1:
        st.markdown('<div class="section-header">Verify Financial Claim</div>', unsafe_allow_html=True)
        
        # API credentials with default values from config/secrets
        with st.expander("API Credentials", expanded=False):
            hf_token = st.text_input("HuggingFace Token", 
                                    value=hf_token,
                                    type="password")
            google_key = st.text_input("Google API Key", 
                                      value=google_key,
                                      type="password")
            
            # Save credentials to config button
            if st.button("Save Credentials"):
                config['hf_token'] = hf_token
                config['google_api_key'] = google_key
                save_config(config)
                st.success("Credentials saved to config file")
        
        # Example claims
        with st.expander("Try an example claim"):
            examples = [
                "Apple's stock price doubled in 2023",
                "Tesla achieved profitability for the first time in 2020",
                "Amazon acquired Whole Foods for $13 billion"
            ]
            
            # Create columns for better spacing
            col1_ex, col2_ex, col3_ex = st.columns(3)
            
            # Add buttons to columns
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
        claim = st.text_area("Enter financial claim to verify:", 
                            key="claim_input",
                            on_change=on_change_claim,
                            value=st.session_state.claim_text,
                            height=150, 
                            placeholder="Example: Tesla's stock price doubled in 2023.")
        
        # Verify button logic
        verify_disabled = not (claim and hf_token and google_key)
        if not claim:
            st.info("Enter a claim to verify")
        elif not (hf_token and google_key):
            st.info("Enter API credentials to verify")
        else:
            st.info("Ready to verify. Click the button to analyze the claim.")
            
        verify = st.button("Verify Claim", disabled=verify_disabled, 
                        type="primary", use_container_width=True)

    # Right column - Results
    with col2:
        st.markdown('<div class="section-header">Verification Results</div>', unsafe_allow_html=True)
        
        # Verification logic
        if verify and claim:
            # Check index files first
            index_issues = check_index_files(config)
            if index_issues:
                st.error("Missing index files")
                for issue in index_issues:
                    st.warning(issue)
                st.stop()
            
            # Verification processing
            with st.spinner("Analyzing claim..."):
                try:
                    # Process the claim
                    start_time = time.time()
                    
                    # Call CLI verification
                    result = call_cli_verify(claim, hf_token, google_key)
                    
                    end_time = time.time()
                    
                    # Handle error case
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        with st.expander("Error Details"):
                            st.code(result.get('details', 'No details available'))
                        st.stop()
                    
                    # Store in history
                    st.session_state.history.append({
                        "claim": claim,
                        "result": result
                    })
                    
                    # Display timing information
                    st.success(f"Analysis completed in {end_time - start_time:.2f} seconds")
                    
                    # Extract verdict info
                    final_verdict = result.get('final_verdict', {})
                    label = final_verdict.get('label', 'unknown').lower()
                    
                    # Verdict display based on label
                    if label == "true":
                        st.success("Claim appears to be TRUE")
                    elif label == "false":
                        st.error("Claim appears to be FALSE")
                    else:
                        st.warning("Not enough information to verify the claim")
                    
                    # Display available verdict details
                    st.write("**Evidence:**", final_verdict.get('evidence', 'No evidence provided'))
                    st.write("**Confidence:**", f"{final_verdict.get('confidence', 0):.2f}")
                    
                except Exception as e:
                    st.error(f"Error processing claim: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    # Footer
    st.markdown('<div class="footer">FinVet: Financial Misinformation Detection System v0.1.4</div>', unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()