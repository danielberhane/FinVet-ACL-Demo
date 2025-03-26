import os
import sys
import streamlit as st
import requests
import subprocess
import tempfile
import json

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
except ImportError:
    st.error("Could not import financial_misinfo package. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        st.success("Package installed. Please refresh the page.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to install package: {e}")
        st.stop()

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
        return True
    
    # GitHub release URL 
    release_url = "https://github.com/danielberhane/FinVet-ACL-Demo/releases/download/v0.1.4/"
    
    try:
        # Download vector store
        if not os.path.exists(vector_path):
            st.info("Downloading vector store index...")
            response = requests.get(f"{release_url}faiss_index.bin")
            response.raise_for_status()
            with open(vector_path, 'wb') as f:
                f.write(response.content)
            st.success("Vector store downloaded!")
        
        # Download metadata
        if not os.path.exists(metadata_path):
            st.info("Downloading metadata index...")
            response = requests.get(f"{release_url}metadata.pkl")
            response.raise_for_status()
            with open(metadata_path, 'wb') as f:
                f.write(response.content)
            st.success("Metadata downloaded!")
        
        return True
    
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

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
    
    # Return first existing path
    for path in paths:
        if path and os.path.exists(path):
            return path
    
    # Fall back to module approach
    return [sys.executable, "-m", "financial_misinfo"]

# Main function
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
    config = load_config(verbose=False)
    
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
                    # Find CLI tool
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
                    
                    try:
                        # Create command
                        if isinstance(cli_tool, list):
                            cmd = cli_tool + ["--config", config_path, "verify", claim]
                        else:
                            cmd = [cli_tool, "--config", config_path, "verify", claim]
                        
                        # Run verification
                        process = subprocess.Popen(
                            cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        stdout, stderr = process.communicate(timeout=180)
                        
                        # Check for errors
                        if process.returncode != 0 or stderr:
                            st.error("Verification failed")
                            with st.expander("Error Details"):
                                st.code(stderr)
                            st.stop()
                        
                        # Display results (simplified)
                        st.success("Verification complete!")
                        
                        # Try to parse the verdict
                        if "true" in stdout.lower():
                            st.success("Claim appears to be TRUE")
                        elif "false" in stdout.lower():
                            st.error("Claim appears to be FALSE")
                        else:
                            st.warning("Not enough information to verify")
                        
                        # Display raw output
                        with st.expander("Raw Output"):
                            st.code(stdout)
                        
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(config_path)
                        except:
                            pass
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("FinVet: Financial Misinformation Detection System v0.1.4")

# Run the app
if __name__ == "__main__":
    main()