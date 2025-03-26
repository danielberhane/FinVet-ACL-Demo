import os
import sys
import warnings
import time

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
def find_cli_tool(verbose=True):
    """Find financial-misinfo CLI tool"""
    import shutil
    
    # Check common locations
    paths = [
        shutil.which("financial-misinfo"),
        os.path.join(sys.prefix, 'bin', 'financial-misinfo'),
        os.path.join(os.path.expanduser('~'), '.local', 'bin', 'financial-misinfo'),
        "/app/.local/bin/financial-misinfo",  # Streamlit Cloud path
    ]
    
    # Log the paths being checked only if verbose
    if verbose:
        st.write("Looking for CLI tool in these locations:")
        for path in paths:
            if path:
                exists = os.path.exists(path)
                st.write(f"- {path}: {'✅ Found' if exists else '❌ Not found'}")
    
    # Return first existing path
    for path in paths:
        if path and os.path.exists(path):
            if verbose:
                st.success(f"Using CLI tool found at: {path}")
            return path
    
    # Install CLI if needed
    if verbose:
        st.warning("CLI not found. Attempting to install...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        
        if verbose:
            st.success("Installation complete. Trying again...")
        
        # Check again
        cli_path = shutil.which("financial-misinfo")
        if cli_path and os.path.exists(cli_path):
            return cli_path
    except Exception as e:
        if verbose:
            st.error(f"Installation failed: {e}")
    
    # Fall back to module approach 
    if verbose:
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
                    # Find CLI tool without displaying messages
                    cli_tool = find_cli_tool(verbose=False)  # Add verbose parameter
                    
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
                    
                    # Start timing verification process
                    start_time = time.time()
                    
                    # Try CLI/module approach without showing diagnostic messages
                    try:
                        if isinstance(cli_tool, list):
                            cmd = cli_tool + ["--config", config_path, "verify", claim]
                        else:
                            cmd = [cli_tool, "--config", config_path, "verify", claim]
                        
                        # Run verification process silently
                        process = subprocess.Popen(
                            cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        stdout, stderr = process.communicate(timeout=180)
                        
                        if process.returncode == 0:
                            # Success! Now process the results
                            end_time = time.time()
                            
                            # Show timing information
                            verification_time = end_time - start_time
                            st.success(f"Analysis completed in {verification_time:.2f} seconds")
                            
                            # Parse the output
                            import re
                            
                            # Extract final verdict label
                            verdict_match = re.search(r"Label:\s*(\w+)", stdout)
                            if verdict_match:
                                verdict = verdict_match.group(1).upper()
                                
                                # Display verdict with appropriate styling
                                if verdict == "TRUE":
                                    st.success(f"## {verdict}")
                                elif verdict == "FALSE":
                                    st.error(f"## {verdict}")
                                else:
                                    st.warning("## NOT ENOUGH INFO")
                            
                            # Extract confidence
                            confidence_match = re.search(r"Confidence:\s*([\d.]+)", stdout)
                            if confidence_match:
                                confidence = float(confidence_match.group(1))
                                st.write(f"**Confidence:** {confidence}")
                            
                            # Extract evidence
                            # Extract evidence
                            # Extract and process evidence
                            st.write("**Evidence:**")
                            evidence_match = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", stdout, re.DOTALL)
                            if evidence_match:
                                # Get raw evidence text
                                evidence = evidence_match.group(1).strip()
                                
                                # Fix the dollar sign issue by ensuring proper spacing
                                evidence = evidence.replace("$13", "$ 13")
                                
                                # Display the evidence
                                st.write(evidence)
                         
                           
                           
                           # Extract source
                            st.subheader("Source:")
                            source_match = re.search(r"Source:\s*(.*?)(?:COMPONENT DETAILS:|$)", stdout, re.DOTALL)
                            if source_match:
                                source = source_match.group(1).strip()
                                with st.container():
                                    st.write(source)
                            
                            # Create component details tabs
                            st.subheader("Component Details")
                            rag_a_tab, rag_b_tab, fact_check_tab = st.tabs(["RAG A", "RAG B", "Fact Check"])
                            
                            # RAG A Details
                            with rag_a_tab:
                                rag_a_section = re.search(r"RAG PIPELINE A:(.*?)(?:RAG PIPELINE B:|$)", stdout, re.DOTALL)
                                if rag_a_section:
                                    rag_a_text = rag_a_section.group(1)
                                    
                                    # Get RAG A verdict
                                    rag_a_verdict = re.search(r"Label:\s*(\w+)", rag_a_text)
                                    if rag_a_verdict:
                                        verdict = rag_a_verdict.group(1).upper()
                                        if verdict == "TRUE":
                                            st.success(verdict)
                                        elif verdict == "FALSE":
                                            st.error(verdict)
                                        else:
                                            st.warning("NOT ENOUGH INFO")
                                    
                                    # Get RAG A confidence
                                    rag_a_conf = re.search(r"Confidence:\s*([\d.]+)", rag_a_text)
                                    if rag_a_conf:
                                        st.write(f"**Confidence:** {rag_a_conf.group(1)}")

                        

                                    # Get RAG A evidence
                                    rag_a_evid = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", rag_a_text, re.DOTALL)
                                    if rag_a_evid:
                                        evidence = rag_a_evid.group(1).strip()
                                        st.write("**Evidence:**")
                                        st.write(evidence)

                                    # Get RAG A source
                                    rag_a_src = re.search(r"Source:\s*(.*?)(?:$|RAG PIPELINE B:)", rag_a_text, re.DOTALL)
                                    if rag_a_src:
                                        source = rag_a_src.group(1).strip()
                                        st.write("**Source:**")
                                        st.write(source)




                            # RAG B Details
                            with rag_b_tab:
                                rag_b_section = re.search(r"RAG PIPELINE B:(.*?)(?:FACT CHECK:|$)", stdout, re.DOTALL)
                                if rag_b_section:
                                    rag_b_text = rag_b_section.group(1)
                                    
                                    # Get RAG B verdict
                                    rag_b_verdict = re.search(r"Label:\s*(\w+)", rag_b_text)
                                    if rag_b_verdict:
                                        verdict = rag_b_verdict.group(1).upper()
                                        if verdict == "TRUE":
                                            st.success(verdict)
                                        elif verdict == "FALSE":
                                            st.error(verdict)
                                        else:
                                            st.warning("NOT ENOUGH INFO")
                                    
                                    # Get RAG B confidence
                                    rag_b_conf = re.search(r"Confidence:\s*([\d.]+)", rag_b_text)
                                    if rag_b_conf:
                                        st.write(f"**Confidence:** {rag_b_conf.group(1)}")
                                    
                                    # Get RAG B evidence
                                    rag_b_evid = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", rag_b_text, re.DOTALL)
                                    if rag_b_evid:
                                        evidence = rag_b_evid.group(1).strip()
                                        st.write("**Evidence:**")
                                        st.markdown(evidence)
                                    
                                    # Get RAG B source
                                    rag_b_src = re.search(r"Source:\s*(.*?)(?:$|FACT CHECK:)", rag_b_text, re.DOTALL)
                                    if rag_b_src:
                                        source = rag_b_src.group(1).strip()
                                        st.write("**Source:**")
                                        st.markdown(source)
                            
                            # Fact Check Details
                            with fact_check_tab:
                                fact_section = re.search(r"FACT CHECK:(.*?)(?:======|$)", stdout, re.DOTALL)
                                if fact_section:
                                    fact_text = fact_section.group(1)
                                    
                                    # Get Fact Check verdict
                                    fact_verdict = re.search(r"Label:\s*(\w+)", fact_text)
                                    if fact_verdict:
                                        verdict = fact_verdict.group(1).upper()
                                        if verdict == "TRUE":
                                            st.success(verdict)
                                        elif verdict == "FALSE":
                                            st.error(verdict)
                                        else:
                                            st.warning("NOT ENOUGH INFO")
                                    
                                    # Get Fact Check confidence
                                    fact_conf = re.search(r"Confidence:\s*([\d.]+)", fact_text)
                                    if fact_conf:
                                        st.write(f"**Confidence:** {fact_conf.group(1)}")
                                    
                                    # Get Fact Check evidence
                                    fact_evid = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", fact_text, re.DOTALL)
                                    if fact_evid:
                                        evidence = fact_evid.group(1).strip()
                                        st.write("**Evidence:**")
                                        st.markdown(evidence)
                                    
                                    # Get Fact Check source
                                    fact_src = re.search(r"Source:\s*(.*?)(?:$|======)", fact_text, re.DOTALL)
                                    if fact_src:
                                        source = fact_src.group(1).strip()
                                        st.write("**Source:**")
                                        st.markdown(source)
                            
                            # Show raw output
                            with st.expander("Raw Output", expanded=False):
                                st.code(stdout)
                        
                        else:
                            # CLI command failed
                            st.error("Verification failed")
                            if stderr:
                                st.code(stderr)
                            
                            # Try direct verification as a fallback
                            st.warning("Trying direct verification method...")
                            
                            try:
                                # Import the system class
                                from financial_misinfo.system import FinancialMisinfoSystem
                                
                                # Create the system with the config
                                system = FinancialMisinfoSystem(config_data)
                                
                                # Run verification
                                st.info(f"Verifying claim: {claim}")
                                result = system.verify(claim)
                                
                                # Display results
                                st.success("Verification complete!")
                                st.write(result)
                            except Exception as direct_error:
                                st.error(f"Direct verification also failed: {str(direct_error)}")
                                st.code(traceback.format_exc())
                    
                    except subprocess.TimeoutExpired:
                        process.kill()
                        st.error("Verification process timed out.")
                        
                    except Exception as cmd_error:
                        st.error(f"Error running verification: {str(cmd_error)}")
                        st.code(traceback.format_exc())
                
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