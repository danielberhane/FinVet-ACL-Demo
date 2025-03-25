# src/financial_misinfo/ui/enhanced_ui.py
import streamlit as st
import subprocess
import json
import re
import os
import time
import asyncio
import nest_asyncio
import traceback
from pathlib import Path
import tempfile

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Import for config handling
from financial_misinfo.utils.config import load_config, save_config
from financial_misinfo.system import FinancialMisinfoSystem


# Add this at the top of your file where other callbacks are defined
def on_change_claim():
    """Callback function when claim text changes"""
    st.session_state.claim_text = st.session_state.claim_input


# This function calls the CLI command and parses the result
def call_cli_verify(claim, hf_token, google_key, timeout=180):
    """Run the verification using the CLI command that we know works"""
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
        
        # First extract the main sections of the output to make parsing more robust
        final_verdict_section = re.search(r"FINAL VERDICT:(.*?)(?:COMPONENT DETAILS:|$)", stdout, re.DOTALL)
        component_details_section = re.search(r"COMPONENT DETAILS:(.*?)(?:=+$)", stdout, re.DOTALL)
        
        # Extract RAG A, RAG B, and FACT CHECK sections from component details
        if component_details_section:
            component_text = component_details_section.group(1)
            rag_a_section = re.search(r"RAG PIPELINE A:(.*?)(?:RAG PIPELINE B:|$)", component_text, re.DOTALL)
            rag_b_section = re.search(r"RAG PIPELINE B:(.*?)(?:FACT CHECK:|$)", component_text, re.DOTALL)
            fact_check_section = re.search(r"FACT CHECK:(.*?)(?:$)", component_text, re.DOTALL)
        else:
            rag_a_section = rag_b_section = fact_check_section = None
        
        # Parse final verdict
        if final_verdict_section:
            final_section_text = final_verdict_section.group(1)
            verdict_match = re.search(r"Label:\s*(\w+)", final_section_text, re.IGNORECASE)
            evidence_match = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", final_section_text, re.DOTALL)
            source_match = re.search(r"Source:\s*(.*?)(?:Confidence:|$)", final_section_text, re.DOTALL)
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", final_section_text)
        else:
            # Fallback to searching the entire output if we couldn't isolate the section
            verdict_match = re.search(r"FINAL VERDICT:\s*\n\s*Label:\s*(\w+)", stdout, re.IGNORECASE)
            evidence_match = re.search(r"FINAL VERDICT:.*?Evidence:\s*(.*?)(?:Source:|$)", stdout, re.DOTALL)
            source_match = re.search(r"FINAL VERDICT:.*?Source:\s*(.*?)(?:Confidence:|$)", stdout, re.DOTALL)
            confidence_match = re.search(r"FINAL VERDICT:.*?Confidence:\s*([\d.]+)", stdout, re.DOTALL)
        
        # Initialize the results structure with default values
        result = {
            "final_verdict": {
                "label": "unknown",
                "evidence": "No evidence provided",
                "source": [],
                "confidence": 0.0
            },
            "rag_A": {
                "label": "unknown",
                "evidence": "No evidence provided",
                "source": [],
                "confidence": 0.0
            },
            "rag_B": {
                "label": "unknown",
                "evidence": "No evidence provided",
                "source": [],
                "confidence": 0.0
            },
            "fact_check": {
                "label": "unknown",
                "evidence": "No evidence provided",
                "source": [],
                "confidence": 0.0
            }
        }
        
        # Populate final verdict if found
        if verdict_match:
            result["final_verdict"]["label"] = verdict_match.group(1).lower()
        if evidence_match:
            result["final_verdict"]["evidence"] = evidence_match.group(1).strip()
        if source_match:
            result["final_verdict"]["source"] = parse_source(source_match.group(1).strip())
        if confidence_match:
            result["final_verdict"]["confidence"] = float(confidence_match.group(1))
        
        # Parse RAG A
        if rag_a_section:
            section_text = rag_a_section.group(1)
            label_match = re.search(r"Label:\s*(\w+)", section_text, re.IGNORECASE)
            evidence_match = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", section_text, re.DOTALL)
            source_match = re.search(r"Source:\s*(.*?)(?:\n\n|$)", section_text, re.DOTALL)
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", section_text)
            
            if label_match:
                result["rag_A"]["label"] = label_match.group(1).lower()
            if evidence_match:
                result["rag_A"]["evidence"] = evidence_match.group(1).strip()
            if source_match:
                result["rag_A"]["source"] = parse_source(source_match.group(1).strip())
            if confidence_match:
                result["rag_A"]["confidence"] = float(confidence_match.group(1))
        
        # Parse RAG B
        if rag_b_section:
            section_text = rag_b_section.group(1)
            label_match = re.search(r"Label:\s*(\w+)", section_text, re.IGNORECASE)
            evidence_match = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", section_text, re.DOTALL)
            source_match = re.search(r"Source:\s*(.*?)(?:\n\n|$)", section_text, re.DOTALL)
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", section_text)
            
            if label_match:
                result["rag_B"]["label"] = label_match.group(1).lower()
            if evidence_match:
                result["rag_B"]["evidence"] = evidence_match.group(1).strip()
            if source_match:
                result["rag_B"]["source"] = parse_source(source_match.group(1).strip())
            if confidence_match:
                result["rag_B"]["confidence"] = float(confidence_match.group(1))
        
        # Parse Fact Check
        if fact_check_section:
            section_text = fact_check_section.group(1)
            label_match = re.search(r"Label:\s*(\w+)", section_text, re.IGNORECASE)
            evidence_match = re.search(r"Evidence:\s*(.*?)(?:Source:|$)", section_text, re.DOTALL)
            source_match = re.search(r"Source:\s*(.*?)(?:\n\n|$)", section_text, re.DOTALL)
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", section_text)
            
            if label_match:
                result["fact_check"]["label"] = label_match.group(1).lower()
            if evidence_match:
                result["fact_check"]["evidence"] = evidence_match.group(1).strip()
            if source_match:
                result["fact_check"]["source"] = parse_source(source_match.group(1).strip())
            if confidence_match:
                result["fact_check"]["confidence"] = float(confidence_match.group(1))
        
        # Include raw output for debugging
        result["_debug"] = {
            "stdout": stdout,
            "stderr": stderr
        }
        
        return result
    
    except Exception as e:
        return {
            "error": str(e),
            "details": traceback.format_exc()
        }

def parse_source(source_text):
    """Parse source text into a list or return as is"""
    if not source_text or source_text.lower() in ["no sources provided", "none"]:
        return []
    
    # Try to split by commas
    if "," in source_text:
        return [s.strip() for s in source_text.split(",")]
    
    return [source_text.strip()]



# Add this function to check for index files
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

# Page config
st.set_page_config(
    page_title="FinVet | Financial Misinfo Detector",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Fixed styling for verdict containers
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
    
    /* Evidence box */
    .evidence-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Source box */
    .source-box {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.95rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #E5E7EB;
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper function to run async functions in Streamlit
def run_async(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args, **kwargs))
    loop.close()
    return result

# Title
st.markdown('<h1 class="main-title">FinVet: Financial Misinformation Detector</h1>', unsafe_allow_html=True)

# Initialize session states
if 'history' not in st.session_state:
    st.session_state.history = []

if 'claim_text' not in st.session_state:
    st.session_state.claim_text = ""

# Configuration
config = load_config(verbose=False)

# Main layout
col1, col2 = st.columns([2, 3])

# Left column - Input
with col1:
    st.markdown('<div class="section-header">Verify Financial Claim</div>', unsafe_allow_html=True)
    
    # API credentials with default values from config
    with st.expander("API Credentials", expanded=False):
        # Use config values as defaults
        hf_token = st.text_input("HuggingFace Token", 
                                value=config.get('hf_token', ''),
                                type="password")
        google_key = st.text_input("Google API Key", 
                                  value=config.get('google_api_key', ''),
                                  type="password")
        
        # Save credentials to config button
        if st.button("Save Credentials"):
            config['hf_token'] = hf_token
            config['google_api_key'] = google_key
            save_config(config)
            st.success("Credentials saved to config file")
    
    # Advanced Settings
    with st.expander("Advanced Settings", expanded=False):
        vector_path = st.text_input("Vector Store Path", 
                          value=os.path.basename(config.get('vector_store_path', '')))
        metadata_path = st.text_input("Metadata Path", 
                            value=os.path.basename(config.get('metadata_path', '')))
        
        # Timeout setting
        timeout = st.slider("Timeout (seconds)", 30, 300, 180, 30)
        
        if st.button("Save Settings"):
            config['vector_store_path'] = os.path.basename(vector_path)
            config['metadata_path'] = os.path.basename(metadata_path)
            save_config(config)
            st.success("Settings saved to config file")
    # Claim input - use session state for the value
    # Claim input - use session state for the value
    claim = st.text_area("Enter financial claim to verify:", 
                        key="claim_input",
                        on_change=on_change_claim,
                        value=st.session_state.claim_text,
                        height=150, 
                        placeholder="Example: Tesla's stock price doubled in 2023.")
    
    # Example claims
    with st.expander("Try an example claim"):
        examples = [
            "Apple's stock price doubled in 2023",
            "Tesla achieved profitability for the first time in 2020",
            "Amazon acquired Whole Foods for $13 billion"
        ]
        
        # Create columns for better spacing
        col1_ex, col2_ex, col3_ex = st.columns(3)
        
        # Add buttons to columns with fixed rerun functionality
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
    
    # Verify button with disabled state
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
    
    if verify and claim:
        # Check index files first
        index_issues = check_index_files(config)
        if index_issues:
            st.error("Missing index files")
            for issue in index_issues:
                st.warning(issue)
            
            st.info("""
            Before verifying claims, you need to build the index. Run this command in your terminal:
            ```
            financial-misinfo build
            ```
            Or make sure your Vector Store Path and Metadata Path are correct in the Advanced Settings.
            """)
            
            # No need to continue processing
            st.stop()
        
        with st.spinner("Analyzing claim..."):
            try:
                # Process using real system API
                start_time = time.time()
                
                # Use CLI method directly for consistency
                # This avoids potential format discrepancies between API and CLI outputs
                result = call_cli_verify(claim, hf_token, google_key, timeout)
                
                end_time = time.time()
                
                # Handle error case
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                    with st.expander("Error Details"):
                        st.code(result.get('details', 'No details available'))
                    # No need to continue processing
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
                label = final_verdict.get('label', 'nei').lower()
                evidence = final_verdict.get('evidence', 'No evidence provided')
                confidence = final_verdict.get('confidence', 0)
                sources = final_verdict.get('source', [])
                
                # Format sources for display
                if isinstance(sources, list):
                    source_text = ", ".join(sources) if sources else "No sources available"
                else:
                    source_text = str(sources) if sources else "No sources available"
                
                # Display verdict with appropriate styling based on label
                # Using direct HTML/CSS to ensure correct rendering
                if label == "true":
                    verdict_html = f"""
                    <div style="background-color: #D1FAE5; color: #065F46; padding: 1.5rem; border-radius: 0.75rem; margin: 1rem 0; border: 1px solid #A7F3D0;">
                        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">TRUE</div>
                        <div style="margin: 0.75rem 0;">
                            <strong>Confidence:</strong> {confidence:.2f}
                        </div>
                    </div>
                    """
                elif label == "false":
                    verdict_html = f"""
                    <div style="background-color: #FEE2E2; color: #B91C1C; padding: 1.5rem; border-radius: 0.75rem; margin: 1rem 0; border: 1px solid #FECACA;">
                        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">FALSE</div>
                        <div style="margin: 0.75rem 0;">
                            <strong>Confidence:</strong> {confidence:.2f}
                        </div>
                    </div>
                    """
                else:  # nei or unknown
                    verdict_html = f"""
                    <div style="background-color: #FEF3C7; color: #92400E; padding: 1.5rem; border-radius: 0.75rem; margin: 1rem 0; border: 1px solid #FDE68A;">
                        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">NOT ENOUGH INFORMATION</div>
                        <div style="margin: 0.75rem 0;">
                            <strong>Confidence:</strong> {confidence:.2f}
                        </div>
                    </div>
                    """
                
                # Render verdict HTML
                st.markdown(verdict_html, unsafe_allow_html=True)
                
                # Confidence bar (additional visual)
                st.progress(min(confidence, 1.0))
                
                # Evidence section
                st.markdown("<strong>Evidence:</strong>", unsafe_allow_html=True)
                st.markdown(f'<div class="evidence-box">{evidence}</div>', unsafe_allow_html=True)
                
                # Source section
                st.markdown("<strong>Source:</strong>", unsafe_allow_html=True)
                st.markdown(f'<div class="source-box">{source_text}</div>', unsafe_allow_html=True)
                
                # Component details in tabs
                st.subheader("Component Details")
                components_tab1, components_tab2, components_tab3 = st.tabs(["RAG A", "RAG B", "Fact Check"])
                
                # Helper function for consistent component rendering
                def render_component(component, tab):
                    """Renders a component tab with improved error handling and consistent formatting"""
                    if not component:
                        tab.warning("No data available for this component")
                        return
                    
                    # Safely get values with defaults
                    comp_label = component.get('label', 'unknown').upper()
                    comp_confidence = component.get('confidence', 0)
                    comp_evidence = component.get('evidence', 'No evidence provided')
                    
                    # Format label with color based on value
                    if comp_label == "TRUE":
                        tab.markdown(f'<div style="background-color: #D1FAE5; color: #065F46; padding: 0.25rem 0.75rem; border-radius: 2rem; font-weight: 600; display: inline-block; font-size: 0.9rem;">TRUE</div>', unsafe_allow_html=True)
                    elif comp_label == "FALSE":
                        tab.markdown(f'<div style="background-color: #FEE2E2; color: #B91C1C; padding: 0.25rem 0.75rem; border-radius: 2rem; font-weight: 600; display: inline-block; font-size: 0.9rem;">FALSE</div>', unsafe_allow_html=True)
                    else:
                        tab.markdown(f'<div style="background-color: #FEF3C7; color: #92400E; padding: 0.25rem 0.75rem; border-radius: 2rem; font-weight: 600; display: inline-block; font-size: 0.9rem;">NOT ENOUGH INFO</div>', unsafe_allow_html=True)
                    
                    # Confidence
                    tab.write(f"**Confidence:** {comp_confidence:.2f}")
                    
                    # Format sources consistently
                    comp_sources = component.get('source', [])
                    if isinstance(comp_sources, list):
                        comp_source_text = ", ".join(comp_sources) if comp_sources else "No sources available"
                    else:
                        comp_source_text = str(comp_sources) if comp_sources else "No sources available"
                    
                    # Source
                    tab.write("**Source:**")
                    tab.markdown(f'<div class="source-box">{comp_source_text}</div>', unsafe_allow_html=True)
                    
                    # Evidence
                    tab.write("**Evidence:**")
                    
                    # Handle potential missing evidence
                    if not comp_evidence or comp_evidence == "No evidence provided":
                        tab.info("No detailed evidence available for this component")
                    else:
                        tab.markdown(f'<div class="evidence-box">{comp_evidence}</div>', unsafe_allow_html=True)
                
                # Render each component
                with components_tab1:
                    render_component(result.get('rag_A', {}), components_tab1)
                
                with components_tab2:
                    render_component(result.get('rag_B', {}), components_tab2)
                
                with components_tab3:
                    render_component(result.get('fact_check', {}), components_tab3)
                
                # Add debugging section
                with st.expander("Debug Information (For Developers)", expanded=False):
                    st.markdown("### Raw Result Structure")
                    st.json(result)
                    
                    if "_debug" in result:
                        st.markdown("### CLI Output")
                        st.code(result["_debug"]["stdout"], language="text")
                        
                        if result["_debug"]["stderr"]:
                            st.markdown("### CLI Error Output")
                            st.code(result["_debug"]["stderr"], language="text")
                
            except Exception as e:
                st.error(f"Error processing claim: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # History section
    if st.session_state.history:
        with st.expander("Previous Verifications", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                verdict = item["result"].get("final_verdict", {}).get("label", "unknown").lower()
                
                # Style based on verdict
                if verdict == "true":
                    st.success(f"**Claim:** {item['claim']}")
                elif verdict == "false":
                    st.error(f"**Claim:** {item['claim']}")
                else:
                    st.warning(f"**Claim:** {item['claim']}")
                
                confidence = item["result"].get("final_verdict", {}).get("confidence", 0)
                st.write(f"**Confidence:** {confidence:.2f}")
                
                # Add button to reload this claim
                if st.button(f"Reload claim {i+1}", key=f"reload_{i}"):
                    st.session_state.claim_text = item['claim']
                    st.rerun()
                
                st.markdown("---")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("FinVet: Financial Misinformation Detection System v0.1.2", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)