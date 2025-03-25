"""Command-line interface for the financial misinformation system."""

import argparse
import asyncio
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import tempfile

from financial_misinfo.system import FinancialMisinfoSystem
from financial_misinfo.utils.config import load_config, save_config, DEFAULT_CONFIG_PATH
from financial_misinfo.utils.visualization import print_results

# Add this at the top with your other imports
import financial_misinfo.utils.config as config_module


# Default training data path
DEFAULT_TRAINING_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    "data", 
    "fine-tune_validation.json"
)

async def build_index(training_data_path: Optional[str] = None, 
                      vector_store_path: Optional[str] = None, 
                      metadata_path: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build and save the index from training data.
    
    Args:
        training_data_path: Path to training data JSON file
        vector_store_path: Optional custom path for vector store
        metadata_path: Optional custom path for metadata
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with build information
    """
    # Use provided config or load configuration silently
    if config is None:
        config = load_config(verbose=False)
    
    # Initialize system
    system = FinancialMisinfoSystem(
        hf_token=config["hf_token"],
        google_api_key=config["google_api_key"]
    )
    
    # Use default path if no path is provided
    if training_data_path is None:
        training_data_path = DEFAULT_TRAINING_DATA
        print(f"Using default training data from {training_data_path}")
    
    # Ensure the file exists
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data file not found at {training_data_path}")
    
    # Use provided paths or config paths
    vector_path = vector_store_path or config.get("vector_store_path")
    meta_path = metadata_path or config.get("metadata_path")
    
    # Load training data
    training_data = await system.orchestrator.data_handler.load_json_file(training_data_path)
    
    # Prepare documents and build index
    documents = system.orchestrator.data_handler.prepare_documents(training_data)
    system.orchestrator.data_handler.build_index(documents)
    
    # Save index and metadata
    system.orchestrator.data_handler.save_index(vector_path, meta_path)
    
    # Return build information
    return {
        'timestamp': datetime.now().isoformat(),
        'training_data': training_data_path,
        'vector_store': vector_path,
        'metadata': meta_path,
        'num_documents': len(documents)
    }

async def verify_claim(claim: str, 
                      config: Optional[Dict[str, Any]] = None,
                      vector_store_path: Optional[str] = None,
                      metadata_path: Optional[str] = None) -> Dict:
    """Verify a single claim using the system."""

    # Use provided config or load it silently
    if config is None:
        config = load_config(verbose=False)
   
        
    system = FinancialMisinfoSystem(
        hf_token=config["hf_token"],
        google_api_key=config["google_api_key"]
    )
    
    # Load index if exists instead of building it
    try:
        # Use custom paths if provided, otherwise use config paths
        vector_path = vector_store_path or config.get("vector_store_path")
        meta_path = metadata_path or config.get("metadata_path")
        
        system.orchestrator.data_handler.load_index(
            vector_path=vector_path,
            metadata_path=meta_path
        )
        print(f"Loaded existing index from {vector_path}")
    except Exception as e:
        print(f"Could not load index: {e}")
        print("Building new index from training data...")
        # Fallback to building index if not found
        await build_index(
            vector_store_path=vector_path,
            metadata_path=meta_path,
            config=config
        )
    
    try:
        # Process claim with full pipeline
        result = await system.orchestrator.process_claim(claim)
        
        # Add verification metadata
        result['verification_info'] = {
            'timestamp': datetime.now().isoformat(),
            'vector_store': vector_path,
            'metadata': meta_path
        }
        
        return result
    except Exception as e:
        print(f"Error processing claim: {str(e)}")
        # Return an error result instead of None
        return {
            'error': str(e),
            'final_verdict': {
                'label': 'error',
                'evidence': f"Error: {str(e)}",
                'source': [],
                'confidence': 0.0
            }
        }

async def batch_verify(input_file: str, output_file: str, 
                      config: Optional[Dict[str, Any]] = None,
                      vector_store_path: Optional[str] = None, 
                      metadata_path: Optional[str] = None) -> None:
    """Verify multiple claims from a file."""
    # Use provided config or load it silently
    if config is None:
        config = load_config(verbose=False)
    
    system = FinancialMisinfoSystem(
        hf_token=config["hf_token"],
        google_api_key=config["google_api_key"]
    )
    
    # Load or build index
    try:
        # Use custom paths if provided, otherwise use config paths
        vector_path = vector_store_path or config.get("vector_store_path")
        meta_path = metadata_path or config.get("metadata_path")
        
        system.orchestrator.data_handler.load_index(
            vector_path=vector_path,
            metadata_path=meta_path
        )
        print(f"Loaded existing index from {vector_path}")
    except Exception:
        print("Building new index from training data...")
        await build_index(
            vector_store_path=vector_path,
            metadata_path=meta_path,
            config=config
        )
    
    # Load claims
    with open(input_file, 'r') as f:
        claims = json.load(f)
    
    results = await system.orchestrator.evaluate_batch(claims)
    
    # Add verification metadata
    for result in results:
        result['verification_info'] = {
            'timestamp': datetime.now().isoformat(),
            'vector_store': vector_path,
            'metadata': meta_path
        }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update config with values from command line arguments."""
    # Copy config to avoid modifying the original
    updated_config = config.copy()
    
    # Update with command line arguments if provided
    if hasattr(args, 'hf_token') and args.hf_token:
        updated_config["hf_token"] = args.hf_token
    
    if hasattr(args, 'google_api_key') and args.google_api_key:
        updated_config["google_api_key"] = args.google_api_key
    
    if hasattr(args, 'vector_store') and args.vector_store:
        updated_config["vector_store_path"] = args.vector_store
    
    if hasattr(args, 'metadata') and args.metadata:
        updated_config["metadata_path"] = args.metadata
    
    if hasattr(args, 'primary_model') and args.primary_model:
        if "models" not in updated_config:
            updated_config["models"] = {}
        updated_config["models"]["primary_llm"] = args.primary_model
    
    if hasattr(args, 'secondary_model') and args.secondary_model:
        if "models" not in updated_config:
            updated_config["models"] = {}
        updated_config["models"]["secondary_llm"] = args.secondary_model
    
    return updated_config

def main():
    """Main entry point for the CLI."""

    
    # Import the function at the top with other imports
    from financial_misinfo.utils.config import ensure_data_files

    # Ensure data files are available before proceeding with any operations
    if not ensure_data_files():
        print("Error: Required data files are missing.")
        return 1

    
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True)
    #print("DEBUG: Starting main()")


    parser = argparse.ArgumentParser(description="Financial Misinformation Detection System")
    
    # Add global arguments that apply to all commands
    parser.add_argument("--vector-store", help="Path to the vector store file")
    parser.add_argument("--metadata", help="Path to the metadata file")
    parser.add_argument("--hf-token", help="HuggingFace API token")
    parser.add_argument("--google-api-key", help="Google API key")
    parser.add_argument("--primary-model", help="Primary LLM model")
    parser.add_argument("--secondary-model", help="Secondary LLM model")
    parser.add_argument("--config", help="Path to config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build index command
    build_parser = subparsers.add_parser("build", help="Build and save the index")
    build_parser.add_argument("training_data", nargs='?', 
                               default=DEFAULT_TRAINING_DATA,
                               help="Path to training data JSON file (optional)")
    
    # Verify claim command
    verify_parser = subparsers.add_parser("verify", help="Verify a single claim")
    verify_parser.add_argument("claim", help="Claim text to verify")
    
    # Batch verification command
    batch_parser = subparsers.add_parser("batch", help="Verify multiple claims")
    batch_parser.add_argument("input", help="Input JSON file with claims")
    batch_parser.add_argument("output", help="Output file for results")
    
    # Configure command
    config_parser = subparsers.add_parser("config", help="Configure the system")
    config_parser.add_argument("--save", help="Save configuration to the specified file")

    # UI command - add this new section
    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit web interface")
    
    args = parser.parse_args()
    
    #print("DEBUG: About to load config")
    # Then load config once with the right verbosity
    if args.config:
        config = load_config(args.config, verbose=True)
    else:
        config = load_config(verbose=True)  # Show the message once
    
    #print(f"DEBUG: Config loaded, hf_token = '{config.get('hf_token')[:4]}...'")


    
    # Update config with command line arguments
    config = update_config_from_args(config, args)
    
    # Validate API keys AFTER all config loading and updates
    if args.command in ["verify", "batch"]:
        if not config.get("hf_token"):
            print("Error: HuggingFace API token is required")
            return 1
        if not config.get("google_api_key"):
            print("Error: Google API key is required")
            return 1
    
    # Execute command
    if args.command == "build":
        build_result = asyncio.run(build_index(
            args.training_data, 
            args.vector_store,
            args.metadata,
            config
        ))
        print(json.dumps(build_result, indent=2))
    
    elif args.command == "verify":
        result = asyncio.run(verify_claim(
        claim=args.claim, 
        config=config,
        vector_store_path=args.vector_store,
        metadata_path=args.metadata
    ))
        
        # Check if the result is valid before printing
        if result is None:
            print("Error: Verification failed. No results returned.")
            return 1
            
        print_results(result)
    
    elif args.command == "batch":
        asyncio.run(batch_verify(
            args.input, 
            args.output, 
            config,
            args.vector_store,
            args.metadata
        ))
        print(f"Results saved to {args.output}")
    
    elif args.command == "config":
        if args.hf_token:
            config["hf_token"] = args.hf_token
        if args.google_api_key:
            config["google_api_key"] = args.google_api_key
        
        save_config(config, args.save)
        print("Configuration updated")



    elif args.command == "ui":
        try:
            import streamlit.web.cli as stcli
            import sys
            from pathlib import Path
            
            # Find the minimal UI file
            # In cli.py, update:
            ui_file = Path(__file__).parent / "ui" / "enhanced_ui.py"
            if not ui_file.exists():
                print(f"Error: UI file not found at {ui_file}")
                return 1
            
            print(f"Launching Streamlit UI from {ui_file}")
            
            # Launch streamlit with appropriate parameters
            sys.argv = ["streamlit", "run", str(ui_file), 
                    "--server.port=8510", 
                    "--server.address=localhost"]
            
            stcli.main()
        except ImportError:
            print("Error: Streamlit is not installed. Please install it with: pip install streamlit")
            return 1

    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()