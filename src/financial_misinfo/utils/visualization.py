"""Visualization utilities for the financial misinformation system."""

from typing import Dict, List

def print_results(results: Dict) -> None:
    """Format and print the complete results of claim verification."""
    terminal_width = 200  # Adjust based on your terminal size
    
    # Helper function to format evidence text (truncate if needed)
    def format_evidence(evidence, max_length=100):
        if not evidence:
            return "No evidence provided"
        if len(evidence) > max_length:
            return evidence[:max_length] + "..."
        return evidence
    
    # Helper function to format sources
    def format_sources(sources):
        if not sources:
            return "No sources provided"
        
        if isinstance(sources, list):
            if not sources:
                return "No sources provided"
            elif len(sources) == 1:
                return sources[0]
            else:
                return ", ".join(sources)
        return str(sources)
    
    # Print header
    print("\n" + "="*terminal_width)
    print("VERIFICATION RESULTS".center(terminal_width))
    print("="*terminal_width)
    
    # Print final verdict
    final = results.get('final_verdict', {})
    print("\nFINAL VERDICT:")
    print(f"  Label: {final.get('label', 'unknown').upper()}")
    print(f"  Confidence: {final.get('confidence', 0.0):.2f}")
    print(f"  Evidence: {final.get('evidence', 'No evidence provided')}")
    print(f"  Source: {format_sources(final.get('source', []))}")
    
    # Print components
    components = [
        ("RAG PIPELINE A", results.get('rag_A', {})),
        ("RAG PIPELINE B", results.get('rag_B', {})),
        ("FACT CHECK", results.get('fact_check', {}))
    ]
    
    print("\nCOMPONENT DETAILS:")
    print("-"*terminal_width)
    
    for title, component in components:
        print(f"\n{title}:")
        print(f"  Label: {component.get('label', 'unknown').upper()}")
        print(f"  Confidence: {component.get('confidence', 0.0):.2f}")
        print(f"  Evidence: {format_evidence(component.get('evidence', 'No evidence provided'))}")
        print(f"  Source: {format_sources(component.get('source', []))}")
    
    # Print footer
    print("\n" + "="*terminal_width)

def print_pipeline_results_compact(results: Dict) -> None:
    """Print pipeline results in a compact, table-like format."""
    print("\n" + "="*170)
    print("PIPELINE RESULTS".center(170))
    print("="*170)
    
    # Print header
    print(f"{'Pipeline':<12} | {'Label':<6} | {'Confidence':<10} | {'Evidence':<30} | Source")
    print("-" * 170)
    
    # Define pipeline order
    pipeline_order = ['rag_A', 'rag_B', 'fact_check']
    
    # Print each pipeline's results
    for pipeline in pipeline_order:
        if pipeline in results:
            result = results[pipeline]
            evidence = result.get('evidence', 'N/A')
            if len(evidence) > 30:
                evidence = evidence[:27] + "..."
                
            print(f"{pipeline:<12} | {result.get('label', 'N/A'):<6} | {result.get('confidence', 'N/A'):<10.2f} | {evidence:<30} | {result.get('source', [])}")
    
    print("="*170)



import os
import json
from datetime import datetime
from typing import Dict, Any

def save_result_to_file(formatted_result: Dict[str, Any], file_path: str = "detailed_results.txt", json_path: str = "results.json"):
    """Save a single result to both text and JSON files."""
    # Create file if it doesn't exist and write header
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            header = "=" * 100 + "\n"
            header += "FINANCIAL MISINFORMATION DETECTION RESULTS\n"
            header += "=" * 100 + "\n\n"
            header += f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f.write(header)
            
    # Get claim number based on file content
    claim_number = 1
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            claim_number = content.count("CLAIM #") + 1
    
    # Format sources
    if isinstance(formatted_result['predicted_source'], list):
        sources = ", ".join(formatted_result['predicted_source']) if formatted_result['predicted_source'] else "None"
    else:
        sources = str(formatted_result['predicted_source'])
    
    # Create formatted text
    text = f"CLAIM #{claim_number}\n"
    text += "=" * 80 + "\n\n"
    
    # Original claim details
    text += "CLAIM TEXT:\n"
    text += f"{formatted_result['claim']}\n\n"
    text += f"TRUE LABEL: {formatted_result['true_label'].upper()}\n"
    
    # Limit evidence length for readability
    true_evidence = formatted_result['true_evidence']
    if len(true_evidence) > 300:
        true_evidence = true_evidence[:297] + "..."
    
    text += "TRUE EVIDENCE:\n"
    text += f"{true_evidence}\n\n"
    text += "-" * 80 + "\n\n"
    
    # Final verdict
    text += "FINAL VERDICT\n"
    text += "-" * 80 + "\n"
    text += f"PREDICTED LABEL: {formatted_result['predicted_label'].upper()}\n"
    text += f"CONFIDENCE: {formatted_result['predicted_confidence']:.2f}\n"
    text += f"SOURCES: {sources}\n\n"
    
    # Prediction evidence
    predicted_evidence = formatted_result['predicted_evidence']
    if len(predicted_evidence) > 300:
        predicted_evidence = predicted_evidence[:297] + "..."
    
    text += "PREDICTION EVIDENCE:\n"
    text += f"{predicted_evidence}\n\n"
    text += "-" * 80 + "\n\n"
    
    # Component results
    text += "COMPONENT RESULTS\n"
    text += "-" * 80 + "\n"
    
    # RAG A
    text += "RAG A:\n"
    text += f"  Label: {formatted_result['rag_A_label'].upper()}\n"
    
    rag_a_evidence = formatted_result['rag_A_evidence']
    if len(rag_a_evidence) > 200:
        rag_a_evidence = rag_a_evidence[:197] + "..."
    
    text += f"  Evidence: {rag_a_evidence}\n\n"
    
    # RAG B
    text += "RAG B:\n"
    text += f"  Label: {formatted_result['rag_B_label'].upper()}\n"
    
    rag_b_evidence = formatted_result['rag_B_evidence']
    if len(rag_b_evidence) > 200:
        rag_b_evidence = rag_b_evidence[:197] + "..."
    
    text += f"  Evidence: {rag_b_evidence}\n\n"
    
    # Fact Check
    text += "Fact Check:\n"
    text += f"  Label: {formatted_result['fact_check_label'].upper()}\n"
    
    fact_check_evidence = formatted_result['fact_check_evidence']
    if len(fact_check_evidence) > 200:
        fact_check_evidence = fact_check_evidence[:197] + "..."
    
    text += f"  Evidence: {fact_check_evidence}\n\n"
    
    # End separator
    text += "=" * 80 + "\n\n"
    
    # Append to text file
    with open(file_path, 'a') as f:
        f.write(text)
    
    # Handle JSON file
    json_data = []
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            json_data = []
    
    json_data.append(formatted_result)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)