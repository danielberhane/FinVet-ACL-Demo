import os
import warnings
import logging
import json
import uuid
import time
import asyncio
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import requests
from dataclasses import dataclass, field
from tqdm import tqdm
from langchain_community.llms import HuggingFaceEndpoint
import math
import traceback
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
import csv
import pickle
from typing import List, Dict, Any

# Missing imports
from financial_misinfo.agents.data_handler import DataHandlerAgent
from financial_misinfo.agents.rag_pipeline import RAGPipeline
from financial_misinfo.agents.fact_check import FactCheckPipeline
from financial_misinfo.agents.result_collector import ResultCollectorAgent
from financial_misinfo.agents.voting import VotingAgent
from financial_misinfo.utils.visualization import print_results

class OrchestratorAgent:
    def __init__(self, hf_token: str, google_api_key: str, 
             primary_model: str = "meta-llama/Llama-3.3-70B-Instruct", 
             secondary_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
    
        self.data_handler = DataHandlerAgent()
        
        self.rag_A = RAGPipeline(
            model=primary_model,
            hf_token=hf_token,
            data_handler=self.data_handler
        )
        
        self.rag_B = RAGPipeline(
            model=secondary_model,
            hf_token=hf_token,
            data_handler=self.data_handler
        )
        
        self.fact_check = FactCheckPipeline(
            model=primary_model,  # Using primary model for fact check
            google_api_key=google_api_key,
            hf_token=hf_token
        )
        
        self.result_collector = ResultCollectorAgent()
        self.voting_system = VotingAgent()
        self.status_store = {}
        self.message_queue = asyncio.Queue()
    
        
    # Updated process_claim method for OrchestratorAgent

    async def process_claim(self, claim: str, task_id: str = None) -> Dict:
        task_id = task_id or str(uuid.uuid4())
        
        try:
            # Run all pipelines
            results = {}
            errors = []

            # Process each pipeline separately with error handling
            try:
                result_a = await self.rag_A.process(claim)
                result_a['pipeline'] = 'rag_A'
                results['rag_A'] = result_a
            except Exception as e:
                error_msg = f"RAG A error: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                results['rag_A'] = self._create_error_result('rag_A', e)

            try:
                result_b = await self.rag_B.process(claim)
                result_b['pipeline'] = 'rag_B'
                results['rag_B'] = result_b
            except Exception as e:
                error_msg = f"RAG B error: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                results['rag_B'] = self._create_error_result('rag_B', e)
            
            try:
                results['fact_check'] = await self.fact_check.process(claim)
            except Exception as e:
                error_msg = f"Fact Check error: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                results['fact_check'] = self._create_error_result('fact_check', e)
            
            # Check if all pipelines failed with vector store errors
            vector_store_errors = [err for err in errors if 'vector store' in err.lower() or 'no attribute' in err.lower()]
            if len(vector_store_errors) >= 2:  # If at least 2 RAG pipelines failed with vector store errors
                return {
                    'task_id': task_id,
                    'claim': claim,
                    'error': "Vector store initialization failed. Please run 'financial-misinfo build' first.",
                    'technical_details': "\n".join(vector_store_errors),
                    'status': 'failed'
                }
            
            normalized_results = self.result_collector.collect(list(results.values()))
            pre_final_verdict = self.voting_system.vote(normalized_results)
            
            return {
                'task_id': task_id,
                'claim': claim,
                'final_verdict': pre_final_verdict['final_verdict'],
                'rag_A': pre_final_verdict['rag_A'],
                'rag_B': pre_final_verdict['rag_B'],
                'fact_check': pre_final_verdict['fact_check'],
                'status': 'completed'
            }

        except Exception as e:
            print(f"Error processing claim: {str(e)}")
            return {
                'task_id': task_id,
                'claim': claim,
                'error': str(e),
                'status': 'failed'
            }
            
    
    def _create_error_result(self, pipeline: str, error: Exception) -> Dict:
        return {
            'pipeline': pipeline,
            'label': 'nei',
            'evidence': f"Error in {pipeline}: {str(error)}",
            'confidence': 0.0
        }
    
    
    async def evaluate_batch(self, test_data: List[Dict]) -> List[Dict]:
        # print(f"Starting batch evaluation of {len(test_data)} claims")
        results = []

        for entry in tqdm(test_data, desc="Processing claims"):
            print("Calling process_claim .......")
            print (entry['claim'])
            
            result = await self.process_claim(entry['claim'])
            # print("Process claim result:", result)  # Debug print

            if 'error' in result:
                # Handle error case
                results.append({
                    'claim': entry['claim'],
                    'true_label': entry.get('label', 'unknown'),
                    'true_evidence': entry.get('evidence', 'No evidence provided'),
                    'predicted_label': 'unknown',
                    'predicted_evidence': f"Error: {result['error']}",
                    'predicted_source': [],
                    'predicted_confidence': 0.0,
                    'rag_A_label': 'unknown',
                    'rag_A_evidence': 'Error occurred',
                    'rag_B_label': 'unknown',
                    'rag_B_evidence': 'Error occurred',
                    'fact_check_label': 'unknown',
                    'fact_check_evidence': 'Error occurred'
                })
                continue

            results.append({
                'claim': entry['claim'],
                'true_label': entry.get('label', 'unknown'),
                'true_evidence': entry.get('evidence', 'No evidence provided'),

                'predicted_label': result.get('final_verdict', {}).get('label', 'unknown'),
                'predicted_evidence': result.get('final_verdict', {}).get('evidence', 'No evidence provided'),
                'predicted_source': result.get('final_verdict', {}).get('source', []),
                'predicted_confidence': result.get('final_verdict', {}).get('confidence', 0.0),

                'rag_A_label': result.get('rag_A', {}).get('label', 'unknown'),
                'rag_A_evidence': result.get('rag_A', {}).get('evidence', 'No evidence provided'),
                'rag_B_label': result.get('rag_B', {}).get('label', 'unknown'),
                'rag_B_evidence': result.get('rag_B', {}).get('evidence', 'No evidence provided'),
                'fact_check_label': result.get('fact_check', {}).get('label', 'unknown'),
                'fact_check_evidence': result.get('fact_check', {}).get('evidence', 'No evidence provided')
            })

        return results