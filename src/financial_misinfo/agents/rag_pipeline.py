
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
from financial_misinfo.core.base import LLMResponseParser
from financial_misinfo.agents.data_handler import DataHandlerAgent

class RAGPipeline (LLMResponseParser):
    def __init__(self, model: str, hf_token: str, data_handler: DataHandlerAgent):

        if not hf_token or hf_token.strip() == "":
            raise ValueError("HuggingFace token is missing or empty")

        self.model_name = model
        self.hf_token = hf_token
        self.data_handler = data_handler
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {hf_token}"}

    async def process(self, query: str, k: int = 5) -> Dict:
        """Main processing pipeline for claims"""
        try:
            # Get context from data handler
            context = await self.retrieve_context(query, k)
            
            # Generate and return response
            response = await self.generate_response(query, context)
            return {
                'label': response.get('label', 'nei'),
                'evidence': response.get('evidence', 'No evidence provided'),
                'confidence': response.get('confidence', 0.0),
                'source': response.get('source', [])
            }
        except Exception as e:
            print(f"Error in RAGPipeline process: {str(e)}")
            return self._handle_error(e)

    async def retrieve_context(self, query: str, k: int = 5) -> Dict:
        """Pass through to DataHandlerAgent's retrieve_context"""
        try:
            return await self.data_handler.retrieve_context(query, k)
        except Exception as e:
            print(f"Error in retrieve_context: {str(e)}")
            return {
                'type': 'error',
                'error': str(e)
            }


    async def generate_response(self, query: str, context: Dict) -> Dict:
        """Generate response based on context and confidence levels using a multi-tiered approach"""
        try:
            similarity_score = context.get('similarity', 0.0)

            """
            # Tier 1: High confidence (>0.6) - Direct use of retrieved info
            if similarity_score > 0.6 and context['type'] == 'high_confidence':
                metadata = context['metadata']
                return {
                    'label': metadata['label'],
                    'evidence': metadata['evidence_sentence'],
                    'source': metadata['hrefs'],
                    'confidence': similarity_score
                }

            """
            
            
            # In RAGPipeline.generate_response, modify the high confidence case:
            if similarity_score > 0.6 and context['type'] == 'high_confidence':
                metadata = context['metadata']

                # Combine the specific evidence sentence with broader context from justification
                evidence_text = metadata['evidence_sentence']

                # Access the justification from the metadata
                if 'metadata' in metadata and 'justification' in metadata['metadata']:
                    justification = metadata['metadata']['justification']
                    if justification:
                        # Add context while avoiding duplication if the evidence is already in the justification
                        if evidence_text not in justification:
                            evidence_text = f"{evidence_text} Context: {justification[:500]}..."  # Limit justification length

                return {
                    'label': metadata['label'],
                    'evidence': evidence_text,
                    'source': metadata['hrefs'],
                    'confidence': similarity_score
                }
            
            
            # Tier 2: Medium confidence (0.4-0.7) - Hybrid approach
            elif similarity_score > 0.4 and context['type'] == 'high_confidence':
                metadata = context['metadata']

                # Use your original prompt but include retrieved evidence as context
                prompt = f"""<s>[INST]
    You are a collaborative team of expert fact-checkers consisting of:

    1. Financial Analysis & Economic Policy Expert:
       - Specialized in analyzing economic claims, financial markets, and monetary policy
       - Deep knowledge of corporate finance and market trends

    2. Political Misinformation & Election Integrity Specialist:
       - Expert in detecting political misinformation and disinformation
       - Specialized knowledge of election systems and voting processes
       - Deep understanding of political campaigns and messaging

    3. Government Policy Analyst:
       - Expertise in legislative processes and policy implementation
       - Deep knowledge of federal, state, and local government operations
       - Specialized in analyzing policy claims and their impacts

    4. Investigative Journalist:
       - Extensive experience in fact-checking and verification
       - Expertise in source evaluation and contextual analysis
       - Skilled in uncovering deceptive claims and narratives

    Carefully analyze the following claim for accuracy and verifiability based on the relevant context. 
    Please note that we have retrieved some evidence that is MODERATELY RELEVANT (similarity score: {similarity_score}).
    If you feel that the below retrieved information is not very related to the query, please ignore this information and 
    simply query the claim.
    
    Retrieved Evidence: {metadata['evidence_sentence']}
    Retrieved Label: {metadata['label']}
    Source: {metadata['hrefs']}

    Claim: {query}

    Analysis Framework:

    1. Core Claims Identification:
       - Break down and identify the main factual assertion(s)
       - Isolate verifiable components
       - Note any important context or qualifiers

    2. Step-by-Step Analysis:
       - Systematically evaluate each component of the claim
       - Cross-reference with known facts and reliable data
       - Consider historical and contextual factors
       - Apply relevant expertise from each team member

    3. Evidence Assessment:
       - Evaluate the reliability of available information
       - Identify any gaps in the evidence
       - Consider potential biases or misleading elements

    4. Final Determination:
       - Based on your collective analysis, label the claim as one of:
           * True: The core factual assertions of the claim are accurate.
           * False: The core factual assertions are demonstrably untrue or unsupported by credible evidence.
           * NEI (Not Enough Information): There is insufficient reliable data for a confident verification.

    Additionally, please provide:
    - Evidence: A detailed, evidence-based justification for your label, highlighting key facts, data, and reasoning from the relevant expert perspectives. If the evidence is not related to the claim, please output "LLM Knowledge"
    - Source: [Specify the most reliable source(s) that support your conclusion. If using LLM knowledge, clearly note "LLM Knowledge" and specify time relevance of the knowledge]
    - Confidence: [0.0-1.0 score, where 0.0 means complete uncertainty and 1.0 means absolute certainty]

    Return your response exactly in the following format (and no additional commentary):

    Provide your response in this exact format:
    Label: [true/false/nei]
    Evidence: [Your detailed reasoning]
    Source: [Your source of information add to it]
    Confidence: [0.0-1.0]
    [/INST]</s>"""

                response = await self._make_llm_request(prompt)
                parsed_response = self._parse_llm_response(response)

                # Process the sources
                source = []
                if parsed_response.get('source'):
                    if isinstance(parsed_response['source'], list):
                        source = parsed_response['source']
                    else:
                        source = [parsed_response['source']]

                # If LLM opted to use the retrieved sources, use those
                if "LLM Knowledge" not in source and "LLM" not in source:
                    source = metadata['hrefs']

                # Blend confidence scores
                blended_confidence = (similarity_score + float(parsed_response['confidence'])) / 2

                return {
                    'label': parsed_response['label'],
                    'evidence': parsed_response['evidence'],
                    'source': source,
                    'confidence': blended_confidence
                }

            # Tier 3: Low confidence (<0.4) - Pure LLM approach with explicit instructions to ignore irrelevant context
            else:
                # Use your original prompt but explicitly instruct to ignore irrelevant context
                prompt = f"""<s>[INST]
    You are a collaborative team of expert fact-checkers consisting of:

    1. Financial Analysis & Economic Policy Expert:
       - Specialized in analyzing economic claims, financial markets, and monetary policy
       - Deep knowledge of corporate finance and market trends

    2. Political Misinformation & Election Integrity Specialist:
       - Expert in detecting political misinformation and disinformation
       - Specialized knowledge of election systems and voting processes
       - Deep understanding of political campaigns and messaging

    3. Government Policy Analyst:
       - Expertise in legislative processes and policy implementation
       - Deep knowledge of federal, state, and local government operations
       - Specialized in analyzing policy claims and their impacts

    4. Investigative Journalist:
       - Extensive experience in fact-checking and verification
       - Expertise in source evaluation and contextual analysis
       - Skilled in uncovering deceptive claims and narratives

    
    Claim: {query}

    Analysis Framework:

    1. Core Claims Identification:
       - Break down and identify the main factual assertion(s)
       - Isolate verifiable components
       - Note any important context or qualifiers

    2. Step-by-Step Analysis:
       - Systematically evaluate each component of the claim
       - Cross-reference with known facts and reliable data
       - Consider historical and contextual factors
       - Apply relevant expertise from each team member

    3. Evidence Assessment:
       - Evaluate the reliability of available information
       - Identify any gaps in the evidence
       - Consider potential biases or misleading elements

    4. Final Determination:
       - Based on your collective analysis, label the claim as one of:
           * True: The core factual assertions of the claim are accurate.
           * False: The core factual assertions are demonstrably untrue or unsupported by credible evidence.
           * NEI (Not Enough Information): There is insufficient reliable data for a confident verification.

    Additionally, please provide:
    - Evidence: A detailed, evidence-based justification for your label, highlighting key facts, data, and reasoning from the relevant expert perspectives. You should use "LLM Knowledge" since the context is irrelevant
    - Source: [Since you're relying on your own knowledge, specify "LLM Knowledge"]
    - Confidence: [0.0-1.0 score, where 0.0 means complete uncertainty and 1.0 means absolute certainty]

    Return your response exactly in the following format (and no additional commentary):

    Provide your response in this exact format:
    Label: [true/false/nei]
    Evidence: [Your detailed reasoning]
    Source: [LLM Knowledge]
    Confidence: [0.0-1.0]
    [/INST]</s>"""

                response = await self._make_llm_request(prompt)
                parsed_response = self._parse_llm_response(response)

                # Process the source properly
                source = []
                if parsed_response.get('source'):
                    if isinstance(parsed_response['source'], list):
                        source = parsed_response['source']
                    else:
                        source = [parsed_response['source']]

                # Only fallback to generic source if none provided
                if not source:
                    source = ['LLM Knowledge']

                return {
                    'label': parsed_response['label'],
                    'evidence': parsed_response['evidence'],
                    'source': source,
                    'confidence': float(parsed_response['confidence'])
                }
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return self._create_error_response(str(e))



            
    async def _make_llm_request(self, prompt: str) -> str:
        """Make request to HuggingFace API with improved validation and error handling."""
        try:
            # Validate token again (in case it was changed after initialization)
            if not self.hf_token or self.hf_token.strip() == "":
                raise ValueError("HuggingFace token is missing or empty")
            
            # Debug statement to show token (partially masked for security)
            token_preview = self.hf_token[:4] + "..." + self.hf_token[-4:] if len(self.hf_token) > 8 else "[empty]"
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.1
                    }
                },
                timeout=30  # Add timeout to prevent hanging on slow responses
            )
            
            # Check for various error conditions
            if response.status_code == 401:
                raise ValueError(f"API authentication failed: Invalid HuggingFace token")
            elif response.status_code == 404:
                raise ValueError(f"Model not found: {self.model_name}")
            elif response.status_code == 503:
                raise ValueError(f"API server overloaded or unavailable")
            elif response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if not result or not isinstance(result, list) or not result[0]:
                raise Exception("Invalid API response format")
            
            return result[0].get('generated_text', '')

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error when calling HuggingFace API: {str(e)}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response from API: {response.text[:100]}...")
        except Exception as e:
            print(f"Error in LLM request: {str(e)}")
            raise    

    def _handle_error(self, error: Exception) -> Dict:
        """Create error response"""
        return {
            'pipeline': self.model_name,
            'label': 'nei',
            'evidence': f"Error: {str(error)}",
            'confidence': 0.0,
            'source': []
        }

    def _create_error_response(self, error_msg: str) -> Dict:
        """Create structured error response"""
        return {
            'label': 'nei',
            'evidence': f"Error: {error_msg}",
            'confidence': 0.0,
            'source': []
        }
        