"""
Agentic analysis pipeline for immune aging intervention discovery.
Simplified version: agent calls tools to retrieve data, LLM interprets results.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from helper import get_aging_signature, get_intervention_signature
from config import MODEL_ID

def create_agent() -> Agent:
    """Create and configure the analysis agent."""
    
    # Initialize OpenAI client
    client = OpenAIChatClient(model_id=MODEL_ID)
    
    # Agent instructions
    instructions = """You are an expert in immune aging analysis.

CRITICAL: You MUST use the available tools to retrieve data before answering questions.

Available tools:
1. get_aging_signature(cell_type, analysis_name='tfa_major_b') - Get aging features
2. get_intervention_signature(intervention, cell_type, analysis_name='tfa_major_b') - Get intervention effects

Example workflow for "what gene signature is reverted by Ruxolitinib in CD8T?":
1. ALWAYS start by calling get_aging_signature(cell_type=cell_type, analysis_name='tfa_major_b')
2. Then call get_intervention_signature(intervention='ruxolitinib', cell_type=cell_type, analysis_name='tfa_major_b')
3. Compare the two signatures to identify reversal patterns
4. Explain findings in clear, natural language

Cell types: CD8T, CD4T, NK, MONO, B
Analysis types: tfa_major_b

Don't guess or make assumptions - use the tools to get real data."""

    # Create agent using as_agent method
    agent = client.as_agent(
        name="AgingAnalysisAgent",
        instructions=instructions,
        tools=[get_aging_signature, get_intervention_signature]
    )
    
    return agent


def run_analysis(prompt: str = None):
    """Run the agent analysis."""
    import asyncio
    
    if prompt is None:
        prompt = "what gene signature is reverted by ruxolitinib in CD8T?"
    
    print(f"\n{'='*80}")
    print(f"QUERY: {prompt}")
    print(f"{'='*80}\n")
    
    agent = create_agent()
    
    # Run async
    async def _run():
        print("AGENT RESPONSE:\n")
        async for chunk in agent.run(prompt, stream=True):
            if chunk.text:
                print(chunk.text, end='', flush=True)
    
    asyncio.run(_run())
    
    print(f"\n\n{'='*80}\n")


if __name__ == "__main__":
    run_analysis()
