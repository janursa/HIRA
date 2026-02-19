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
from agent_framework import InMemoryHistoryProvider
from helper import (
    get_aging_signature, 
    get_intervention_signature,
    get_available_cell_types,
    get_available_interventions
)
from config import MODEL_ID

def create_agent() -> Agent:
    """Create and configure the analysis agent."""
    
    # Initialize OpenAI client
    client = OpenAIChatClient(model_id=MODEL_ID)
    
    # Agent instructions
    instructions = """You are an expert in immune aging analysis with full conversation memory.

YOU HAVE CONVERSATION MEMORY: You can reference previous questions, answers, and context from earlier in the conversation. When users ask follow-up questions or refer to previous topics, use the conversation history.

CRITICAL VALIDATION WORKFLOW:
1. When user asks about a cell type or intervention, FIRST check if it's valid:
   - Call get_available_cell_types() to see valid cell type names
   - Call get_available_interventions() to see valid intervention names
   
2. If the user's request mentions a cell type or intervention that is NOT in the available lists:
   - Tell the user the exact name is not available
   - Show them the available options
   - Ask them to clarify which one they meant
   - DO NOT proceed with data retrieval until clarified

3. If names are close matches (e.g., 'monocytes' vs 'MONO', 'ruxo' vs 'Ruxolitinib'):
   - Use the correct name from the available lists
   - Proceed with data retrieval

4. For data retrieval:
   - Call get_aging_signature(cell_type, analysis_name='tfa_major_b')
   - Call get_intervention_signature(intervention, cell_type, analysis_name='tfa_major_b')
   - Compare and explain in natural language

Available tools:
- get_available_cell_types() - List all valid cell type names
- get_available_interventions() - List all valid intervention names  
- get_aging_signature(cell_type, analysis_name) - Get aging features
- get_intervention_signature(intervention, cell_type, analysis_name) - Get intervention effects

IMPORTANT: Always validate names BEFORE attempting data retrieval."""

    # EXPLICITLY create InMemoryHistoryProvider to ensure conversation memory works
    # Note: source_id is optional, defaults to "in_memory" if not provided
    history_provider = InMemoryHistoryProvider(source_id="in_memory")
    
    # Create agent using as_agent method
    agent = client.as_agent(
        name="AgingAnalysisAgent",
        instructions=instructions,
        tools=[
            get_available_cell_types,
            get_available_interventions,
            get_aging_signature, 
            get_intervention_signature
        ],
        context_providers=[history_provider]  # EXPLICIT provider for memory
    )
    
    return agent


def run_analysis(prompt: str = None):
    """Run the agent analysis (single query mode)."""
    import asyncio
    
    if prompt is None:
        prompt = "what gene signature is reverted by ruxolitinib in CD4 T?"
    
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


def run_interactive():
    """Run the agent in interactive mode with conversation memory."""
    import asyncio
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Immune Aging Analysis Agent")
    print("="*80)
    print("\nThe agent maintains conversation history within this session.")
    print("You can ask follow-up questions and reference previous queries.")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'clear' to reset conversation history")
    print("="*80 + "\n")
    
    agent = create_agent()
    
    async def chat_loop():
        # Create a session to maintain conversation history
        session = agent.create_session()
        
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                print("\n[Conversation history cleared]")
                session = agent.create_session()
                continue
            
            # Get agent response WITHOUT streaming (streaming breaks memory in agent-framework)
            print("\nAgent: ", end='', flush=True)
            try:
                result = await agent.run(user_input, session=session)
                print(result)
            except Exception as e:
                print(f"\n[Error: {e}]")
    
    asyncio.run(chat_loop())


if __name__ == "__main__":
    import sys
    run_interactive()
    # # Check if interactive mode requested
    # if len(sys.argv) > 1 and sys.argv[1] in ['-i', '--interactive']:
    #     run_interactive()
    # else:
    #     run_analysis()
