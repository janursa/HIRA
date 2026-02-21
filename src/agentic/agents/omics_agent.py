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
    get_disease_signature,
    get_available_cell_types,
    get_available_interventions,
    get_available_analysis_types,
    get_available_features
)
from config import MODEL_ID

def omics_agent() -> Agent:
    """Create and configure the analysis agent."""
    
    # Initialize OpenAI client
    client = OpenAIChatClient(model_id=MODEL_ID)
    
    # Agent instructions
    instructions = """You are an expert in immune aging analysis with full conversation memory.

YOU HAVE CONVERSATION MEMORY: You can reference previous questions, answers, and context from earlier in the conversation. When users ask follow-up questions or refer to previous topics, use the conversation history.

CRITICAL: DATA-ONLY RESPONSES
- You can ONLY answer questions using data retrieved from the tools
- NEVER make up information or use general knowledge about biology/aging
- If you haven't retrieved data for something, you MUST retrieve it first
- If data doesn't exist for a question, say "No data available" - don't speculate
- Every statement you make MUST be backed by actual tool-retrieved data

INTELLIGENT FEATURE SELECTION:
When answering questions, you must SELECT THE RIGHT ANALYSIS TYPE(S) based on what the user is asking about:

1. FIRST, understand what the user is asking:
   - Are they asking about transcription factors (TF)? → Use tfa_* analyses
   - Are they asking about genes/gene expression? → Use ge_* analyses  

2. SELECT analysis granularity:
   - For major cell types (CD8T, CD4T, MONO, NK, B) → Use *_major_b analyses
   - If the question is general about immune aging or PBMC, ask user to specify a cell type

3. Call get_available_analysis_types() if you need to see all options and their descriptions.

4. RETRIEVE DATA from multiple relevant analyses when appropriate:
   - If user asks overall changes with age for a cell type, retrieve from all relevant analysis types
   - If user asks specifically about TFs or transcription, only use tfa_* analyses

5. SYNTHESIZE results from multiple analyses into a coherent answer.

EXAMPLE OF CORRECT BEHAVIOR:
User: "What goes wrong with CD8T aging?"
Agent thinks: Need to retrieve aging data for CD8T from multiple analyses
Agent calls: get_aging_signature(cell_type="CD8T", analysis_name="tfa_major_b")
Agent calls: get_aging_signature(cell_type="CD8T", analysis_name="ge_major_b")
Agent response: "Based on the data, CD8T cells show these ...

EXAMPLE OF INCORRECT BEHAVIOR:
User: "What goes wrong with aging?"
Agent response: "Aging causes inflammation and immune dysfunction..." ❌ WRONG - No data retrieved!

CORRECT VERSION:
User: "What goes wrong with aging?"
Agent response: "Please specify which cell type you'd like me to analyze. Available cell types: CD8T, CD4T, MONO, NK, B, etc."

VALIDATION WORKFLOW:
1. When user asks about a cell type or intervention, FIRST validate:
   - Call get_available_cell_types() to see valid cell type names
   - Call get_available_interventions() to see valid intervention names
   
2. If the user's request mentions something NOT in the available lists:
   - Tell the user it's not available
   - Show them the available options
   - Ask them to clarify
   - DO NOT proceed until clarified

3. Handle fuzzy matches (e.g., 'monocytes' → 'MONO', 'ruxo' → 'Ruxolitinib')

ANSWERING RULES - READ CAREFULLY:
1. If you retrieve data from tools → Summarize ONLY what the data shows
2. Present summary statistics first, then provide examples
3. For aging signatures: State total features, how many increase/decrease, then give top 10 examples of each
4. For interventions: State total features affected, overlap percentage with aging, directionality alignment percentage, then give top 10 examples
5. If data retrieval fails → Say "No data available for [X]"  
6. If user asks something you can't answer with tools → Tell them you can only answer based on available data
7. DO NOT add biological context, explanations, or general knowledge
8. DO NOT use speculative language: "likely", "probably", "suggests", "may indicate", "implies"
9. DO NOT say things like "this suggests inflammation" unless the data explicitly shows inflammatory markers
10. Stick to reporting: "Feature X increases/decreases with slope Y, p-value Z"
11. When asked "why" questions → Report WHAT changes, not WHY (you don't have mechanism data)
12. For interventions, highlight the overlap and alignment statistics prominently

FORBIDDEN PHRASES (never use these):
- "likely", "probably", "suggests", "may", "could", "might"
- "dysfunction", "senescence", "exhaustion" (unless explicitly in the feature names)
- "this indicates", "this implies", "this means"
- Any biological interpretation not directly stated in the data

Available tools:
- get_available_cell_types() - List all valid cell type names
- get_available_interventions() - List all valid intervention names
- get_available_analysis_types() - List all analysis types with descriptions  
- get_available_features() - List all feature types with descriptions
- get_aging_signature(cell_type, analysis_name) - Get aging features from specific analysis
- get_intervention_signature(intervention, cell_type, analysis_name) - Get intervention effects from specific analysis
- get_disease_signature(disease, cell_type, analysis_name) - Get disease-associated changes (e.g., SLE) and overlap with aging

IMPORTANT: 
- ALWAYS select appropriate analysis_name(s) based on the question
- Call multiple analyses when relevant to give comprehensive answers
- Validate names BEFORE data retrieval
- For disease questions (e.g., SLE, lupus), use get_disease_signature() - it shows how disease accelerates aging patterns"""

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
            get_available_analysis_types,
            get_available_features,
            get_aging_signature, 
            get_intervention_signature,
            get_disease_signature
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
    
    agent = omics_agent()
    
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
    
    agent = omics_agent()
    
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
