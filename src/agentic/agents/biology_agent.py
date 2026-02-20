"""
Biology Interpreter Agent
Takes factual data findings and provides biological context and interpretation.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework import InMemoryHistoryProvider
from config import MODEL_ID


def biology_agent() -> Agent:
    """Create the biology interpretation agent."""
    
    client = OpenAIChatClient(model_id=MODEL_ID)
    
    instructions = """You are an expert biologist specializing in immunology.

YOUR ROLE:
You receive FACTUAL DATA from immune aging experiments (gene/TF expression changes, immune disease, and intervention effects) 
and provide BIOLOGICAL INTERPRETATION and CONTEXT.

CRITICAL RULES:
1. You will be provided with DATA-ONLY findings (lists of genes, TFs, slopes, p-values)
2. Your job is to INTERPRET what these changes mean biologically
3. You can and should use your biological knowledge to:
   - Explain what these genes/TFs do
   - Describe relevant pathways and mechanisms  
   - Connect findings to known aging biology

4. ALWAYS distinguish between:
   - FACT (from the data): "BATF increases with age (slope=0.51, p<0.001)"
   - INTERPRETATION (your expertise): "BATF is a key regulator of T cell exhaustion and its increase suggests age-related T cell dysfunction"

5. Structure your responses as:
   **Data Summary:** [restate the key findings]
   **Biological Context:** [explain what these genes/pathways do]
   **Interpretation:** [what this means for aging biology]
   **Known Biology:** [relevant published findings if you know them]
   **Implications:** [potential functional consequences]

  
REMEMBER:
- Be specific and technical - this is for scientific audiences
- Acknowledge uncertainty when appropriate ("may contribute to", "potentially involved in")
- Don't invent citations, but DO use your knowledge of the field

"""

    history_provider = InMemoryHistoryProvider(source_id="in_memory")
    
    agent = client.as_agent(
        name="BiologyInterpreter",
        instructions=instructions,
        context_providers=[history_provider]
    )
    
    return agent


async def interpret_findings(findings: str, biology_agent: Agent = None, session = None) -> str:
    """
    Take factual findings and get biological interpretation.
    
    Args:
        findings: Factual data findings from the data agent
        biology_agent: Optional pre-created biology interpreter agent
        session: Optional session for conversation memory
        
    Returns:
        Biological interpretation
    """
    if biology_agent is None:
        biology_agent = create_biology_interpreter()
    
    if session is None:
        session = biology_agent.create_session()
    
    result = await biology_agent.run(findings, session=session)
    return result


# For testing
if __name__ == "__main__":
    import asyncio
    
    # Test with example data
    test_findings = """
    CD8T aging analysis shows:
    
    Transcription Factors Increasing with Age:
    - BATF (slope=0.5068, p_adj=0.0000)
    - CEBPB (slope=0.5688, p_adj=0.0000)
    - CEBPG (slope=0.5724, p_adj=0.0000)
    - ATF2 (slope=0.5334, p_adj=0.0000)
    
    Transcription Factors Decreasing with Age:
    - DNMT1 (slope=-0.5414, p_adj=0.0000)
    - BCL11B (slope=-0.5375, p_adj=0.0000)
    - BACH1 (slope=-0.4920, p_adj=0.0000)
    - BACH2 (slope=-0.5290, p_adj=0.0000)
    """
    
    async def test():
        agent = create_biology_interpreter()
        session = agent.create_session()
        
        print("="*80)
        print("DATA FINDINGS (from data agent):")
        print("="*80)
        print(test_findings)
        print()
        
        print("="*80)
        print("BIOLOGICAL INTERPRETATION:")
        print("="*80)
        result = await agent.run(test_findings, session=session)
        print(result)
    
    asyncio.run(test())
