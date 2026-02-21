"""
Context Agent: Provides biological context and interpretation based on manuscript knowledge.
Has access to the manuscript summarizing findings on aging and interventions.
"""

import os
from pathlib import Path
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework import InMemoryHistoryProvider
from config import MODEL_ID


def load_manuscript_content():
    """Load the manuscript content for context."""
    data_dir = Path(__file__).parent.parent / "data"
    manuscript_path = data_dir / "manuscrtipt.txt"
    
    if not manuscript_path.exists():
        return "Manuscript not found."
    
    with open(manuscript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content


def context_agent() -> Agent:
    """Create and configure the context agent with manuscript knowledge."""
    
    # Load manuscript content
    manuscript = load_manuscript_content()
    
    # Initialize OpenAI client
    client = OpenAIChatClient(model_id=MODEL_ID)
    
    # Agent instructions
    instructions = f"""You are an expert biology interpreter specializing in immunology, aging, and transcriptional regulation.

YOUR KNOWLEDGE BASE:
You have access to a comprehensive manuscript about immune aging analysis. Use this as your PRIMARY source of biological context.

MANUSCRIPT CONTENT:
{manuscript}

YOUR ROLE:
You receive DATA-ONLY findings from the omics agent (raw statistics about transcription factors, genes, etc.) and provide biological interpretation by:

- Provide complementary context for the received data from the manuscript.
- Adding more biological insights from your expertise and the manuscript content.
- DISTINGUISHING between:
    - FACT: What the data shows (from omics agent)
    - MANUSCRIPT CONTEXT: What has been reported in the manuscript
    - INTERPRETATION: What it means biologically (your expertise)

OUTPUT STRUCTURE:
Always structure your responses as:

**Data Summary:**
[Briefly restate the key factual findings from the omics agent]

**Manuscript Context:**
[Connect to context from the manuscript]

**Biological Context:**
[Explain what these findings mean biologically, referencing relevant pathways, mechanisms, and biological processes]

**Known Biology:**
[Reference established biological knowledge about these findings from literature (both listed in the manuscript and from your expertise)]


GUIDELINES:
1. Be clear when you're interpreting vs stating facts
2. Explain mechanisms when known
3. Acknowledge uncertainty when appropriate
4. Focus on biological insight, not speculation
"""

    # Create InMemoryHistoryProvider for conversation memory
    history_provider = InMemoryHistoryProvider(source_id="in_memory")
    
    # Create agent
    agent = client.as_agent(
        name="ContextAgent",
        instructions=instructions,
        tools=[],  # No tools needed - uses manuscript knowledge
        context_providers=[history_provider]
    )
    
    return agent


async def interpret_findings(findings: str, agent: Agent, session=None):
    """
    Helper function to interpret findings using the context agent.
    
    Args:
        findings: The data findings to interpret
        agent: The context agent
        session: Optional session for conversation memory
        
    Returns:
        Interpretation response
    """
    if session is None:
        session = agent.create_session()
    
    prompt = f"""Please provide biological interpretation of the following data findings:

{findings}

Provide context from the manuscript where relevant and explain the biological significance."""
    
    response = await agent.run(prompt, session=session)
    return response


# Test code
async def test_context_agent():
    """Test the context agent."""
    agent = context_agent()
    session = agent.create_session()
    
    # Test prompt
    test_findings = """
AGING SIGNATURE FOR CD8T - tfa_major_b

SUMMARY STATISTICS:
  • Total significant features: 368
  • Features increasing with age: 158
  • Features decreasing with age: 210

TOP 10 EXAMPLES - DECREASING WITH AGE:
  AEBP1, AHR, AKAP8L, ANKZF1, ASH1L, ATF4, ATF6, BACH1, BACH2, BCL11B

TOP 10 EXAMPLES - INCREASING WITH AGE:
  AEBP2, AKNA, ARHGAP35, ARID5B, ASCL2, ATF2, ATF6B, BATF, BAZ2A, BAZ2B
"""
    
    print("Testing Context Agent\n" + "="*80)
    response = await interpret_findings(test_findings, agent, session)
    print(response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_context_agent())
