"""
Two-Agent Pipeline: Data Retrieval → Context Interpretation
Combines the data-grounded agent with the context agent.
"""

import asyncio
from agents.omics_agent import omics_agent
from agents.context_agent import context_agent


class MainPipeline:
    """Pipeline that chains data agent → context interpreter."""
    
    def __init__(self):
        """Initialize both agents."""
        self.data_agent = omics_agent()
        self.context_agent = context_agent()
        self.data_session = self.data_agent.create_session()
        self.context_session = self.context_agent.create_session()
    
    async def ask(self, question: str, include_interpretation: bool = True) -> dict:
        """
        Ask a question and get both data findings and biological interpretation.
        
        Args:
            question: User's question
            include_interpretation: Whether to include biology interpretation
            
        Returns:
            dict with 'data_findings' and optionally 'interpretation'
        """
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}\n")
        
        # Step 1: Get factual data
        print("🔬 DATA AGENT (retrieving factual findings)...")
        print("-" * 80)
        data_findings = await self.data_agent.run(question, session=self.data_session)
        print(data_findings)
        print()
        
        result = {
            'question': question,
            'data_findings': str(data_findings)
        }
        
        if include_interpretation:
            # Step 2: Get biological interpretation
            print("🧬 CONTEXT AGENT (adding biological context)...")
            print("-" * 80)
            
            # Create prompt for context agent
            context_prompt = f"""The following data was found for the question: "{question}"

DATA FINDINGS:
{data_findings}

Please provide biological interpretation of these findings."""
            
            interpretation = await self.context_agent.run(context_prompt, session=self.context_session)
            print(interpretation)
            result['interpretation'] = str(interpretation)
        
        return result
    
    async def interactive_mode(self):
        """Run in interactive mode with both agents."""
        print("\n" + "="*80)
        print("Main PIPELINE - Interactive Mode")
        print("="*80)
        print("\nThis pipeline combines:")
        print("  1. DATA AGENT: Retrieves factual findings from omics analysis")
        print("  2. CONTEXT AGENT: Adds biological context and interpretation using manuscript knowledge")
        print("\nWorkflow:")
        print("  - Ask your question")
        print("  - Get factual data results")
        print("  - Choose whether to get interpretation/context")
        print("\nCommands:")
        print("  - Type 'quit' or 'exit' to end session")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            try:
                # Step 1: Always show data first
                print(f"\n{'='*80}")
                print(f"QUESTION: {user_input}")
                print(f"{'='*80}\n")
                
                print("🔬 DATA AGENT (retrieving factual findings)...")
                print("-" * 80)
                data_findings = await self.data_agent.run(user_input, session=self.data_session)
                print(data_findings)
                print()
                
                # Step 2: Ask if user wants interpretation
                try:
                    want_context = input("Would you like biological interpretation/context? (y/n): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n\nExiting...")
                    break
                
                if want_context in ['y', 'yes']:
                    print("\n🧬 CONTEXT AGENT (adding biological context)...")
                    print("-" * 80)
                    
                    context_prompt = f"""The following data was found for the question: "{user_input}"

DATA FINDINGS:
{data_findings}

Please provide biological interpretation of these findings."""
                    
                    interpretation = await self.context_agent.run(context_prompt, session=self.context_session)
                    print(interpretation)
                    print()
                
            except Exception as e:
                print(f"\n[Error: {e}]")


async def main():
    """Example usage of the main pipeline."""
    pipeline = MainPipeline()
    
    # Example questions
    questions = [
        "What transcription factors change in CD8T cells with aging?",
        "How does Ruxolitinib affect MONO cells?",
    ]
    
    for q in questions:
        await pipeline.ask(q)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-i', '--interactive']:
        # Interactive mode
        asyncio.run(MainPipeline().interactive_mode())
    else:
        # Demo mode
        print("Running demo. Use --interactive for interactive mode.")
        asyncio.run(main())
