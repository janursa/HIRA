"""
Two-Agent Pipeline: Data Retrieval → Biology Interpretation
Combines the data-grounded agent with the biology interpreter agent.
"""

import asyncio
from agents.omics_agent import omics_agent
from agents.biology_agent import biology_agent


class MainPipeline:
    """Pipeline that chains data agent → biology interpreter."""
    
    def __init__(self):
        """Initialize both agents."""
        self.data_agent = omics_agent()
        self.biology_agent = biology_agent()
        self.data_session = self.data_agent.create_session()
        self.biology_session = self.biology_agent.create_session()
    
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
            print("🧬 BIOLOGY INTERPRETER (adding context)...")
            print("-" * 80)
            
            # Create prompt for biology agent
            bio_prompt = f"""The following data was found for the question: "{question}"

DATA FINDINGS:
{data_findings}

Please provide biological interpretation of these findings."""
            
            interpretation = await self.biology_agent.run(bio_prompt, session=self.biology_session)
            print(interpretation)
            result['interpretation'] = str(interpretation)
        
        return result
    
    async def interactive_mode(self):
        """Run in interactive mode with both agents."""
        print("\n" + "="*80)
        print("Main PIPELINE - Interactive Mode")
        print("="*80)
        print("\nThis pipeline combines:")
        print("  1. DATA AGENT: Retrieves factual findings from your data")
        print("  2. BIOLOGY INTERPRETER: Adds biological context and interpretation")
        print("\nOptions:")
        print("  - Type your question and press Enter")
        print("  - Type 'data-only' to skip interpretation for next question")
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
            
            if user_input.lower() == 'data-only':
                print("[Next question will be data-only]")
                try:
                    user_input = input("\nYou: ").strip()
                    if not user_input:
                        continue
                    await self.ask(user_input, include_interpretation=False)
                except Exception as e:
                    print(f"\n[Error: {e}]")
                continue
            
            try:
                await self.ask(user_input, include_interpretation=True)
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
