# Immune Aging Analysis Agent - Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                                  │
│                                                                           │
│  User asks: "What is the aging story of CD8T?"                          │
│  User asks: "How does Ruxolitinib affect that?"  (references CD8T)      │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    AGENT (GPT-4o via OpenAI)                             │
│                                                                           │
│  Instructions:                                                            │
│  • Expert in immune aging analysis                                       │
│  • Full conversation memory                                              │
│  • MUST validate cell types/interventions before retrieval               │
│  • Uses 4 tools to gather data                                          │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MEMORY                                    │
│                   (InMemoryHistoryProvider)                              │
│                                                                           │
│  • Stores all messages in session.state["in_memory"]["messages"]        │
│  • Loaded BEFORE each agent.run() via before_run hook                   │
│  • Saved AFTER each agent.run() via after_run hook                      │
│  • Maintains context across conversation                                 │
│                                                                           │
│  Note: Streaming mode DISABLED (breaks after_run hook)                  │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         TOOL SELECTION                                    │
│                                                                           │
│  Agent decides which tool(s) to call based on user question             │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌─────────────────────┐   ┌─────────────────────────┐
        │  VALIDATION TOOLS   │   │   DATA RETRIEVAL TOOLS  │
        └─────────────────────┘   └─────────────────────────┘
                    │                         │
        ┌───────────┴──────────┐  ┌──────────┴───────────┐
        ▼                      ▼  ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ get_available_   │  │ get_available_   │  │ get_aging_       │  │ get_intervention_│
│ cell_types()     │  │ interventions()  │  │ signature()      │  │ signature()      │
│                  │  │                  │  │                  │  │                  │
│ Returns list of  │  │ Returns list of  │  │ Args:            │  │ Args:            │
│ valid cell types │  │ valid            │  │ • cell_type      │  │ • intervention   │
│ (Major & Sub)    │  │ interventions    │  │ • analysis_name  │  │ • cell_type      │
│                  │  │                  │  │                  │  │ • analysis_name  │
│ Example:         │  │ Example:         │  │ Returns:         │  │                  │
│ • CD4T           │  │ • Ruxolitinib    │  │ Dict with:       │  │ Returns:         │
│ • CD8T           │  │ • Metformin      │  │ • increasing_    │  │ Dict with:       │
│ • MONO           │  │ • Rapamycin      │  │   features       │  │ • significant_   │
│ • NK             │  │ • etc.           │  │ • decreasing_    │  │   features       │
│ • B              │  │                  │  │   features       │  │ • n_sig_features │
└────────┬─────────┘  └────────┬─────────┘  │ • n_sig_features │  │                  │
         │                     │             └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │                     │
         └─────────────────────┴──────────────────────┴─────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION                                         │
│                                                                           │
│  Tools read from hiara package:                                          │
│  • hiara.base.retrieve_sig_stats()  - Get aging signatures              │
│  • hiara.base.retrieve_stats()      - Get intervention effects          │
│                                                                           │
│  Data source: /Users/jno24/Documents/projs/ongoing/hiara/base_folder/   │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    AGENT REASONING                                        │
│                                                                           │
│  GPT-4o receives tool results and:                                       │
│  1. Interprets the data                                                  │
│  2. Combines information from multiple tools if needed                   │
│  3. Uses conversation memory to understand context                       │
│  4. Formulates natural language response                                 │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    RESPONSE TO USER                                       │
│                                                                           │
│  Natural language answer explaining:                                     │
│  • Aging signatures (genes increasing/decreasing with age)               │
│  • Intervention effects (what treatments do to cell types)               │
│  • Comparisons and insights                                              │
└──────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE CONVERSATION FLOW                              │
└──────────────────────────────────────────────────────────────────────────┘

User: "What is the aging story of CD8T?"
  │
  ▼
Agent: [Validates "CD8T" using get_available_cell_types()]
  │
  ▼
Agent: [Calls get_aging_signature(cell_type="CD8T", analysis_name="tfa_major_b")]
  │
  ▼
Agent: "The aging signature for CD8T cells highlights changes in 368 genes:
       - Decreasing: AEBP1, AHR, ATF6, BCL11B...
       - Increasing: AEBP2, ARHGAP35, BAZ2A, CEBPG..."
  │
  │ [Memory stores: User question + Agent response]
  │
  ▼
User: "How does Ruxolitinib affect that?"
  │
  ▼
Agent: [Loads memory - sees "that" = CD8T from previous context]
  │
  ▼
Agent: [Validates "Ruxolitinib" using get_available_interventions()]
  │
  ▼
Agent: [Calls get_intervention_signature(intervention="Ruxolitinib", 
                                         cell_type="CD8T", 
                                         analysis_name="tfa_major_b")]
  │
  ▼
Agent: "Ruxolitinib does not have significant effects on CD8T cells - 
       no significant features were found."
  │
  │ [Memory stores: User question + Agent response]
  │
  ▼
User: "What about MONO?"
  │
  ▼
Agent: [Loads memory - understands context is about Ruxolitinib effects]
  │
  ▼
Agent: [Calls get_intervention_signature(intervention="Ruxolitinib", 
                                         cell_type="MONO", 
                                         analysis_name="tfa_major_b")]
  │
  ▼
Agent: [Returns Ruxolitinib effects on MONO cells]


┌──────────────────────────────────────────────────────────────────────────┐
│                    KEY TECHNICAL DETAILS                                  │
└──────────────────────────────────────────────────────────────────────────┘

1. FRAMEWORK: Microsoft Agent Framework (agent-framework==1.0.0b260212)
   • OpenAIChatClient with GPT-4o model
   • Tool calling via @tool decorator
   • Context providers for memory management

2. MEMORY IMPLEMENTATION:
   • InMemoryHistoryProvider(source_id="in_memory")
   • Explicitly added to context_providers list
   • NON-STREAMING mode (streaming breaks after_run hook)
   • Messages stored in session.state["in_memory"]["messages"]

3. TOOLS (4 total):
   • get_available_cell_types() - Validation
   • get_available_interventions() - Validation
   • get_aging_signature() - Data retrieval
   • get_intervention_signature() - Data retrieval

4. DATA SOURCE:
   • Custom hiara package functions
   • Pre-computed results stored in base_folder/
   • Analysis types: tfa_major_b, tfa_sub_b, etc.

5. VALIDATION WORKFLOW:
   • Agent MUST validate names before data retrieval
   • If invalid: Shows available options, asks for clarification
   • If valid: Proceeds with data retrieval
   • Handles fuzzy matches (e.g., "monocytes" → "MONO")
