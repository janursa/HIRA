"""Helper functions for agentic analysis pipeline."""

import pandas as pd
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from agent_framework import tool
from hiara.src.feature_association.helper import retrieve_sig_stats
from config import interventions, cell_types, interventions_2_dataset


@tool
def get_available_cell_types() -> str:
    """
    Get list of all available cell types for analysis.
    Use this to validate cell type names before querying data.
    """
    output = "AVAILABLE CELL TYPES:\n\n"
    for granularity, cts in cell_types.items():
        output += f"{granularity}:\n"
        output += f"  {', '.join(cts)}\n\n"
    output += "Common aliases:\n"
    output += "  - 'monocytes' or 'mono' → MONO\n"
    output += "  - 'cd8 t cells', 'cd8t', 'cd8' → CD8T\n"
    output += "  - 'cd4 t cells', 'cd4t', 'cd4' → CD4T\n"
    output += "  - 'natural killer', 'nk cells' → NK\n"
    output += "  - 'b cells' → B\n"
    return output


@tool
def get_available_interventions() -> str:
    """
    Get list of all available interventions (drugs and cytokines).
    Use this to validate intervention names before querying data.
    """
    output = "AVAILABLE INTERVENTIONS:\n\n"
    for intervention_type, intervention_list in interventions.items():
        output += f"{intervention_type.upper()} (dataset: {interventions_2_dataset[intervention_type]}):\n"
        sorted_list = sorted(intervention_list)
        for i in sorted_list:
            output += f"  - {i}\n"
        output += f"\nTotal: {len(intervention_list)} {intervention_type}s\n\n"
    return output

@tool
def get_aging_signature(cell_type: str, analysis_name: str = 'tfa_major_b') -> str:
    """Retrieve aging-associated features for a cell type."""
    try:
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            cell_type=cell_type
        ).drop_duplicates(subset=['cell_type', 'gene'])
        
        output = f"AGING SIGNATURE FOR {cell_type.upper()} ({analysis_name})\n"
        output += f"Total significant features: {len(stats_df)}\n\n"
        
        decreasing = stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            output += "DECREASING WITH AGE:\n"
            for _, row in decreasing.head(20).iterrows():
                output += f"  ↓ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get('p_adj', row.get('meta_p_adj', 0)):.4f}\n"
        
        if len(increasing) > 0:
            output += "\nINCREASING WITH AGE:\n"
            for _, row in increasing.head(20).iterrows():
                output += f"  ↑ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get('p_adj', row.get('meta_p_adj', 0)):.4f}\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving aging signature: {e}"


@tool
def get_intervention_signature(intervention: str, cell_type: str, analysis_name: str = 'tfa_major_b') -> str:
    """Retrieve intervention effects for a cell type."""
    try:
        
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            dataset='op',
            cell_type=cell_type
        ).drop_duplicates(subset=['cell_type', 'gene', 'comparison'])
        
        # Filter for this specific intervention and significant results
        stats_df = stats_df[
            (stats_df['comparison'] == intervention) 
        ].copy()
        
        if len(stats_df) == 0:
            return f"No significant features found for {intervention} in {cell_type}. The intervention may not have significant effects in this cell type or the data may not be available."
        
        output = f"INTERVENTION SIGNATURE FOR {intervention.upper()} in {cell_type.upper()} ({analysis_name})\n"
        output += f"Total significant features: {len(stats_df)}\n\n"
        
        decreasing = stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            output += "DECREASED BY INTERVENTION:\n"
            for _, row in decreasing.head(20).iterrows():
                p_col = 'p_value_adj' if 'p_value_adj' in row else 'p_adj'
                output += f"  ↓ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get(p_col, 0):.4f}\n"
        
        if len(increasing) > 0:
            output += "\nINCREASED BY INTERVENTION:\n"
            for _, row in increasing.head(20).iterrows():
                p_col = 'p_value_adj' if 'p_value_adj' in row else 'p_adj'
                output += f"  ↑ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get(p_col, 0):.4f}\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving intervention signature: {e}"
