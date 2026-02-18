"""Helper functions for agentic analysis pipeline."""

import pandas as pd
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from agent_framework import tool
from hiara.src.feature_association.helper import retrieve_sig_stats

@tool
def get_aging_signature(cell_type: str, analysis_name: str = 'tfa_major_b') -> str:
    """Retrieve aging-associated features for a cell type."""
    try:
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            cell_type=cell_type
        )
        
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
        )
        
        output = f"INTERVENTION SIGNATURE FOR {intervention.upper()} in {cell_type.upper()} ({analysis_name})\n"
        output += f"Total significant features: {len(stats_df)}\n\n"
        
        decreasing = stats_df[stats_df['slope'] < 0].copy() if 'slope' in stats_df.columns else stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy() if 'slope' in stats_df.columns else stats_df[stats_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            output += "DECREASED BY INTERVENTION:\n"
            for _, row in decreasing.head(20).iterrows():
                slope_col = 'slope_condition' if 'slope_condition' in row else 'slope'
                p_col = 'p_value_adj' if 'p_value_adj' in row else 'p_adj'
                output += f"  ↓ {row['gene']}: slope={row[slope_col]:.4f}, p_adj={row.get(p_col, 0):.4f}\n"
        
        if len(increasing) > 0:
            output += "\nINCREASED BY INTERVENTION:\n"
            for _, row in increasing.head(20).iterrows():
                slope_col = 'slope_condition' if 'slope_condition' in row else 'slope'
                p_col = 'p_value_adj' if 'p_value_adj' in row else 'p_adj'
                output += f"  ↑ {row['gene']}: slope={row[slope_col]:.4f}, p_adj={row.get(p_col, 0):.4f}\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving intervention signature: {e}"
