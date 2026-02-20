"""Helper functions for agentic analysis pipeline."""

import pandas as pd
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from agent_framework import tool
from hiara.src.feature_association.helper import retrieve_sig_stats
from hiara.src.config import ANALYSIS_DEF, FEATURE_DEF
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
def get_available_analysis_types() -> str:
    """
    Get all available analysis types with descriptions.
    Use this to understand what data is available and select appropriate analysis types based on the user's question.
    
    Analysis types represent different feature types and granularities:
    - TF activity (tfa_*): Transcription factor regulatory activity
    - Gene expression (ge_*): Raw gene expression levels
    - Cell type features (ct_*): Cell composition and polarization
    - Cell-cell communication (ccc_*): Intercellular signaling
    """
    output = "AVAILABLE ANALYSIS TYPES:\n\n"
    for analysis_name, description in ANALYSIS_DEF.items():
        output += f"• {analysis_name}:\n  {description}\n\n"
    
    output += "\nRECOMMENDATIONS:\n"
    output += "• For TF/transcription factor questions → use tfa_major_b or tfa_sub_b\n"
    output += "• For gene expression questions → use ge_major_b or ge_sub_b\n"
    output += "• For cell composition/frequency → use ct_freq\n"
    output += "• For cell polarization → use ct_pol_dist\n"
    output += "• For cell communication/interactions → use ccc_sub_b\n"
    output += "• For major cell types (CD8T, CD4T, MONO, NK, B) → use *_major_b\n"
    output += "• For sub cell types (Tcm_Naive_CD4, etc.) → use *_sub_b\n"
    
    return output


@tool
def get_available_features() -> str:
    """
    Get all available feature types with descriptions.
    Use this to understand what biological features can be analyzed.
    """
    output = "AVAILABLE FEATURE TYPES:\n\n"
    for feature_name, description in FEATURE_DEF.items():
        output += f"• {feature_name}:\n  {description}\n\n"
    
    return output

@tool
def get_aging_signature(cell_type: str, analysis_name: str) -> str:
    """
    Retrieve aging-associated features for a cell type from a specific analysis.
    
    Args:
        cell_type: Cell type name (e.g., 'CD8T', 'MONO'). Use get_available_cell_types() to see valid names.
        analysis_name: Analysis type to use (e.g., 'tfa_major_b', 'ge_major_b', 'ct_freq'). 
                      Use get_available_analysis_types() to see all options and select based on the question.
    
    Returns:
        String with aging signature including genes/features that increase or decrease with age.
    
    Examples:
        - For TF activity in CD8T: get_aging_signature('CD8T', 'tfa_major_b')
        - For gene expression in MONO: get_aging_signature('MONO', 'ge_major_b')
        - For cell composition: get_aging_signature('CD8T', 'ct_freq')
    """
    try:
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            cell_type=cell_type
        ).drop_duplicates(subset=['cell_type', 'gene'])
        
        output = f"AGING SIGNATURE FOR {cell_type.upper()} - {analysis_name}\n"
        output += f"Analysis: {ANALYSIS_DEF.get(analysis_name, analysis_name)}\n"
        output += f"Total significant features: {len(stats_df)}\n\n"
        
        decreasing = stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            output += "DECREASING WITH AGE:\n"
            for _, row in decreasing.head(20).iterrows():
                output += f"  ↓ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get('p_adj', row.get('meta_p_adj', 0)):.4f}\n"
            if len(decreasing) > 20:
                output += f"  ... and {len(decreasing) - 20} more\n"
        
        if len(increasing) > 0:
            output += "\nINCREASING WITH AGE:\n"
            for _, row in increasing.head(20).iterrows():
                output += f"  ↑ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get('p_adj', row.get('meta_p_adj', 0)):.4f}\n"
            if len(increasing) > 20:
                output += f"  ... and {len(increasing) - 20} more\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving aging signature for {cell_type} with {analysis_name}: {e}"


@tool
def get_intervention_signature(intervention: str, cell_type: str, analysis_name: str) -> str:
    """
    Retrieve intervention effects for a cell type from a specific analysis.
    
    Args:
        intervention: Intervention name (e.g., 'Ruxolitinib', 'Metformin'). Use get_available_interventions() to see valid names.
        cell_type: Cell type name (e.g., 'CD8T', 'MONO'). Use get_available_cell_types() to see valid names.
        analysis_name: Analysis type to use (e.g., 'tfa_major_b', 'ge_major_b'). 
                      Use get_available_analysis_types() to see all options and select based on the question.
    
    Returns:
        String with intervention signature including genes/features that are increased or decreased by the intervention.
    
    Examples:
        - For TF activity effects: get_intervention_signature('Ruxolitinib', 'CD8T', 'tfa_major_b')
        - For gene expression effects: get_intervention_signature('Metformin', 'MONO', 'ge_major_b')
    """
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
            return f"No significant features found for {intervention} in {cell_type} using {analysis_name}. The intervention may not have significant effects in this cell type for this analysis, or the data may not be available."
        
        output = f"INTERVENTION SIGNATURE FOR {intervention.upper()} in {cell_type.upper()} - {analysis_name}\n"
        output += f"Analysis: {ANALYSIS_DEF.get(analysis_name, analysis_name)}\n"
        output += f"Total significant features: {len(stats_df)}\n\n"
        
        decreasing = stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            output += "DECREASED BY INTERVENTION:\n"
            for _, row in decreasing.head(20).iterrows():
                p_col = 'p_value_adj' if 'p_value_adj' in row else 'p_adj'
                output += f"  ↓ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get(p_col, 0):.4f}\n"
            if len(decreasing) > 20:
                output += f"  ... and {len(decreasing) - 20} more\n"
        
        if len(increasing) > 0:
            output += "\nINCREASED BY INTERVENTION:\n"
            for _, row in increasing.head(20).iterrows():
                p_col = 'p_value_adj' if 'p_value_adj' in row else 'p_adj'
                output += f"  ↑ {row['gene']}: slope={row['slope']:.4f}, p_adj={row.get(p_col, 0):.4f}\n"
            if len(increasing) > 20:
                output += f"  ... and {len(increasing) - 20} more\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving intervention signature for {intervention} in {cell_type} with {analysis_name}: {e}"
