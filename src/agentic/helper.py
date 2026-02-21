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
        String with aging signature summary statistics and top 10 examples for each direction.
    
    Examples:
        - For TF activity in CD8T: get_aging_signature('CD8T', 'tfa_major_b')
        - For gene expression in MONO: get_aging_signature('MONO', 'ge_major_b')
        - For cell composition: get_aging_signature('CD8T', 'ct_freq')
    """
    try:
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            cell_type=cell_type,
            feature_props=['centrality']
        ).drop_duplicates(subset=['cell_type', 'gene'])
        
        decreasing = stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy()
        
        output = f"AGING SIGNATURE FOR {cell_type.upper()} - {analysis_name}\n"
        output += f"Analysis: {ANALYSIS_DEF.get(analysis_name, analysis_name)}\n\n"
        
        output += f"SUMMARY STATISTICS:\n"
        output += f"  • Total significant features: {len(stats_df)}\n"
        output += f"  • Features increasing with age: {len(increasing)}\n"
        output += f"  • Features decreasing with age: {len(decreasing)}\n\n"
        
        if len(decreasing) > 0:
            # Sort by centrality (descending) and get top 10 most central
            decreasing_sorted = decreasing.sort_values('centrality', ascending=False)
            output += "TOP 10 MOST CENTRAL - DECREASING WITH AGE:\n"
            genes = [row['gene'] for _, row in decreasing_sorted.head(10).iterrows()]
            output += "  " + ", ".join(genes) + "\n"
        
        if len(increasing) > 0:
            # Sort by centrality (descending) and get top 10 most central
            increasing_sorted = increasing.sort_values('centrality', ascending=False)
            output += "\nTOP 10 MOST CENTRAL - INCREASING WITH AGE:\n"
            genes = [row['gene'] for _, row in increasing_sorted.head(10).iterrows()]
            output += "  " + ", ".join(genes) + "\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving aging signature for {cell_type} with {analysis_name}: {e}"


@tool
def get_intervention_signature(intervention: str, cell_type: str, analysis_name: str) -> str:
    """
    Retrieve intervention effects for a cell type from a specific analysis.
    Compares with aging signatures to calculate overlap and directionality alignment.
    
    Args:
        intervention: Intervention name (e.g., 'Ruxolitinib', 'Metformin'). Use get_available_interventions() to see valid names.
        cell_type: Cell type name (e.g., 'CD8T', 'MONO'). Use get_available_cell_types() to see valid names.
        analysis_name: Analysis type to use (e.g., 'tfa_major_b', 'ge_major_b'). 
                      Use get_available_analysis_types() to see all options and select based on the question.
    
    Returns:
        String with intervention signature summary including overlap with aging and directionality alignment.
    
    Examples:
        - For TF activity effects: get_intervention_signature('Ruxolitinib', 'CD8T', 'tfa_major_b')
        - For gene expression effects: get_intervention_signature('Metformin', 'MONO', 'ge_major_b')
    """
    try:
        
        # Get intervention data
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            dataset='op',
            cell_type=cell_type,
            feature_props=['centrality']
        ).drop_duplicates(subset=['cell_type', 'gene', 'comparison'])
        
        # Filter for this specific intervention
        intervention_df = stats_df[
            (stats_df['comparison'] == intervention) 
        ].copy()
        
        if len(intervention_df) == 0:
            return f"No significant features found for {intervention} in {cell_type} using {analysis_name}. The intervention may not have significant effects in this cell type for this analysis, or the data may not be available."
        
        # Get aging signature for comparison
        aging_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            cell_type=cell_type
        ).drop_duplicates(subset=['cell_type', 'gene'])
        
        # Calculate overlap
        intervention_genes = set(intervention_df['gene'])
        aging_genes = set(aging_df['gene'])
        overlap_genes = intervention_genes & aging_genes
        
        # Calculate directionality alignment (opposite directions = therapeutic)
        aligned_genes = []
        for gene in overlap_genes:
            int_slope = intervention_df[intervention_df['gene'] == gene]['slope'].values[0]
            age_slope = aging_df[aging_df['gene'] == gene]['slope'].values[0]
            # Opposite slopes mean intervention reverses aging (therapeutic)
            if (int_slope > 0 and age_slope < 0) or (int_slope < 0 and age_slope > 0):
                aligned_genes.append(gene)
        
        overlap_percent = (len(overlap_genes) / len(intervention_genes) * 100) if len(intervention_genes) > 0 else 0
        aligned_percent = (len(aligned_genes) / len(overlap_genes) * 100) if len(overlap_genes) > 0 else 0
        
        output = f"INTERVENTION SIGNATURE FOR {intervention.upper()} in {cell_type.upper()} - {analysis_name}\n"
        output += f"Analysis: {ANALYSIS_DEF.get(analysis_name, analysis_name)}\n\n"
        
        output += f"SUMMARY STATISTICS:\n"
        output += f"  • Total features affected by {intervention}: {len(intervention_df)}\n"
        output += f"  • Features overlapping with aging signature: {len(overlap_genes)} ({overlap_percent:.1f}%)\n"
        output += f"  • Features with aligned directionality (reverses aging): {len(aligned_genes)} ({aligned_percent:.1f}% of overlap)\n\n"
        
        decreasing = intervention_df[intervention_df['slope'] < 0].copy()
        increasing = intervention_df[intervention_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            # Sort by centrality (descending) and get top 10 most central
            decreasing_sorted = decreasing.sort_values('centrality', ascending=False)
            output += "TOP 10 MOST CENTRAL - DECREASED BY INTERVENTION:\n"
            genes_with_markers = []
            for _, row in decreasing_sorted.head(10).iterrows():
                gene = row['gene']
                if gene in aligned_genes:
                    genes_with_markers.append(f"{gene} [REVERSES AGING]")
                elif gene in overlap_genes:
                    genes_with_markers.append(f"{gene} [in aging sig]")
                else:
                    genes_with_markers.append(gene)
            output += "  " + ", ".join(genes_with_markers) + "\n"
        
        if len(increasing) > 0:
            # Sort by centrality (descending) and get top 10 most central
            increasing_sorted = increasing.sort_values('centrality', ascending=False)
            output += "\nTOP 10 MOST CENTRAL - INCREASED BY INTERVENTION:\n"
            genes_with_markers = []
            for _, row in increasing_sorted.head(10).iterrows():
                gene = row['gene']
                if gene in aligned_genes:
                    genes_with_markers.append(f"{gene} [REVERSES AGING]")
                elif gene in overlap_genes:
                    genes_with_markers.append(f"{gene} [in aging sig]")
                else:
                    genes_with_markers.append(gene)
            output += "  " + ", ".join(genes_with_markers) + "\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving intervention signature for {intervention} in {cell_type} with {analysis_name}: {e}"


@tool
def get_disease_signature(disease: str, cell_type: str, analysis_name: str) -> str:
    """
    Retrieve disease-associated changes for a cell type from a specific analysis.
    Compares with aging signatures to show overlap and directionality concordance (accelerated aging).
    
    Args:
        disease: Disease name. Currently supported: 'SLE' (Systemic Lupus Erythematosus)
        cell_type: Cell type name (e.g., 'CD8T', 'CD4T', 'MONO'). Use get_available_cell_types() to see valid names.
        analysis_name: Analysis type to use (e.g., 'tfa_major_b', 'ge_major_b'). 
                      Use get_available_analysis_types() to see all options and select based on the question.
    
    Returns:
        String with disease signature summary including overlap with aging and directionality concordance.
    
    Examples:
        - For TF activity changes in SLE: get_disease_signature('SLE', 'CD8T', 'tfa_major_b')
        - For gene expression changes in SLE: get_disease_signature('SLE', 'CD4T', 'ge_major_b')
    """
    try:
        # Map disease name to dataset
        disease_dataset_map = {
            'SLE': 'perez_sle',
            'Lupus': 'perez_sle',
            'systemic lupus erythematosus': 'perez_sle'
        }
        
        dataset = disease_dataset_map.get(disease, disease_dataset_map.get(disease.upper()))
        if dataset is None:
            return f"Disease '{disease}' not recognized. Currently supported: SLE (Systemic Lupus Erythematosus)"
        
        # Get disease data
        stats_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            dataset=dataset,
            cell_type=cell_type,
            feature_props=['centrality']
        ).drop_duplicates(subset=['cell_type', 'gene'])
        
        if len(stats_df) == 0:
            return f"No significant features found for {disease} in {cell_type} using {analysis_name}."
        
        # Get aging signature for comparison
        aging_df = retrieve_sig_stats(
            analysis_name=analysis_name,
            cell_type=cell_type,
            feature_props=['centrality']
        ).drop_duplicates(subset=['cell_type', 'gene'])
        
        # Calculate overlap
        disease_genes = set(stats_df['gene'])
        aging_genes = set(aging_df['gene'])
        overlap_genes = disease_genes & aging_genes
        
        # Calculate directionality concordance (same direction = accelerates aging)
        concordant_genes = []
        for gene in overlap_genes:
            disease_slope = stats_df[stats_df['gene'] == gene]['slope'].values[0]
            age_slope = aging_df[aging_df['gene'] == gene]['slope'].values[0]
            # Same slopes mean disease accelerates aging pattern
            if (disease_slope > 0 and age_slope > 0) or (disease_slope < 0 and age_slope < 0):
                concordant_genes.append(gene)
        
        overlap_percent = (len(overlap_genes) / len(disease_genes) * 100) if len(disease_genes) > 0 else 0
        concordant_percent = (len(concordant_genes) / len(overlap_genes) * 100) if len(overlap_genes) > 0 else 0
        
        output = f"DISEASE SIGNATURE FOR {disease.upper()} in {cell_type.upper()} - {analysis_name}\n"
        output += f"Analysis: {ANALYSIS_DEF.get(analysis_name, analysis_name)}\n\n"
        
        output += f"SUMMARY STATISTICS:\n"
        output += f"  • Total features altered in {disease}: {len(stats_df)}\n"
        output += f"  • Features overlapping with aging signature: {len(overlap_genes)} ({overlap_percent:.1f}%)\n"
        output += f"  • Features with concordant directionality (accelerates aging): {len(concordant_genes)} ({concordant_percent:.1f}% of overlap)\n\n"
        
        decreasing = stats_df[stats_df['slope'] < 0].copy()
        increasing = stats_df[stats_df['slope'] > 0].copy()
        
        if len(decreasing) > 0:
            # Sort by centrality (descending) and get top 10 most central
            decreasing_sorted = decreasing.sort_values('centrality', ascending=False)
            output += "TOP 10 MOST CENTRAL - DECREASED IN DISEASE:\n"
            genes_with_markers = []
            for _, row in decreasing_sorted.head(10).iterrows():
                gene = row['gene']
                if gene in concordant_genes:
                    genes_with_markers.append(f"{gene} [ACCELERATES AGING]")
                elif gene in overlap_genes:
                    genes_with_markers.append(f"{gene} [in aging sig]")
                else:
                    genes_with_markers.append(gene)
            output += "  " + ", ".join(genes_with_markers) + "\n"
        
        if len(increasing) > 0:
            # Sort by centrality (descending) and get top 10 most central
            increasing_sorted = increasing.sort_values('centrality', ascending=False)
            output += "\nTOP 10 MOST CENTRAL - INCREASED IN DISEASE:\n"
            genes_with_markers = []
            for _, row in increasing_sorted.head(10).iterrows():
                gene = row['gene']
                if gene in concordant_genes:
                    genes_with_markers.append(f"{gene} [ACCELERATES AGING]")
                elif gene in overlap_genes:
                    genes_with_markers.append(f"{gene} [in aging sig]")
                else:
                    genes_with_markers.append(gene)
            output += "  " + ", ".join(genes_with_markers) + "\n"
        
        return output
        
    except Exception as e:
        return f"Error retrieving disease signature for {disease} in {cell_type} with {analysis_name}: {e}"
