"""
Configuration for condition-based analyses (disease and perturbation).

This module centralizes all dataset-specific configurations to eliminate
code duplication between disease and perturbation analyses.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
from collections import OrderedDict


@dataclass
class ConditionConfig:
    """Configuration for a specific dataset analysis."""
    
    # Core identifiers
    name: str
    analysis_type: Literal['disease', 'perturbation', 'aging']
    
    # Data columns
    condition_column: str  # Column name in obs: 'condition', 'perturbation', 'Max_WHO_Group'
    control_group: str     # Control group name: 'healthy', 'PBS', 'Dimethyl Sulfoxide'
    
    # Treatments to compare (can be 'all' for auto-detection)
    treatment_groups: List[str] | Literal['all'] = 'all'
    
    # Statistical parameters
    test_type: str = 'unpaired'  # 'unpaired', 'mixed-effect', 'paired'
    
    # Mixed-effects model parameters (used when test_type='mixed-effect')
    mixed_effects_formula: Optional[str] = None  # R-style formula, e.g., "feature_values ~ condition"
    mixed_effects_group: Optional[str] = None    # Random effects grouping variable, e.g., 'donor_id'
    
    # Comparison mode
    comparison_mode: Literal['opposite', 'same', 'both'] = 'opposite'
    
    # Special handling
    control_mapping: Optional[Dict[str, str]] = None  # For datasets with multiple controls
    name_mapping: Optional[Dict[str, str]] = None     # Rename conditions for display
    
    # Display
    display_name: Optional[str] = None
    
    # Plotting control
    target_treatments: Optional[List[str]] = None  # Which conditions to plot in overlap analysis
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name
        
        # Auto-detect condition column variants
        if self.name in ['op', 'parsebioscience']:
            # These datasets might have 'condition' or 'perturbation'
            self.condition_column_variants = ['condition', 'perturbation']
        else:
            self.condition_column_variants = [self.condition_column]


# Dataset configurations
DATASET_CONFIGS = {
    # ========== DISEASE DATASETS ==========
    "SLE_European": ConditionConfig(
        name="SLE_European",
        analysis_type='disease',
        condition_column='condition',
        control_group='normal',
        treatment_groups=['systemic lupus erythematosus'],
        test_type='unpaired',
        comparison_mode='same',  # Pathological aging (same direction as aging)
        display_name='SLE',
        name_mapping={
            'normal': 'healthy',
            'systemic lupus erythematosus': 'SLE'
        }
    ),
    
    "Covid_50MHH": ConditionConfig(
        name="Covid_50MHH",
        analysis_type='disease',
        condition_column='Max_WHO_Group',
        control_group='mild',
        treatment_groups='all',  # Will include moderate, severe
        test_type='unpaired',
        comparison_mode='same',
        display_name='COVID-19',
    ),
    
    # ========== PERTURBATION DATASETS ==========
    "op": ConditionConfig(
        name="op",
        analysis_type='perturbation',
        condition_column='condition',  # Will auto-detect 'perturbation' if needed
        control_group='Dimethyl Sulfoxide',
        treatment_groups='all',  # Auto-detect all drugs
        test_type='mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        mixed_effects_group='plate_name',
        comparison_mode='opposite',  # Rejuvenating (opposite to aging)
        display_name='OP Compounds',
        target_treatments=['Ruxolitinib']
    ),
    
    "CXCL9": ConditionConfig(
        name="CXCL9",
        analysis_type='perturbation',
        condition_column='condition',
        control_group=None,  # Multiple controls handled specially
        treatment_groups=[
            '24 h RPMI + ruxolitinib',
            '24 h LPS + ruxolitinib',
            '24 h LPS'
        ],
        test_type='mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        mixed_effects_group='donor_id',
        comparison_mode='opposite',
        display_name='Ruxolitinib',
        control_mapping={
            '24 h RPMI + ruxolitinib': '24 h RPMI',
            '24 h LPS + ruxolitinib': '24 h LPS',
            '24 h LPS': '24 h RPMI'
        },
        name_mapping=OrderedDict({
            '24 h RPMI + ruxolitinib': 'Ruxolitinib (ctr: RPMI)',
            '24 h LPS + ruxolitinib': 'Ruxolitinib (ctr: LPS)',
            '24 h LPS': 'LPS (ctr: RPMI)'
        }),
        target_treatments=['Ruxolitinib (ctr: RPMI)', 'Ruxolitinib (ctr: LPS)'],
    ),
    
    "parsebioscience": ConditionConfig(
        name="parsebioscience",
        analysis_type='perturbation',
        condition_column='condition',  # Will auto-detect 'perturbation' if needed
        control_group='PBS',
        treatment_groups='all',  # Auto-detect all cytokines
        test_type='mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        mixed_effects_group='donor_id',
        comparison_mode='opposite',
        display_name='Cytokines',
        target_treatments=['IL-10'],
    ),
    
    # ========== AGING DATASETS (Longitudinal) ==========
    "soundlife": ConditionConfig(
        name="soundlife",
        analysis_type='aging',
        condition_column='age_group',  # 'young' vs 'old'
        control_group='young',  # Baseline comparison group
        treatment_groups=['old'],  # Compare older adults to young
        test_type='mixed-effect',  # Account for repeated measures per donor
        mixed_effects_formula='feature_values ~ condition',  # Simple age effect (can be extended)
        mixed_effects_group='donor_id',  # Random intercept per donor
        comparison_mode='same',  # Aging signatures (same direction as reference aging)
        display_name='Sound Life (Aging)',
        target_treatments=['Older (55-65y)'],  # Focus on aging effects
        name_mapping={
            'young': 'Young (25-35y)',
            'old': 'Older (55-65y)'
        }
    ),
}


def get_config(dataset_name: str) -> ConditionConfig:
    """
    Get configuration for a dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    
    Returns
    -------
    ConditionConfig
        Configuration object
    
    Raises
    ------
    ValueError
        If dataset not found
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {available}"
        )
    return DATASET_CONFIGS[dataset_name]


def list_datasets(analysis_type: Optional[str] = None) -> List[str]:
    """
    List available datasets, optionally filtered by analysis type.
    
    Parameters
    ----------
    analysis_type : str, optional
        Filter by 'disease', 'perturbation', or None for all
    
    Returns
    -------
    List[str]
        List of dataset names
    """
    if analysis_type is None:
        return list(DATASET_CONFIGS.keys())
    
    return [
        name for name, config in DATASET_CONFIGS.items()
        if config.analysis_type == analysis_type
    ]
