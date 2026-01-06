"""
Configuration for condition-based analyses (disease and perturbation).

This module centralizes all dataset-specific configurations to eliminate
code duplication between disease and perturbation analyses.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
from collections import OrderedDict

# Dataset name mapping: raw -> processed
DATASET_NAME_MAPPING = {
    "data1": "onek1k",
    "data7_allTPs_jalil": "abf300",
    "data12": "zhang",
    "data13": "aida"
}


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
    
    # Data filtering (for splitting datasets into subsets)
    data_filter: Optional[Dict[str, any]] = None  # E.g., {'vaccinated': True, 'followup_day': [0, 7, 90]}
    
    # Comparison mode
    comparison_mode: Literal['opposite', 'same', 'both'] = 'opposite'
    
    # Special handling
    control_mapping: Optional[Dict[str, str]] = None  # For datasets with multiple controls
    name_mapping: Optional[Dict[str, str]] = None     # Rename conditions for display
    condition_mapping: Optional[Dict[str, str]] = None  # Transform actual condition values in data (applied when loading)
    
    # Display
    display_name: Optional[str] = None
    config_label: Optional[str] = None  # Label for this specific config (when multiple configs per dataset)
    
    # Plotting control
    target_treatments: Optional[List[str]] = None  # Which conditions to plot in overlap analysis
    
    # ========== CLOCK ANALYSIS SPECIFIC ==========
    # Statistical testing for clock predictions
    clock_test_type: Optional[str] = None  # 'paired', 'unpaired', 'mixed_effect'
    clock_group_key: Optional[str] = None  # For mixed effects in clock analysis (e.g., 'donor_id')
    clock_pvalue_correction: Optional[str] = 'corrected'  # 'raw' or 'corrected' (FDR)
    clock_pvalue_threshold: float = 0.05  # Significance threshold
    
    # Experiment pairs for perturbation datasets (list of tuples)
    clock_experiments: Optional[List[tuple]] = None  # [(control, treatment), ...]
    
    # Display options for clock plots
    clock_pretty_names: Optional[Dict[str, str]] = None  # Rename conditions for clock plots
    clock_mock_names: bool = False  # Mock compound names (keep top 1, rename others)
    
    # Plot configuration for perturbations
    clock_plot_config: Optional[Dict[str, any]] = None  # Dataset-specific plot parameters
    
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
        control_group='healthy',
        treatment_groups=['SLE'],
        test_type='unpaired',
        comparison_mode='same',  # Pathological aging (same direction as aging)
        display_name='SLE',
        name_mapping={
            'normal': 'healthy',
            'systemic lupus erythematosus': 'SLE'
        },
        condition_mapping={
            'normal': 'healthy',
            'systemic lupus erythematosus': 'SLE'
        },
        # Clock analysis settings
        clock_test_type='unpaired',
        clock_pvalue_threshold=0.05,
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
        treatment_groups=['Ruxolitinib'],  # Auto-detect all drugs
        test_type='mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        mixed_effects_group='plate_name',
        comparison_mode='opposite',  # Rejuvenating (opposite to aging)
        display_name='OP Compounds',
        target_treatments=['Ruxolitinib'],
        # Clock analysis settings
        clock_test_type='mixed_effect',
        clock_group_key='plate_name',
        clock_pvalue_correction='corrected',
        clock_pvalue_threshold=0.05,
        clock_experiments='auto',  # Auto-generate from data: (control, treatment) for all treatments
        clock_mock_names=True,  # Mock compound names (keep top 1, rename others)
        clock_plot_config={
            'rejuvenating': {'figsize': (7.5, 3), 'margins': (0.05, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
            'aging': {'figsize': (5, 3), 'margins': (0.05, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
        },
        name_mapping = {}
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
        # Clock analysis settings
        clock_test_type='mixed_effect',
        clock_group_key='donor_id',
        clock_pvalue_correction='raw',
        clock_pvalue_threshold=0.05,
        clock_experiments=[
            ('24 h RPMI', '24 h LPS'),
            ('24 h RPMI', '24 h RPMI + ruxolitinib'),
            ('24 h LPS', '24 h LPS + ruxolitinib'),
        ],
        clock_pretty_names={
            '24 h LPS': 'LPS \n (ctr: RPMI)',
            '24 h LPS + ruxolitinib': 'Ruxolitinib \n (ctr: LPS)',
            '24 h RPMI': 'RPMI',
            '24 h RPMI + ruxolitinib': 'Ruxolitinib \n (ctr: RPMI)',
        },
        clock_plot_config={
            'rejuvenating': {'figsize': (4, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
            'aging': {'figsize': (7, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        },
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
        # Clock analysis settings
        clock_test_type='mixed_effect',
        clock_group_key='donor_id',
        clock_pvalue_correction='corrected',
        clock_pvalue_threshold=0.05,
        clock_experiments='auto',  # Auto-generate from data
        clock_plot_config={
            'CD4T': {
                'rejuvenating': {'figsize': (4, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
                'aging': {'figsize': (10, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
            },
            'default': {
                'rejuvenating': {'figsize': (7, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
                'aging': {'figsize': (10, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
            },
        },
    ),
    
    # ========== AGING DATASETS (Longitudinal) ==========
    "soundlife": [
        # Pure aging - ALL samples (no filtering)
        ConditionConfig(
            name="soundlife",
            analysis_type='aging',
            condition_column='age_group',
            control_group='young',
            treatment_groups=['old'],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ age_group',  # Simple age effect
            mixed_effects_group='donor_id',
            comparison_mode='same',
            display_name='Sound Life (Aging - All Samples)',
            config_label='aging_all',
            data_filter=None,  # No filtering - use all samples
            name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        )
    ]
}


def get_config(dataset_name: str, config_label: Optional[str] = None) -> List[ConditionConfig]:
    """
    Get configuration(s) for a dataset.
    
    Datasets can have either a single config or a list of configs.
    Always returns a list for consistent handling.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    config_label : str, optional
        Specific config label to retrieve (for datasets with multiple configs)
    
    Returns
    -------
    List[ConditionConfig]
        List of configuration objects (even if only one config)
    
    Raises
    ------
    ValueError
        If dataset not found or config_label not found
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Ensure we always return a list
    if isinstance(config, list):
        # If config_label is specified, filter to that specific config
        if config_label is not None:
            matching_configs = [c for c in config if c.config_label == config_label]
            if not matching_configs:
                available_labels = [c.config_label for c in config if c.config_label]
                raise ValueError(
                    f"Config label '{config_label}' not found for dataset '{dataset_name}'. "
                    f"Available labels: {', '.join(available_labels)}"
                )
            return matching_configs
        return config
    else:
        return [config]

