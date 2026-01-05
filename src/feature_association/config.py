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
        # ===== AGING ANALYSES =====
        
        # 1. Pure aging - ALL samples (no filtering)
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='aging',
        #     condition_column='age_group',
        #     control_group='young',
        #     treatment_groups=['old'],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ age_group',  # Simple age effect
        #     mixed_effects_group='donor_id',
        #     comparison_mode='same',
        #     display_name='Sound Life (Aging - All Samples)',
        #     config_label='aging_all',
        #     data_filter=None,  # No filtering - use all samples
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # # 2. Pure aging - Baseline only (original analysis)
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='aging',
        #     condition_column='age_group',
        #     control_group='young',
        #     treatment_groups=['old'],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='same',
        #     display_name='Sound Life (Aging - Baseline)',
        #     config_label='aging_baseline',
        #     data_filter={
        #         'sample.visitName': ['Flu Year 1 Day 0', 'Flu Year 2 Day 0', 
        #                              'Immune Variation Day 0', 'Immune Variation Day 7', 
        #                              'Immune Variation Day 90']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # # 3. Pure aging - CMV Negative only
        ConditionConfig(
            name="soundlife",
            analysis_type='aging',
            condition_column='age_group',
            control_group='young',
            treatment_groups=['old'],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ age_group + C(Q("sample.visitName"))',
            mixed_effects_group='donor_id',
            comparison_mode='same',
            display_name='Sound Life (Aging - CMV Negative)',
            config_label='aging_cmv_neg',
            data_filter={
                'subject.cmv': ['Negative']
            },
            name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'},
            # Clock analysis configuration
            clock_pretty_names={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        ),
        
        # 4. Pure aging - CMV Positive only
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='aging',
        #     condition_column='age_group',
        #     control_group='young',
        #     treatment_groups=['old'],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='same',
        #     display_name='Sound Life (Aging - CMV Positive)',
        #     config_label='aging_cmv_pos',
        #     data_filter={
        #         'subject.cmv': ['Positive']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # ===== VACCINATION ANALYSES - ALL SUBJECTS =====
        
        # # Day 0: Baseline (all subjects)
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='perturbation',
        #     condition_column='vaccinated',
        #     control_group=False,
        #     treatment_groups=[True],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ vaccinated + age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='opposite',
        #     display_name='Sound Life (Vaccination Day 0 - All)',
        #     config_label='vacc_d0_all',
        #     data_filter={
        #         'sample.visitName': ['Flu Year 1 Day 0', 'Flu Year 2 Day 0', 'Immune Variation Day 0']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # Day 7: Peak response (all subjects)
        ConditionConfig(
            name="soundlife",
            analysis_type='perturbation',
            condition_column='vaccinated',
            control_group=False,
            treatment_groups=[True],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ vaccinated + age_group',
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (Vaccination Day 7 - All)',
            config_label='vacc_d7_all',
            data_filter={
                'sample.visitName': ['Flu Year 1 Day 7', 'Flu Year 2 Day 7', 'Immune Variation Day 7']
            },
            name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        ),
        
        # Day 90: Memory response (all subjects)
        ConditionConfig(
            name="soundlife",
            analysis_type='perturbation',
            condition_column='vaccinated',
            control_group=False,
            treatment_groups=[True],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ vaccinated + age_group',
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (Vaccination Day 90 - All)',
            config_label='vacc_d90_all',
            data_filter={
                'sample.visitName': ['Flu Year 1 Day 90', 'Flu Year 2 Day 90', 'Immune Variation Day 90']
            },
            name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        ),
        
        # # ===== VACCINATION ANALYSES - CMV NEGATIVE =====
        
        # Day 0 - CMV Negative
        ConditionConfig(
            name="soundlife",
            analysis_type='perturbation',
            condition_column='vaccinated',
            control_group=False,
            treatment_groups=[True],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ vaccinated + age_group',
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (Vaccination Day 0 - CMV Neg)',
            config_label='vacc_d0_cmv_neg',
            data_filter={
                'sample.visitName': ['Flu Year 1 Day 0', 'Flu Year 2 Day 0', 'Immune Variation Day 0'],
                'subject.cmv': ['Negative']
            },
            name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        ),
        
        # # Day 7 - CMV Negative
        ConditionConfig(
            name="soundlife",
            analysis_type='perturbation',
            condition_column='vaccinated',
            control_group=False,
            treatment_groups=[True],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ vaccinated + age_group',
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (Vaccination Day 7 - CMV Neg)',
            config_label='vacc_d7_cmv_neg',
            data_filter={
                'sample.visitName': ['Flu Year 1 Day 7', 'Flu Year 2 Day 7', 'Immune Variation Day 7'],
                'subject.cmv': ['Negative']
            },
            name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        ),
        
        # # Day 90 - CMV Negative
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='perturbation',
        #     condition_column='vaccinated',
        #     control_group=False,
        #     treatment_groups=[True],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ vaccinated + age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='opposite',
        #     display_name='Sound Life (Vaccination Day 90 - CMV Neg)',
        #     config_label='vacc_d90_cmv_neg',
        #     data_filter={
        #         'sample.visitName': ['Flu Year 1 Day 90', 'Flu Year 2 Day 90', 'Immune Variation Day 90'],
        #         'subject.cmv': ['Negative']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # # ===== VACCINATION ANALYSES - CMV POSITIVE =====
        
        # # Day 0 - CMV Positive
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='perturbation',
        #     condition_column='vaccinated',
        #     control_group=False,
        #     treatment_groups=[True],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ vaccinated + age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='opposite',
        #     display_name='Sound Life (Vaccination Day 0 - CMV Pos)',
        #     config_label='vacc_d0_cmv_pos',
        #     data_filter={
        #         'sample.visitName': ['Flu Year 1 Day 0', 'Flu Year 2 Day 0', 'Immune Variation Day 0'],
        #         'subject.cmv': ['Positive']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # # Day 7 - CMV Positive
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='perturbation',
        #     condition_column='vaccinated',
        #     control_group=False,
        #     treatment_groups=[True],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ vaccinated + age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='opposite',
        #     display_name='Sound Life (Vaccination Day 7 - CMV Pos)',
        #     config_label='vacc_d7_cmv_pos',
        #     data_filter={
        #         'sample.visitName': ['Flu Year 1 Day 7', 'Flu Year 2 Day 7', 'Immune Variation Day 7'],
        #         'subject.cmv': ['Positive']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # # Day 90 - CMV Positive
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='perturbation',
        #     condition_column='vaccinated',
        #     control_group=False,
        #     treatment_groups=[True],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ vaccinated + age_group',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='opposite',
        #     display_name='Sound Life (Vaccination Day 90 - CMV Pos)',
        #     config_label='vacc_d90_cmv_pos',
        #     data_filter={
        #         'sample.visitName': ['Flu Year 1 Day 90', 'Flu Year 2 Day 90', 'Immune Variation Day 90'],
        #         'subject.cmv': ['Positive']
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # ===== COMBINED TEMPORAL ANALYSIS - ALL COHORTS & INTERACTIONS =====
        # Interaction model: Tests how vaccination effects vary by age and CMV status
        # Also tests how CMV effects vary by age
        # Formula includes 2-way interactions: day*age, day*CMV, CMV*age
        # ConditionConfig(
        #     name="soundlife",
        #     analysis_type='perturbation',
        #     condition_column='followup_day',  # Primary variable for temporal analysis
        #     control_group=0,  # Day 0 as reference (pre-vaccination baseline)
        #     treatment_groups=[7, 90],
        #     test_type='mixed-effect',
        #     mixed_effects_formula='feature_values ~ C(followup_day) * C(age_group) + C(followup_day) * C(Q("subject.cmv")) + C(Q("subject.cmv")) * C(age_group) + C(vaccinated)',
        #     mixed_effects_group='donor_id',
        #     comparison_mode='same',
        #     display_name='Sound Life (Vaccination with Interactions)',
        #     config_label='vacc_temporal_interactions',
        #     data_filter={
        #         'sample.visitName': [
        #             'Flu Year 1 Day 0', 'Flu Year 1 Day 7', 'Flu Year 1 Day 90',
        #             'Flu Year 2 Day 0', 'Flu Year 2 Day 7', 'Flu Year 2 Day 90',
        #             'Immune Variation Day 0', 'Immune Variation Day 7', 'Immune Variation Day 90'
        #         ]
        #     },
        #     name_mapping={'young': 'Young (25-35y)', 'old': 'Older (55-65y)'}
        # ),
        
        # # ===== DISEASE (CMV) EFFECT ANALYSES =====
        # CMV effect across BOTH age groups (combined analysis)
        ConditionConfig(
            name="soundlife",
            analysis_type='disease',
            condition_column='subject.cmv',
            control_group='Negative',
            treatment_groups=['Positive'],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ C(Q("subject.cmv")) + C(Q("sample.visitName"))',  # Escape both column names with dots
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (CMV Effect)',
            config_label='cmv',
            data_filter=None,  # No age group filter - include both young and old
            name_mapping={'Negative': 'CMV-', 'Positive': 'CMV+'},
            # Clock analysis configuration
            clock_test_type='unpaired',
            clock_pvalue_threshold=0.05
        ),
        
        # CMV effect in YOUNG subjects
        ConditionConfig(
            name="soundlife",
            analysis_type='disease',
            condition_column='subject.cmv',
            control_group='Negative',
            treatment_groups=['Positive'],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ C(Q("subject.cmv")) + C(Q("sample.visitName"))',  # Escape both column names with dots
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (CMV Effect - Young)',
            config_label='cmv_young',
            data_filter={
                'age_group': 'young'
            },
            name_mapping={'Negative': 'CMV-', 'Positive': 'CMV+'},
            # Clock analysis configuration
            clock_test_type='unpaired',
            clock_pvalue_threshold=0.05
        ),
        
        # CMV effect in OLD subjects
        ConditionConfig(
            name="soundlife",
            analysis_type='disease',
            condition_column='subject.cmv',
            control_group='Negative',
            treatment_groups=['Positive'],
            test_type='mixed-effect',
            mixed_effects_formula='feature_values ~ C(Q("subject.cmv")) + C(Q("sample.visitName"))',  # Escape both column names with dots
            mixed_effects_group='donor_id',
            comparison_mode='opposite',
            display_name='Sound Life (CMV Effect - Old)',
            config_label='cmv_old',
            data_filter={
                'age_group': 'old'
            },
            name_mapping={'Negative': 'CMV-', 'Positive': 'CMV+'},
            # Clock analysis configuration
            clock_test_type='unpaired',
            clock_pvalue_threshold=0.05
        ),
    ],
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
    
    def get_analysis_type(config):
        """Helper to get analysis type from config or list of configs."""
        if isinstance(config, list):
            return config[0].analysis_type
        return config.analysis_type
    
    return [
        name for name, config in DATASET_CONFIGS.items()
        if get_analysis_type(config) == analysis_type
    ]
