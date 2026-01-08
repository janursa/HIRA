
"""
Configuration for condition-based analyses (disease and perturbation).
This module centralizes all dataset-specific configurations to eliminate
code duplication between disease and perturbation analyses.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
from collections import OrderedDict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict
import warnings
import os
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*anndata.*", category=FutureWarning)
# Variables
meta_analysis_min_cohorts = 3
grn_consensus_min_degree = 3
# Dataset name mapping: raw -> processed
DATASET_NAME_MAPPING = {
    "data1": "onek1k",
    "data7_allTPs_jalil": "abf300",
    "data12": "zhang",
    "data13": "aida",
    "SLE": "perez_sle"
}
try:
    base_dir = os.environ["CIIM_DATA_DIR"]
    task_grn_benchmark_dir = os.environ["TASK_GRN_BENCHMARK_DIR"]
except KeyError:
    import platform
    if platform.system() == 'Linux':
        base_dir = '/vol/projects/jnourisa/'
        # base_dir = '/home/jnourisa/projs/ongoing/ciim/'
        task_grn_benchmark_dir = '/home/jnourisa/projs/ongoing/task_grn_inference/'
    else:
        base_dir = '/Users/jno24/Documents/projs/ongoing/ciim/base_folder'
        task_grn_benchmark_dir = '/Users/jno24/Documents/projs/ongoing/task_grn_inference/'
SAVE_DIR = f'{base_dir}/output/'
FEATURES_DIR = f'{SAVE_DIR}/features/'
CLOCKS_DIR = f"{SAVE_DIR}/clock/"
PLOTS_DIR = f"{SAVE_DIR}/plots/"
PRIOR_DIR = f"{base_dir}/prior/"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CLOCKS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PRIOR_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

surrogate_names = {
                    'onek1k':'OneK1K',
                    'abf300': 'ABF300',
                    'zhang': 'Zhang',
                    'aida': 'AIDA',
                    'perez_sle': 'Perez',
                    'op': 'OPSCA',
                    'parsebioscience': 'Parse Bioscience', 
                    'soundlife': 'SoundLife',
                    'CXCL9': 'CXCL9',
                    }
# - datasets
ALL_DATASETS = ['onek1k', 'abf300', 'aida', 'perez_sle', 'CXCL9', 'op', 'parsebioscience', 'soundlife']
AGING_COHORTS = ['onek1k', 'abf300', 'soundlife', 'aida', 'perez_sle']
DISCOVERY_COHORTS = ['onek1k', 'abf300', 'aida', 'perez_sle']
# DISCOVERY_COHORTS = ['onek1k', 'abf300', 'zhang']
CLOCK_TRAINING_COHORTS = [
                'onek1k',
                'abf300'
                ]
CLOCK_TEST_COHORTS = [
                'zhang',
                'aida',
                'perez_sle'
                ]
# - palettes  
colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple
set2_colors = sns.color_palette("Set2", n_colors=len(ALL_DATASETS))
palette_datasets = {d: color for d, color in zip(ALL_DATASETS, set2_colors)}
palette_datasets_pretty = {surrogate_names[d]:color for d, color in palette_datasets.items()}
palette_regulation = {'Positive': '#56B4E9', 'Negative': 'lightcoral'}

palette_genders = {"Male": "#1f78b4", "Female": "#ff7f00", 'Both': '#999999'}
palette_trend = {'Inconsistent': 'gray', 'Increase in aging': '#E52B50', 'Decrease in aging': '#B0BF1A'}
cmap_trend = LinearSegmentedColormap.from_list(
            "aging_map", [palette_trend['Decrease in aging'], '#F0F0F0', palette_trend['Increase in aging']], N=10
        )
palette_trend_2 = OrderedDict([
    ('Increase in aging', '#E52B50'),
    ('Decrease in aging', '#B0BF1A'),
])
palette_disease_effect = OrderedDict([
    ('Increase in disease', '#A83279'),   # magenta-rose (reddish, but cooler tone)
    ('Decrease in disease', '#4CAF50'),   # leafy green (darker and more saturated)
])
palette_treatment = OrderedDict([
    ('Increase after treatment', '#FF7F0E'),   # bright orange (stays on warm side, but clearly distinct)
    ('Decrease after treatment', '#1E8449'),   # forest green (darker and more neutral)
])
CELL_TYPES = ['CD4T', 'CD8T', 'NK', 'B', 'MONO']
palette_cell_types = {name: color for name, color in zip(CELL_TYPES, ['#E69F00', '#56B4E9', '#F0E442', '#002266', '#998000'])}
# - mapping
mapping_major_2_minor = {
    'B': ['Naive_B', 'Memory_B'],
    'CD4T': ['Tcm_Naive_CD4', 'Tem_Effector_CD4', 'Treg'],
    'CD8T': ['Tem_Trm_CD8', 'Tem_Temra_CD8', 'Tcm_Naive_CD8', 'MAIT'],
    'MONO': ['NonClassic_MONO', 'Classic_MONO'],
    'NK': ['CD16_NK', 'NK']
 }
mapping_minor_2_major = {
    'Tcm_Naive_CD4': 'CD4T',
    'Tem_Effector_CD4': 'CD4T',
    
    'Tem_Trm_CD8': 'CD8T',
    'Tem_Temra_CD8': 'CD8T',
    'Tcm_Naive_CD8': 'CD8T',
    'MAIT': 'CD8T',
    
    'NK': 'NK',
    'CD16_NK': 'NK',
    'Classic_MONO': 'MONO',
    'NonClassic_MONO': 'MONO',
    
    'Naive_B': 'B',
    'Memory_B': 'B',
}
minor_cell_types = list(mapping_minor_2_major.keys())
# par_simulation = {
#         'simulation_iteration': 3,
#         'n_donors': 20,
#         'data_type': 'bulk',
#         'version': 'all_data',
#         'reg_type': 'ridge',
#         'feature_type': 'gene_expression',
#         'perturbation_mode': 'overexpression',
#         'tfs': None,
#     }

@dataclass
class ConditionConfig:
    """Configuration for a specific dataset analysis."""
    
    # Core identifiers
    name: str
    
    pseudobulk_group: Optional[List[str]] = None
    # Data columns
    condition_column: Optional[str] = None # Column name in obs: 'condition', 'perturbation', 'Max_WHO_Group'
    control_group: Optional[str] = None     # Control group name: 'healthy', 'PBS', 'Dimethyl Sulfoxide'
    
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
    
# Dataset configurations
DATASET_CONFIGS = {
    # ========== Population aging ==========
    "aida": ConditionConfig(
        name="aida",
        pseudobulk_group=['donor_id', 'cell_type', 'age']
    ), 
    'zhang': ConditionConfig(
        name="zhang",
        pseudobulk_group=['donor_id', 'cell_type', 'age']
    ),
    'onek1k': ConditionConfig(
        name="onek1k",
        pseudobulk_group=['donor_id', 'cell_type', 'age']
    ),
    'abf300': ConditionConfig(
        name="abf300",
        pseudobulk_group=['donor_id', 'cell_type', 'age']
    ),


    # ========== DISEASE DATASETS ==========
    "perez_sle": ConditionConfig(
        name="perez_sle",
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

        # Clock analysis settings
        clock_test_type='unpaired',
        pseudobulk_group=['donor_id', 'cell_type', 'age']
       
    ),
    
    # ========== PERTURBATION DATASETS ==========
    "op": ConditionConfig(
        name="op",
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
        clock_experiments='auto',  # Auto-generate from data: (control, treatment) for all treatments
        clock_mock_names=True,  # Mock compound names (keep top 1, rename others)
        clock_plot_config={
            'rejuvenating': {'figsize': (7.5, 3), 'margins': (0.05, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
            'aging': {'figsize': (5, 3), 'margins': (0.05, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
        },
        name_mapping = {},
        pseudobulk_group=['cell_type', 'plate_name', 'condition', 'well', 'donor_id']
    ),
    
    "CXCL9": ConditionConfig(
        name="CXCL9",
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
        pseudobulk_group=['cell_type', 'pool_id', 'condition', 'donor_id']
    ),
    
    "parsebioscience": ConditionConfig(
        name="parsebioscience",
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
    "soundlife": ConditionConfig(
            name="soundlife",
            comparison_mode='same',
            pseudobulk_group=['cell_type', 'donor_id', 'visitName'],
            name_mapping={'healthy': 'healthy', 'CMV':'healthy'},
        )
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
        Configuration object for the dataset
    
    Raises
    ------
    ValueError
        If dataset not found
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    
    config = DATASET_CONFIGS[dataset_name]
    
    return config
