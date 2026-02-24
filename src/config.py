
"""
Configuration for condition-based analyses (disease and perturbation).
This module centralizes all dataset-specific configurations to eliminate
code duplication between disease and perturbation analyses.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from collections import OrderedDict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict
import warnings
import os
import platform
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*anndata.*", category=FutureWarning)


CLOCK_V = 'V1'
USE_LOCAL_CLOCK = True  # If True, use clocks saved in CLOCKS_DIR;

SUB_CT_LABEL = 'Sub_CT'
MAJOR_CT_LABEL = 'Major_CT'
FEATURE_TYPES = ['tf_activity', 'gene_expression', 'gene_score', 'aging_hallmarks', 'tfa_peg', 'ct_freq', 'ct_pol_dist', 'cc_interaction']
FEATURE_DEF = {
    'tf_activity': 'TF activity',
    'gene_expression': 'Gene expression',
    'gene_score': 'Gene score',
    'aging_hallmarks': 'Aging hallmarks',
    'tfa_peg': 'TF activity association with progenitor-effector gradient',
    'ct_freq': 'Cell type composition (frequency)',
    'ct_pol_dist': 'Naive/effector polarization distance calculated using TF activity features',
    'cc_interaction': 'Cell-cell communication features from sub cell type analysis',
}
DATA_TYPES = ['sc', 'bulk', 'bulk_minor', 'metacell']
# - mapping
mapping_major_2_minor = {
    'B': ['Naive_B', 'Memory_B'],
    'CD4T': ['Tcm_Naive_CD4', 'Tem_Effector_CD4'],
    'CD8T': ['Tcm_Naive_CD8', 'Tem_Trm_CD8', 'Tem_Temra_CD8', 'MAIT'],
    'MONO': [
            'NonClassic_MONO', 
             'Classic_MONO'
             ],
    'NK': ['CD16_NK']
 }

mapping_minor_2_major = {
    'Tcm_Naive_CD4': 'CD4T',
    'Tem_Effector_CD4': 'CD4T',
    
    'Tcm_Naive_CD8': 'CD8T',
    'Tem_Trm_CD8': 'CD8T',
    'Tem_Temra_CD8': 'CD8T',
    'MAIT': 'CD8T',
    
    'CD16_NK': 'NK',
    'Classic_MONO': 'MONO',
    'NonClassic_MONO': 'MONO',
    
    'Naive_B': 'B',
    'Memory_B': 'B',
}
SUB_CTS = list(mapping_minor_2_major.keys())
MAJOR_CTS = ['CD4T', 'CD8T', 'NK', 'B', 'MONO']
CONFIG_FA = {
    'tfa_major_b': {'data_type': 'bulk', 'feature_type': 'tf_activity', 'granularity': MAJOR_CT_LABEL}, 
    'tfa_sub_b': {'data_type': 'bulk_minor', 'feature_type': 'tf_activity', 'granularity': SUB_CT_LABEL},
    'sub_tf_markers': {'data_type': 'bulk_minor', 'feature_type': 'tf_activity', 'granularity': SUB_CT_LABEL, 
                        'trend_labels': ['Higher in this group', 'Lower in this group'],
                        'cell_types': ['CD4T', 'CD8T', 'B', 'MONO']},
    'ge_major_b': {'data_type': 'bulk', 'feature_type': 'gene_expression', 'granularity': MAJOR_CT_LABEL},
    'ge_sub_b': {'data_type': 'bulk_minor', 'feature_type': 'gene_expression', 'granularity': SUB_CT_LABEL},
    'tfa_peg': {'data_type': 'sc', 'feature_type': 'tfa_peg', 'granularity': MAJOR_CT_LABEL, 'cell_types': ['CD8T']},
    'ct_freq': {'data_type': 'sc', 'feature_type': 'ct_freq', 'granularity': SUB_CT_LABEL, 'cell_types': ['CD8T', 'CD4T']},
    'ct_pol_dist': {'data_type': 'bulk_minor', 'feature_type':'ct_pol_dist', 'granularity': MAJOR_CT_LABEL, 'cell_types': ['CD4T', 'CD8T']},
    'ccc_sub_b': {'data_type': 'sc', 'feature_type': 'cc_interaction', 'granularity': SUB_CT_LABEL, 'cell_types': ['all']},
    'ccc_major_b': {'data_type': 'sc', 'feature_type': 'cc_interaction', 'granularity': MAJOR_CT_LABEL, 'cell_types': ['all']},
}
ANALYSIS_DEF = {
    'tfa_major_b': 'TF activity features from major cell type analysis',
    'tfa_sub_b': 'TF activity features from sub cell type analysis',
    'sub_tf_markers': 'TF markers for each sub cell types',
    'ge_major_b': 'Gene expression features from major cell type analysis',
    'ge_sub_b': 'Gene expression features from sub cell type analysis',
    'tfa_peg': 'TF activity association with progenitor-effector gradient',
    'ct_freq': 'Cell type composition (frequency)',
    'ct_pol_dist': 'Naive/effector polarization distance calculated using TF activity features',
    'ccc_sub_b': 'Cell-cell communication features from sub cell type analysis',
}
def get_config_fa(analysis_name):
    """Get feature analysis configuration by name."""
    if analysis_name not in CONFIG_FA:
        raise ValueError(f"Analysis '{analysis_name}' not found in configuration. Available: {list(CONFIG_FA.keys())}")
    return CONFIG_FA[analysis_name]

def get_available_fa_analyses():
    """Get list of available feature analysis configurations."""
    return list(CONFIG_FA.keys())

DISCOVERY_COHORTS = ['aida', 'perez_sle', 'onek1k', 'abf300'] #
# DISCOVERY_COHORTS = ['perez_sle', 'aida']
AGING_COHORTS = ['onek1k', 'abf300', 'aida', 'perez_sle', 'soundlife', 'zhang']
ALL_DATASETS = ['onek1k', 'abf300', 'aida', 'perez_sle', 'CXCL9', 'op', 'parsebioscience', 'soundlife', 'zhang']
CLOCK_TRAINING_COHORTS = [
                'onek1k',
                'abf300',
                # 'aida',
                # 'zhang',
                # 'soundlife'
                ]
CLOCK_TEST_COHORTS = [
                'aida',
                'perez_sle',
                'zhang'
                # 'onek1k',
                # 'abf300'
                ]
NET_WEIGHT_THRESHOLD = None  # 0.05 # Minimum absolute weight for edges in GRN 
NET_MAX_SIZE = 100_000
CLOCK_CV_SCORING = 'spearman'  # 'r2' or 'spearman'
TUNE_CLOCK = True
META_MIN_COHORT = 2
CONSENSUS_MIN_DEGREE = 2 
CORR_THRESHOLD = 0.1 # minimum absolute correlation for feature association with age
TF_MIN_TARGET = 5

DATASET_NAME_MAPPING = {
    "data1": "onek1k",
    "data7_allTPs_jalil": "abf300",
    "data12": "zhang",
    "data13": "aida",
    "SLE": "perez_sle"
}

if platform.system() == 'Linux':
    HIARA_DIR = '/home/jnourisa/projs/ongoing/hiara/'
    base_dir = '/vol/projects/jnourisa/hiara/'
    TASK_GRN_BENCHMARK_DIR = '/home/jnourisa/projs/ongoing/task_grn_inference/'
else:
    HIARA_DIR = '/Users/jno24/Documents/projs/ongoing/hiara/'
    base_dir = '/Users/jno24/Documents/projs/ongoing/hiara/base_folder'
    TASK_GRN_BENCHMARK_DIR = '/Users/jno24/Documents/projs/ongoing/task_grn_inference/'

DATA_DIR = f'{base_dir}/datasets/'
PRIOR_DIR = f'{base_dir}/prior/'

OUTPUT_DIR = f'{base_dir}/output/'
GRNS_DIR = f'{OUTPUT_DIR}/grns'
    
FEATURES_DIR = f'{OUTPUT_DIR}/features/'
CLOCKS_DIR = f"{OUTPUT_DIR}/clock/"
PLOTS_DIR = f"{OUTPUT_DIR}/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRNS_DIR, exist_ok=True)
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
                    'healthy': 'Healthy',
                    
                    'RPMI + ruxolitinib': 'RPMI + Ruxolitinib',
                    'LPS + ruxolitinib': 'LPS + Ruxolitinib',
                    'LPS': 'LPS',
                    'RPMI': 'RPMI',
                    'Dimethyl Sulfoxide': 'DMSO',

                    'tf_activity': 'TF activity',
                    'tfa_peg': 'PEG',
                    'ct_pol_dist': 'Naive/Effector transcriptional distance',

                    'Tcm_Naive_CD4': 'Tcm/Naive CD4',
                    'Tem_Effector_CD4': 'Tem/Effector CD4',
                    'Tcm_Naive_CD8': 'Tcm/Naive CD8',
                    'Tem_Trm_CD8': 'Tem/Trm CD8',
                    'Tem_Temra_CD8': 'Tem/Temra CD8',
                    'MAIT': 'MAIT',
                    'CD16_NK': 'CD16 NK',
                    'NonClassic_MONO': 'Non-Classic Mono',
                    'Classic_MONO': 'Classic Mono',
                    'Naive_B': 'Naive B',
                    'Memory_B': 'Memory B',

                

                    }
surrogate_names_reverse = {v: k for k, v in surrogate_names.items()}

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

palette_major_cts = {name: color for name, color in zip(MAJOR_CTS, ['#E69F00', '#56B4E9', '#F0E442', '#002266', '#998000'])}

# MAJOR_CTS = ['CD4T']
palette_sub_cts = {
    # CD4T: naive (green) -> effector (red)
    'Tcm_Naive_CD4': "#056608",      # Green (naive)
    'Tem_Effector_CD4': "#C5F436",   # Red (effector)
    
    # CD8T: naive (green) -> differentiated (red)
    'Tcm_Naive_CD8': '#4CAF50',      # Green (naive)
    'Tem_Trm_CD8': '#FF9800',        # Orange (intermediate)
    'Tem_Temra_CD8': '#FF5722',      # Deep Orange (more differentiated)
    'MAIT': "#5A0303",               # Dark Red (highly differentiated)
    
    # NK: less activated (green) -> more activated (red)
    'CD16_NK': "#5D50EF",            # Light Red
    
    # MONO: classical (green) -> non-classical (red)
    'Classic_MONO': "#D4BA35",       # Light Green
    'NonClassic_MONO': "#C5A637",    # Light Red
    
    # B cells: naive (green) -> memory (red)
    'Naive_B': "#AD1CE6",            # Green (naive)
    'Memory_B': "#BA824E",           # Red (memory)
}

@dataclass
class ConditionConfig:
    """Configuration for a specific dataset analysis."""
    
    # Core identifiers
    name: str
    
    bulk_group: Optional[List[str]] = None
    # Data columns
    condition_column: Optional[str] = None # Column name in obs: 'condition', 'perturbation', 'Max_WHO_Group'
    
    # Treatments to compare (can be 'all' for auto-detection)
    treatment_groups: Optional[List[str]] = None
    
    # Statistical parameters
    test_type: str = 'unpaired'  # 'unpaired', 'mixed-effect', 'paired'
    
    # Mixed-effects model parameters (used when test_type='mixed-effect')
    mixed_effects_formula: Optional[str] = None  # R-style formula, e.g., "feature_values ~ condition"
    mixed_effects_group: Optional[str] = None    # Random effects grouping variable, e.g., 'donor_id'
    
    # Data filtering (for splitting datasets into subsets)
    data_filter: Optional[Dict[str, any]] = None  # E.g., {'vaccinated': True, 'followup_day': [0, 7, 90]}
    
    
    # Special handling
    control_mapping: Optional[Dict[str, str]] = None  # For datasets with multiple controls
    name_mapping: Optional[Dict[str, str]] = field(default_factory=dict)    # Rename conditions for display
    condition_mapping: Optional[Dict[str, str]] = None  # Transform actual condition values in data (applied when loading)
    
    # Display
    display_name: Optional[str] = None
    
    # Plotting control
    target_treatments: Optional[List[str]] = None  # Which conditions to plot in overlap analysis
    
    
    # ========== CLOCK ANALYSIS SPECIFIC ==========
    # Statistical testing for clock predictions
    clock_test_type: Optional[str] = None  # 'paired', 'unpaired', 'mixed-effect'
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
        bulk_group=['donor_id', 'age']
    ), 
    'zhang': ConditionConfig(
        name="zhang",
        bulk_group=['donor_id', 'age']
    ),
    'onek1k': ConditionConfig(
        name="onek1k",
        bulk_group=['donor_id', 'age']
    ),
    'abf300': ConditionConfig(
        name="abf300",
        bulk_group=['donor_id', 'age']
    ),


    # ========== DISEASE DATASETS ==========
    "perez_sle": ConditionConfig(
        name="perez_sle",
        condition_column='condition',
        control_mapping='healthy',
        treatment_groups=['healthy', 'SLE'],
        test_type='unpaired',
        display_name='SLE',
        # target_conditions=None,
        name_mapping={
            'normal': 'healthy',
            'systemic lupus erythematosus': 'SLE',
            'SLE vs healthy': 'SLE'
        },

        # Clock analysis settings
        clock_test_type='unpaired',
        bulk_group=['donor_id', 'age']
       
    ),
    
    # ========== PERTURBATION DATASETS ==========
    "op": ConditionConfig(
        name="op",
        condition_column='condition',  # Will auto-detect 'perturbation' if needed
        control_mapping='DMSO',
        treatment_groups=['DMSO', 'Ruxolitinib'],  # Auto-detect all drugs
        test_type='mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        mixed_effects_group='donor_id',
        display_name='OP Compounds',
        # target_conditions=['Ruxolitinib'],
        # Clock analysis settings
        clock_test_type='mixed-effect',
        clock_group_key='donor_id',
        clock_pvalue_correction='corrected',
        clock_experiments='all',  # Auto-generate from data: (control, treatment) for all treatments
        clock_mock_names=True,  # Mock compound names (keep top 1, rename others)
        # clock_plot_config={
        #     'rejuvenating': {'figsize': (7.5, 3), 'margins': (0.05, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
        #     'aging': {'figsize': (5, 3), 'margins': (0.05, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1)},
        # },
        name_mapping = {
            'Dimethyl Sulfoxide': 'DMSO',
            'Ruxolitinib vs DMSO': 'Ruxolitinib',
        },
        bulk_group=['condition', 'donor_id']
    ),
    
    "CXCL9": ConditionConfig(
        name="CXCL9",
        condition_column='condition',
        treatment_groups=[
            'RPMI',
            'RPMI + ruxolitinib',
            'LPS + ruxolitinib',
            'LPS'
        ],
        test_type= 'mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        mixed_effects_group='donor_id',
        display_name='Ruxolitinib',
        control_mapping={
            'RPMI': 'RPMI',
            'RPMI + ruxolitinib': 'RPMI',
            'LPS + ruxolitinib': 'LPS',
            'LPS': 'RPMI'
        },
        name_mapping=OrderedDict({
            '24 h RPMI': 'RPMI',
            '24 h LPS': 'LPS',
            '24 h RPMI + ruxolitinib': 'RPMI + ruxolitinib',
            '24 h LPS + ruxolitinib': 'LPS + ruxolitinib',
            'RPMI + ruxolitinib vs RPMI': 'Ruxolitinib (ctr: RPMI)',
            'LPS + ruxolitinib vs LPS': 'Ruxolitinib (ctr: LPS)',
            'LPS vs RPMI': 'LPS (ctr: RPMI)'
        }),
        
        # target_conditions=['Ruxolitinib (ctr: RPMI)', 'Ruxolitinib (ctr: LPS)'],
        # Clock analysis settings
        clock_test_type='mixed-effect',
        # clock_group_key='donor_id',
        clock_pvalue_correction='raw',
        clock_experiments=[
            ('RPMI', 'LPS'),
            ('RPMI', 'RPMI + ruxolitinib'),
            ('LPS', 'LPS + ruxolitinib'),
        ],
        clock_pretty_names={
            'LPS': 'LPS \n (ctr: RPMI)',
            'LPS + ruxolitinib': 'Ruxolitinib \n (ctr: LPS)',
            'RPMI': 'RPMI',
            'RPMI + ruxolitinib': 'Ruxolitinib \n (ctr: RPMI)',
        },
        # clock_plot_config={
        #     'rejuvenating': {'figsize': (4, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        #     'aging': {'figsize': (7, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        # },
        bulk_group=['condition', 'donor_id']
    ),
    
    "parsebioscience": ConditionConfig(
        name="parsebioscience",
        condition_column='condition',  
        control_mapping='PBS',
        treatment_groups=['PBS', 'IL-10'], 
        test_type= 'mixed-effect',#'mixed-effect',
        mixed_effects_formula='feature_values ~ condition',
        bulk_group=['condition', 'donor_id', 'well'],
        mixed_effects_group='donor_id',
        display_name='Cytokines',
        # Clock analysis settings
        name_mapping={
            'IL-10 vs PBS': 'IL-10',
        },
        clock_test_type='mixed-effect',
        clock_group_key='donor_id',
        clock_pvalue_correction='corrected',
        clock_experiments='all',  # Auto-generate from data
        # clock_plot_config={
        #     'CD4T': {
        #         'rejuvenating': {'figsize': (5, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        #         'aging': {'figsize': (10, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        #     },
        #     'default': {
        #         'rejuvenating': {'figsize': (7, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        #         'aging': {'figsize': (10, 3), 'margins': (0.12, 0.2), 'ha': 'right', 'bbox_to_anchor': (1, 1.2)},
        #     },
        # },
    ),
    
    # ========== AGING DATASETS (Longitudinal) ==========
    "soundlife": ConditionConfig(
            name="soundlife",
            bulk_group=['donor_id', 'visitName'],
        )
}

def get_config(dataset: str) -> ConditionConfig:
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
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset}' not found in configurations.")

    config = DATASET_CONFIGS[dataset]
    
    return config
