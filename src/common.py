import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import OrderedDict
import warnings
import os
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*anndata.*", category=FutureWarning)

import platform

if platform.system() == 'Linux':
    base_dir = '/vol/projects/jnourisa/'
    task_grn_benchmark_dir = '/home/jnourisa/projs/ongoing/task_grn_inference/'
else:
    base_dir = '/Users/jno24/Documents/projs/ongoing/ciim/base_folder'
    task_grn_benchmark_dir = '/Users/jno24/Documents/projs/ongoing/task_grn_inference/'
save_dir = f'{base_dir}/output/'
clock_save_dir = f"{save_dir}/clock/"
plots_dir = f"{save_dir}/plots/"
prior_dir = f"{base_dir}/prior/"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(clock_save_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(f"{plots_dir}/insilico_perturbation/", exist_ok=True)

surrogate_names = {'batch_1':'Batch 1', 'batch_2':'Batch 2', 'all_batches':'All batches', 
                    '34-':'35 below', '35_44':'35-45', '45_54':'45-55', '55_64':'55-65', '65_75':'65-75',

                    'data1': 'C1: European', 'data7_allTPs_jalil': 'C2: European', 'data12': 'C3: European',
                    'data13_Korean': 'C4: Korean', 'data13_Japanese': 'C4: Japanese',
                    
                    'SLE_Asian': 'C5: Asian',
                    'SLE_Asian_normal': 'C5: Asian Healthy',
                    'SLE_Asian_systemic lupus erythematosus': 'C5: Asian SLE',
                    'SLE_European': 'C5: European',
                    'SLE_European_normal': 'C5: European Healthy',
                    'SLE_European_systemic lupus erythematosus': 'C5: European SLE',
                    'Covid_50MHH': 'C6: Covid',
                    'normal': 'Healthy',
                    'systemic lupus erythematosus': 'SLE',
                    'mild': 'Mild',
                    'severe': 'Severe',
                    'european': 'European',
                    'asian': 'Asian',
                    'collectri': 'CollecTRI',
                    'both': 'European & Asian',
                    'Naive_B': 'Naive B',
                    'Tem_Trm_CD8': 'Tem/Trm CD8T',
                    'Tem_Temra_CD8': 'Tem/Temra CD8T',
                    'Tcm_Naive_CD8': 'Tcm/Naive CD8T',
                    'Tcm_Naive_CD4': 'Tcm/Naive CD4T',
                    'Tem_Temra_CD4': 'Tem/Temra CD4T',
                    'Tem_Effector_CD4': 'Tem/Effector CD4T',
                    'Treg': 'Treg',
                    'MAIT': 'MAIT',
                    'Classic_MONO': 'Classic Mono',
                    'NonClassic_MONO': 'Non-Classic Mono',
                    'Naive_B': 'Naive B',
                    'Memory_B': 'Memory B',
                    'CD16_NK': 'CD16+ NK',
                    'NK': 'NK',
                    'Myc Targets V1': 'MYC',
                    'mTORC1 Signaling': 'mTORC1'}
surrogate_names_reverse = {v: k for k, v in surrogate_names.items()}

# - datasets
datasets_e = ['data1', 'data7_allTPs_jalil', 'data12', 'SLE_European']
datasets_a = ['data13_Korean' , 'data13_Japanese']
datasets_all = datasets_e + datasets_a

# aging_clock_train_datasets = ['data1', 'data7_allTPs_jalil', 'SLE_European', 'data13_Japanese', 'data12', 'data13_Korean']
aging_clock_train_datasets = ['data1', 'data7_allTPs_jalil', 'data12', 'data13_Japanese']
datasets_disease = ['SLE_European', 'Covid_50MHH']
datasets_drug_perturbation = ['op', 'CXCL9']

# - palettes  
colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple
set2_colors = sns.color_palette("Set2", n_colors=len(datasets_all)+2)
palette_datasets = {d: color for d, color in zip(datasets_all+['collectri', 'Covid_50MHH'], set2_colors)}
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
cell_types = ['CD4T', 'CD8T', 'NK', 'B', 'MONO']
palette_cell_types = {name: color for name, color in zip(cell_types, ['#E69F00', '#56B4E9', '#F0E442', '#002266', '#998000'])}

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

if True: # define palette for minor cell types
    from matplotlib.colors import to_rgb, to_hex
    import colorsys
    # Function to create a set of distinct but related colors from a base color
    def generate_color_variants(base_color, n_variants):
        base_rgb = to_rgb(base_color)
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        variants = []
        lightness_factors = [0.8 + 0.2 * i for i in range(n_variants)]
        for factor in lightness_factors:
            new_l = max(0, min(1, l * factor))
            new_rgb = colorsys.hls_to_rgb(h, new_l, s)
            variants.append(to_hex(new_rgb))
        return variants

    # Count how many minor types per major
    from collections import defaultdict

    grouped_minors = defaultdict(list)
    for minor, major in mapping_minor_2_major.items():
        grouped_minors[major].append(minor)

    # Assign variant colors
    palette_minor_types = {}
    for major, minors in grouped_minors.items():
        variants = generate_color_variants(palette_cell_types[major], len(minors))
        for minor, color in zip(minors, variants):
            palette_minor_types[minor] = color

palette_minor_types_pretty = {surrogate_names[minor]: color for minor, color in palette_minor_types.items()}


par_simulation = {
        'simulation_iteration': 3,
        'n_donors': 20,
        'data_type': 'bulk',
        'version': 'all_data',
        'reg_type': 'ridge',
        'feature_type': 'gene_expression',
        'perturbation_mode': 'overexpression',
        'tfs': None,
    }