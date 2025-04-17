import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
                    'european': 'European',
                    'asian': 'Asian',}


cell_types = ['CD4T', 'CD8T', 'NK', 'B', 'MONO']


# - datasets
datasets_e = ['data1', 'data7_allTPs_jalil', 'data12', 'SLE_European']
datasets_a = ['data13_Korean' , 'data13_Japanese', 'SLE_Asian']
datasets_all = datasets_e + datasets_a

# - palettes  
colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple
set2_colors = sns.color_palette("Set2", n_colors=len(datasets_all))
palette_datasets = {d: color for d, color in zip(datasets_all, set2_colors)}
palette_datasets_pretty = {surrogate_names[d]: color for d, color in zip(datasets_all, set2_colors)}
palette_datasets_pretty = {surrogate_names[d]:color for d, color in palette_datasets.items()}
palette_regulation = {'Positive': '#009E73', 'Negative': 'lightcoral'}
palette_cell_types = {name: color for name, color in zip(cell_types, ['#E69F00', '#56B4E9', '#F0E442', '#002266', '#998000'])}
palette_genders = {"Male": "#1f78b4", "Female": "#ff7f00", 'Both': '#999999'}
palette_trend = {'Inconsistent': 'gray', 'Increase in aging': '#E52B50', 'Decrease in aging': '#B0BF1A'}
cmap_trend = LinearSegmentedColormap.from_list(
            "aging_map", [palette_trend['Decrease in aging'], '#F0F0F0', palette_trend['Increase in aging']], N=10
        )
palette_trend_2 = {key: palette_trend[key] for key in ['Increase in aging', 'Decrease in aging']}

palette_disease_effect = {'Increase in disease': 'Orange', 'Decrease in disease': '#0072B2'}


palette_sub_types = {
    'Tem_Trm_CD8': '#E69F00',    
    'Tem_Temra_CD8': '#56B4E9',   
    'Tcm_Naive_CD8': '#009E73',   
    'Tcm_Naive_CD4': '#F0E442', 
    'Tem_Temra_CD4': '#0072B2',   
    'Tem_Effector_CD4': '#D55E00',    
    'Treg': '#CC79A7',    
    'MAIT': 'green',   
    'CD8a/a': 'grey',
    'Classic_MONO': 'lightcoral', 
    'NonClassic_MONO': '#ff7f00',     
}
# - mapping
mapping_major_2_minor = {
    'B': ['Naive_B', 'Aged_B', 'Memory_B', 'Plasma_B', 'Plasmablasts_B'],
    'CD4T': ['Tcm_Naive_CD4', 'Tem_Effector_CD4', 'Treg'],
    'CD8T': ['Tem_Trm_CD8', 'Tem_Temra_CD8', 'Tcm_Naive_CD8', 'MAIT'],
    'MONO': ['NonClassic_MONO', 'Classic_MONO'],
    'NK': ['CD16_NK', 'NK']
 }

mapping_minor_2_major = {
    'Naive_B': 'B',
    'Aged_B': 'B',
    'Memory_B': 'B',
    'Plasma_B': 'B',
    'Plasmablasts_B': 'B',
    'Tcm_Naive_CD4': 'CD4T',
    'Tem_Effector_CD4': 'CD4T',
    'Treg': 'CD4T',
    'Tem_Trm_CD8': 'CD8T',
    'Tem_Temra_CD8': 'CD8T',
    'Tcm_Naive_CD8': 'CD8T',
    'MAIT': 'CD8T',
    'NonClassic_MONO': 'MONO',
    'Classic_MONO': 'MONO',
    'CD16_NK': 'NK',
    'NK': 'NK'
}

