import seaborn as sns



colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple

major_cell_types = ['B', 'CD4T', 'CD8T', 'MONO', 'NK']
palette_cell_types = {name: color for name, color in zip(major_cell_types, colors_blind[:len(major_cell_types)])}

surrogate_names = {'batch_1':'Batch 1', 'batch_2':'Batch 2', 'all_batches':'All batches', 
                    '34-':'35 below', '35_44':'35-45', '45_54':'45-55', '55_64':'55-65', '65_75':'65-75',
                    'data1': 'Cohort 1', 'data7_allTPs_jalil': 'Cohort 2',
                    
                    'data13': 'Cohort 3', 
                    'data12': 'Cohort 4',
                    'SLE': 'SLE Cohort',
                    'normal': 'Control',
                    'systemic lupus erythematosus': 'SLE'}

datasets = ['data1', 'data7_allTPs_jalil']
v_datasets = ['data13', 'data12']
datasets_disease = ['SLE']
datasets_healthy = datasets + v_datasets 
datasets_all = datasets_healthy + datasets_disease
# - color palette

# palette_datasets = {'Discovery Cohort 1': '#56B4E9', 'Discovery Cohort 2':'#D55E00', 
#                     'Validation Cohort 1':'#009E73', 'Validation Cohort 2':'#F0E442',
#                     'SLE Cohort':'#CC79A7'}
palette_datasets = {d:color for d, color in zip(datasets_all, ['#56B4E9', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#0072B2'])}
palette_datasets_pretty = {surrogate_names[d]:color for d, color in palette_datasets.items()}

palette_regulation = {'Positive': '#009E73', 'Negative': 'lightcoral'}
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


palette_gender = {"Male": "#1f78b4", "Female": "#ff7f00"}
palette_trend = {'Inconsistent': 'gray', 'Increase in aging': 'red', 'Decrease in aging': 'green'}
palette_disease_effect = {'Increase in disease': 'Orange', 'Decrease in disease': '#0072B2'}

palette_twoagegroups = {name:color for name, color in zip(["Below 50", "Above 50"], sns.color_palette("Set2", 2))}
palette_trend_2 = {key: palette_trend[key] for key in ['Increase in aging', 'Decrease in aging']}
cell_types = ['CD4T', 'CD8T', 'NK', 'B', 'MONO']

# mapping_major_2_minor = {
#         'MONO': ['Classic_MONO', 'NonClassic_MONO'],
#         'CD8T': ['MAIT', 'Tcm_Naive_CD8', 'Tem_Temra_CD8', 'Tem_Trm_CD8'],
#         'CD4T': ['Tcm_Naive_CD4', 'Tem_Effector_CD4', 'Treg'],
#         'B': ['Memory_B', 'Naive_B'],
#         'NK': ['CD16_NK', 'NK']
#     }

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

