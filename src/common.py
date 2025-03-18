import seaborn as sns



colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple

major_cell_types = ['B cells', 'CD4+ T cells', 'CD8+ T cells', 'Myeloid cells', 'NK cells']
cell_type_palette = {name: color for name, color in zip(major_cell_types, colors_blind[:len(major_cell_types)])}

surrogate_names = {'batch_1':'Batch 1', 'batch_2':'Batch 2', 'all_batches':'All batches', 
                    '34-':'35 below', '35_44':'35-45', '45_54':'45-55', '55_64':'55-65', '65_75':'65-75',
                    'data1_male': 'External validation', 'pbmc_ageing_downsample_male': 'Downsampled data',
                    'data1': 'Dataset 1', 'data7_allTPs_jalil': 'Dataset 2'}

datasets = ['data1', 'data7_allTPs_jalil']

# - color palette
color_palette = sns.color_palette("tab10", len(datasets))  
palette_datasets = {label: color for label, color in zip(datasets, color_palette)}
cell_types = ['CD4T', 'CD8T', 'NK', 'B', 'MONO']