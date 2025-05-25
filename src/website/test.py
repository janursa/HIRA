
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import OrderedDict
import os
import anndata as ad


# --------- variables 
website_dir = '/vol/projects/jnourisa/website/'
website_input_dir = f'{website_dir}/input'
website_tmp_dir = f'{website_dir}/tmp'


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
                    'NK': 'NK',}


# - datasets
datasets_e = ['data1', 'data7_allTPs_jalil', 'data12', 'SLE_European']
datasets_a = ['data13_Korean' , 'data13_Japanese']
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
    ('Decrease in aging', '#B0BF1A'),
    ('Increase in aging', '#E52B50'),
])
from collections import OrderedDict

palette_disease_effect = OrderedDict([
    ('Decrease in disease', '#0072B2'),
    ('Increase in disease', 'Orange'),
])

palette_treatment = OrderedDict([
    ('Decrease after treatment', '#0072B2'),
    ('Increase after treatment', 'Orange'),
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




# - ----------------------- functions

    
def bin_feature_values(adata):
    # - bin 
    expr = adata.to_df()
    expr = expr.merge(adata.obs[['age']], left_index=True, right_index=True, how='left').set_index('age')
    expr.sort_index(inplace=True)
    expr['age_bin'] = (expr.index.astype(int) // 5) * 5
    expr_mean = expr.groupby('age_bin').mean().T
    # Normalize expression
    min_vals = expr_mean.min(axis=1)
    max_vals = expr_mean.max(axis=1)
    expr_mean = (expr_mean.sub(min_vals, axis=0)).div(max_vals - min_vals, axis=0)
    return expr_mean
def retrieve_feature_data(dataset, cell_type, type, feature_type='tf_activity'):
    cell_type_major = mapping_minor_2_major.get(cell_type, cell_type)
    file_path = f'{website_input_dir}/{feature_type}/{dataset}_{cell_type_major}_{type}.h5ad'
    if os.path.exists(file_path) == False:
        raise ValueError(f'File {file_path} does not exist')
    adata = ad.read_h5ad(file_path)
    
    return adata
def heatplot_age_trend(mean_expr, cmap="viridis", cbar_title="Gene expression", y_label="Genes", figsize=(2.5, 3), ax=None, show_cbar=True, shrink=.7):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    # Plot heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mean_expr, cmap=cmap, cbar=show_cbar, 
                cbar_kws={
                    "shrink": shrink,
                    "aspect": 10,       # Lower values = thicker colorbar (default is ~20)
                    "fraction": 0.1    # Controls the width space the cbar takes in the figure
                },
                 ax=ax)
    ax.set_yticks(np.arange(mean_expr.shape[0]) + 0.5)
    ax.set_yticklabels(mean_expr.index, rotation=0)

    # Adjust colorbar
    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(cbar_title, rotation=90, labelpad=5)

    # Labels and formatting
    ax.set_xlabel("Age")
    # ax.set_ylabel(y_label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

def plot_feature_values_per_datasets(cell_type, features, type, datasets, feature_type='gene_expression', age_limit=[20, 75], cluster=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_datasets = len(datasets)
    n_features = len(features)
    fig, axes = plt.subplots(1, n_datasets, figsize=(n_datasets*3, .2*n_features+1), sharey=False)
    for i, (dataset) in enumerate(datasets):
        adata = retrieve_feature_data(dataset, cell_type, type, feature_type=feature_type)
        adata = adata[:, adata.var_names.isin(features)]
        
        if age_limit is not None:
            adata = adata[(adata.obs['age'] < age_limit[1]) & (adata.obs['age'] > age_limit[0])]
        # - plot target gene expression trend    
        ax = axes[i]
        mean_expr = bin_feature_values(adata)
        if cluster:
            if i == 0:
                from sklearn.cluster import KMeans
                if len(mean_expr) > 2:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(mean_expr)
                    mean_expr['cluster'] = clusters
                    mean_expr = mean_expr.sort_values('cluster').drop(columns=['cluster'])
                ordered_targets = mean_expr.index
            else:
                mean_expr = mean_expr.reindex(ordered_targets)
        else:
            mean_expr = mean_expr.reindex(features)
        if i == 0:
            show_cbar = True
        else:
            show_cbar = False
        
        heatplot_age_trend(mean_expr, cmap='viridis' if feature_type=='gene_expression' else 'magma', 
                            cbar_title="Gene expression" if feature_type=='gene_expression' else "TF activity", 
                            y_label="Genes" if feature_type=='gene_expression' else "TFs", 
                            ax=ax, 
                            show_cbar=show_cbar)
        if i != 0:
            # ax.set_yticklabels([])
            ax.set_ylabel('')
        ax.set_title(surrogate_names[dataset], pad=10, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.suptitle(cell_type, fontsize=12, fontweight='bold', y=1.05)

    return fig



if __name__ == '__main__':
    os.makedirs(website_tmp_dir, exist_ok=True)
    features=['FOXO1', 'TCF7', 'BACH2', 'FOS', 'FOSB']
    type = 'bulk'
    cell_type='CD8T'
    datasets = ['data7_allTPs_jalil', 'data13_Korean']
    fig = plot_feature_values_per_datasets(cell_type, features, type, datasets, feature_type='tf_activity', age_limit=[20, 75], cluster=False)
    fig.savefig(f'{website_tmp_dir}/test.png', dpi=300, bbox_inches='tight')
