
from hiara.src.utils.util import retrieve_adata, MAJOR_CT_LABEL, SUB_CT_LABEL
from hiara.src.config import MAJOR_CTS, SUB_CTS


MODEL_ID = "gpt-4o"
interventions_2_dataset = {
    'drug': 'op',
    'cytokine': 'parsebioscience',
}
interventions = {
    'drug': retrieve_adata(dataset=interventions_2_dataset['drug'], only_obs=True)['condition'].unique().tolist(),
    'cytokine': retrieve_adata(dataset=interventions_2_dataset['cytokine'], only_obs=True)['condition'].unique().tolist(),
}
cell_types = {
    SUB_CT_LABEL: SUB_CTS,
    MAJOR_CT_LABEL: MAJOR_CTS,
}

def get_analysis_defs():
    pass
def get_analysis_def():
    pass