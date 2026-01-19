from hiara import CELL_TYPES, retrieve_net_consensus
for cell_type in CELL_TYPES:
    retrieve_net_consensus(cell_type=cell_type, promotor_only=False, force=True)
    retrieve_net_consensus(cell_type=cell_type, promotor_only=True, force=True)

    