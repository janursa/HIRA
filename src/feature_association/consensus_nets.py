from hiara import MAJOR_CTS, retrieve_net_consensus
for cell_type in MAJOR_CTS:
    retrieve_net_consensus(cell_type=cell_type, promotor_only=False, force=True)
    retrieve_net_consensus(cell_type=cell_type, promotor_only=True, force=True)

    