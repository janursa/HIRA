"""Rebuild the consensus GRNs for every major cell type.

Rerun after changing NET_SKELETON / NET_MAX_SIZE / CONSENSUS_MIN_DEGREE in config.py.

Usage: python src/feature_association/consensus_nets.py
Writes: GRNS_DIR/consensus_net_<cell_type>.csv
"""
from hira import MAJOR_CTS, retrieve_net_consensus
for cell_type in MAJOR_CTS:
    retrieve_net_consensus(cell_type=cell_type, force=True)
