import pandas as pd
import random
import pandas as pd

import numpy as np
from hira.src.insilico_perturbation.optimization.helper import obtimize_function
from hira.src.insilico_perturbation.optimization.helper import run_greedy_tf_optimization_parallel
from hira.src.config import OUTPUT_DIR

if __name__ == "__main__":
    tfs = np.loadtxt(f"{OUTPUT_DIR}/candidate_tfs.txt", dtype=str)
    n_jobs = 20

    selected_tfs, df_results = run_greedy_tf_optimization_parallel(
        TF_all=list(tfs),
        obtimize_function=obtimize_function,
        max_tfs_n=5,
        n_jobs=n_jobs
    )
    df_results.to_csv(f"{OUTPUT_DIR}/optimization_results.csv", index=False)