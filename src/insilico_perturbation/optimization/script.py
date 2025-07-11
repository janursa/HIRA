import pandas as pd
import random
import pandas as pd

import numpy as np
from ciim.src.insilico_perturbation.optimization.helper import obtimize_function
from ciim.src.insilico_perturbation.optimization.helper import run_greedy_tf_optimization_parallel
from ciim.src.common import save_dir

if __name__ == "__main__":
    tfs = np.loadtxt(f"{save_dir}/candidate_tfs.txt", dtype=str)
    n_jobs = 20


    selected_tfs, df_results = run_greedy_tf_optimization_parallel(
        TF_all=list(tfs),
        obtimize_function=obtimize_function,
        max_tfs_n=5,
        constraint_threshold=1.0,
        n_jobs=n_jobs
    )
    df_results.to_csv(f"{save_dir}/optimization_results.csv", index=False)