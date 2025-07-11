

def run_greedy_tf_optimization_parallel(
    TF_all,
    obtimize_function,
    max_tfs_n=5,
    constraint_threshold=2.0,
    max_iteration=100,
    n_jobs=-1  # Use all available cores
):  
    import pandas as pd
    from joblib import Parallel, delayed

    selected = []
    history = []
    available_tfs = TF_all.copy()
    iteration = 0

    while len(selected) < max_tfs_n and available_tfs:
        # Parallel evaluation of all trial sets
        trial_sets = [(tf, selected + [tf]) for tf in available_tfs]

        def eval_candidate(tf, trial_set):
            res = obtimize_function(trial_set)
            return {
                "tf": tf,
                "age_shift": res["age_shift"],
                "detr_score": res["detr_score"],
                "valid": res["detr_score"] <= constraint_threshold
            }

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(tf, tfs) for tf, tfs in trial_sets
        )

        # Find best valid candidate
        best_candidate = None
        best_age_shift = float("inf")
        best_detr_score = None

        for res in results:
            print(res["age_shift"], ' best_age_shift', best_age_shift, 'detr_score', res["detr_score"])
            if res["valid"] and res["age_shift"] < best_age_shift:
                best_candidate = res["tf"]
                best_age_shift = res["age_shift"]
                print(best_age_shift)
                best_detr_score = res["detr_score"]
        aaaa

        if best_candidate is None:
            print(f"Stopped at size {len(selected)}: no candidates met the constraint.")
            break

        selected.append(best_candidate)
        available_tfs.remove(best_candidate)

        update = {
            "step": len(selected),
            "added_tf": best_candidate,
            "tfs": ",".join(selected),
            "age_shift": best_age_shift,
            "detr_score": best_detr_score,
            "selected": True
        }
        if True:
            print(update)
        history.append(update)

        iteration += 1
        if iteration > max_iteration:
            print(f"Stopped at size {len(selected)}: reached max iterations.")
            break

    df_results = pd.DataFrame(history)
    return selected, df_results


def obtimize_function(tfs):
    from ciim.src.common import par_simulation, datasets_all
    from ciim.src.insilico_perturbation.helper import wrapper_in_silico_perturbation, summarize_pathway_scores
    par = par_simulation
    par['tfs'] = tfs
    n_jobs = 20
    cell_type = 'CD8T'

    raw_rr = wrapper_in_silico_perturbation(par, n_jobs=n_jobs, cell_types=[cell_type], datasets=datasets_all)
    age_shift = raw_rr['age_shift']['age_shift'].mean()
    gene_score_shift_log2fc = summarize_pathway_scores(raw_rr, cols=['cell_type'])['gene_score_shift_log2fc'].abs().mean() #TODO: this is problematic. we are just assuming that any change in the gene score of these pathways are bad
    return {
        'age_shift': age_shift,         # Want to minimize
        'detr_score': gene_score_shift_log2fc          # Want to keep low
    }