
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ciim.src.common import colors_blind

def plot_greedy_tf_selection(df_results, figsize=(3, 2)):
    fig, ax1 = plt.subplots(figsize=figsize)
    color1 = colors_blind[1]
    color2 = colors_blind[0]
    margins = dict(x=0.2, y=0.1)

    # Plot age_shift
    ax1.plot(df_results['step'], df_results['age_shift'], marker='o', color=color1, label='Age Shift')
    # ax1.set_xlabel('Step (TFs added)')
    ax1.set_ylabel('Age Shift \n (pseudo-years)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Show TFs on x-axis
    ax1.set_xticks(df_results['step'])
    ax1.set_xticklabels(df_results['added_tf'], rotation=45, ha='right')
    

    # Plot detr_score on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df_results['step'], df_results['best_genescore_shift'], marker='s', linestyle='--', color=color2, label='Risk score')
    ax2.set_ylabel('Risk score', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=[.4, .7], frameon=False)

    # ax1.spines[['top', 'right']].set_visible(False)  # Hide top and right spines
    # ax2.spines[['top']].set_visible(False)  # Hide top and right spines
    ax1.margins(**margins)
    ax2.margins(**margins)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.2)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.2)
    
    # Add a title
    # plt.title('Greedy TF Selection: Age Shift vs Detrimental Score', pad=15)
    # plt.tight_layout()
    
def run_greedy_tf_optimization_parallel(
    TF_all,
    obtimize_function,
    max_tfs_n=5,
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
                "genescore_shift": res["genescore_shift"],
                "overall_score": res["overall_score"]
            }

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(tf, tfs) for tf, tfs in trial_sets
        )

        # Find best valid candidate
        best_overall_score = 0
        best_candidate = None
        best_genescore_shift = None

        for res in results:
            if res["overall_score"] > best_overall_score:
                best_candidate = res["tf"]
                best_age_shift = res["age_shift"]
                best_genescore_shift = res["genescore_shift"]
                best_overall_score = res["overall_score"]


        selected.append(best_candidate)
        available_tfs.remove(best_candidate)

        update = {
            "step": len(selected),
            "added_tf": best_candidate,
            "tfs": ",".join(selected),
            "age_shift": best_age_shift,
            "best_genescore_shift": best_genescore_shift,
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


def obtimize_function(tfs, cell_type='CD8T', n_jobs=20, genefc_threshold=2.2,
                        age_shift_scale=10,
                        genescore_shift_scale=1):
    from ciim.src.common import par_simulation, datasets_all
    from ciim.src.insilico_perturbation.helper import wrapper_in_silico_perturbation
    from ciim.src.insilico_perturbation.biological_analysis.helper import summarize_pathway_scores, summarize_essential_genes_shift
    par = par_simulation
    par['tfs'] = tfs

    # - run the simulation
    raw_rr = wrapper_in_silico_perturbation(par, n_jobs=n_jobs, cell_types=[cell_type], datasets=datasets_all)
    # - get the age shift
    age_shift = raw_rr['age_shift']['age_shift'].mean()

    # - summarize the results
    score_shift_summary = summarize_pathway_scores(raw_rr, cols=['cell_type', 'tf'])

    # - get the gene score shift: essential genes
    score_shift_mean = summarize_essential_genes_shift(score_shift_summary) # get the mean for the essential genes
    score_shift_mean = score_shift_mean['value'][0]

    # - ovearal objective score
    score_shift_norm = score_shift_mean/genescore_shift_scale
    age_shift_norm = age_shift/age_shift_scale

    diff = max(score_shift_norm - genefc_threshold, 0)
    score_penalty = np.exp(diff) - 1

    overall_score = (-age_shift_norm) - score_penalty

    # gene_score_shift_log2fc = summarize_pathway_scores(raw_rr, cols=['cell_type'])['gene_score_shift_log2fc'].abs().mean() #TODO: this is problematic. we are just assuming that any change in the gene score of these pathways are bad
    return {
        'age_shift': age_shift,         # Want to minimize
        'genescore_shift': score_shift_mean,          # Want to keep low
        'overall_score': overall_score,  # Want to maximize
    }