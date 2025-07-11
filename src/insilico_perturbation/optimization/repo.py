
def run_bayesian_optimization(
    TF_all,
    obtimize_function,
    n_tfs=5,
    total_trials=50,
    constraint_threshold=2.0
):
    from ax.service.managed_loop import optimize
    # --- Define Objective Function ---
    def ax_objective(params):
        tf_list = list({params[f"tf{i+1}"] for i in range(n_tfs)})
        if len(tf_list) < n_tfs:
            return {"age_shift": 10.0, "detr_score": 1.0}  # Penalize duplicates
        result = obtimize_function(tf_list)
        return {
            "age_shift": result["age_shift"],
            "detr_score": result["detr_score"]
        }

    # --- Search Space ---
    search_space = [
        {
            "name": f"tf{i+1}",
            "type": "choice",
            "values": TF_all,
        } for i in range(n_tfs)
    ]

    # --- Run Ax Optimization ---
    best_parameters, values, experiment, model = optimize(
        parameters=search_space,
        evaluation_function=ax_objective,
        objective_name="age_shift",
        outcome_constraints=[f"detr_score <= {constraint_threshold}"],
        total_trials=total_trials,
        minimize=True,
    )

    # --- Build Comprehensive Results DataFrame ---
    data = experiment.fetch_data().df
    df_scores = data.pivot(index="trial_index", columns="metric_name", values="mean").reset_index()

    trial_param_records = []
    for trial_index, trial in experiment.trials.items():
        arm = trial.arm
        tf_values = [v for k, v in sorted(arm.parameters.items()) if k.startswith("tf")]
        trial_param_records.append({
            "trial_index": trial_index,
            "tfs": ",".join(sorted(tf_values))
        })
    df_params = pd.DataFrame(trial_param_records)
    df_all = pd.merge(df_params, df_scores, on="trial_index")

    # Add 'selected' flag
    best_tfs_str = ",".join(sorted(best_parameters.values()))
    df_all["selected"] = df_all["tfs"] == best_tfs_str

    # Sort and return
    df_all = df_all.sort_values("age_shift").reset_index(drop=True)

    return best_parameters, values, df_all
def get_run_record(experiment, best_parameters):
    # 1. Fetch scores from Ax experiment
    data = experiment.fetch_data().df

    # 2. Extract TFs and combine into a single column
    trial_param_records = []

    for trial_index, trial in experiment.trials.items():
        arm = trial.arm
        tf_values = [v for k, v in sorted(arm.parameters.items()) if k.startswith("tf")]
        trial_param_records.append({
            "trial_index": trial_index,
            "tfs": ",".join(sorted(tf_values))  # sort to standardize order
        })

    df_params = pd.DataFrame(trial_param_records)

    # 3. Reshape scores to wide format
    df_scores = data.pivot(index="trial_index", columns="metric_name", values="mean").reset_index()

    # 4. Merge
    df_all = pd.merge(df_params, df_scores, on="trial_index")
    chosen_tfs = sorted(best_parameters.values())
    chosen_tfs_str = ",".join(chosen_tfs)

    # Add 'selected' column: True if tfs match the chosen set
    df_all["selected"] = df_all["tfs"] == chosen_tfs_str
    return df_all