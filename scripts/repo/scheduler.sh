# bash_file="scripts/run_process_perturbation.sh"
bash_file="scripts/run_train_nn.sh"
jid=$(sbatch $bash_file | awk '{print $4}'); \
out_file="logs/${jid}.out"; \
err_file="logs/${jid}.err"; \
while [ ! -f "$out_file" ] || [ ! -f "$err_file" ]; do sleep 1; done; \
echo "$out_file" is running