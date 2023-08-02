import os

tasks = []
control_modes = []

exp_name = "finetune_mtbc_main"
run_time = "02:00:00"
num_evaluation_episodes = 30
rollout_seed = 9999

base_dir = "/home/chanb/scratch/data"
expert_dir = os.path.join(base_dir, "experts")

exp_dir = os.path.join(base_dir, exp_name)

save_dir = os.path.join(base_dir, f"evaluations/results-{exp_name}")

dat_content = ""
num_runs = 0
for task in tasks:
    for control_mode in control_modes:
        curr_env_dir = os.path.join(exp_dir, task, control_mode, "runs")
        curr_expert_dir = os.path.join(expert_dir, task, control_mode)
        curr_save_dir = os.path.join(save_dir, task, control_mode)
        os.makedirs(curr_save_dir, exist_ok=True)

        expert_paths = {}
        trained_paths = {}
        env_seeds = []
        for bc_variant_name in os.listdir(curr_env_dir):
            if bc_variant_name == ".DS_Store":
                continue
            env_seeds.append(bc_variant_name.split(".")[2])
            trained_paths[env_seeds[-1]] = os.path.join(curr_env_dir, bc_variant_name)

        for expert_path, _, filenames in os.walk(curr_expert_dir):
            env_seed = os.path.basename(os.path.dirname(expert_path))
            if env_seed not in env_seeds:
                continue

            for filename in filenames:
                if filename != "config.json":
                    continue

                expert_paths[env_seed] = expert_path

        for env_i, env_seed in enumerate(env_seeds):
            reference_agent_path = expert_paths[env_seed]

            trained_dir = trained_paths[env_seed]
            num_variants = len(os.listdir(trained_dir))

            episodic_returns = {}
            for variant_i, variant_name in enumerate(["expert", *os.listdir(trained_dir)]):
                print(
                    f"Processing {variant_name} ({variant_i + 1} / {num_variants + 1} variants)"
                )

                if variant_name == "expert":
                    dat_content += "export env_seed={} rollout_seed={} num_evaluation_episodes={} variant_name={} runs_path={} --save_dir={}\n".format(
                        env_seed, rollout_seed, num_evaluation_episodes, variant_name, reference_agent_path, curr_save_dir
                    )
                else:
                    variant_path = os.path.join(trained_dir, variant_name)
                    for variant in os.listdir(variant_path):
                        runs_path = os.path.join(variant_path, variant)
                        dat_content += "export env_seed={} rollout_seed={} num_evaluation_episodes={} variant_name={} runs_path={} reference_agent_path={} --save_dir={}\n".format(
                            env_seed, rollout_seed, num_evaluation_episodes, variant_name, runs_path, reference_agent_path, curr_save_dir
                        )

dat_path = os.path.join(
    f"./export-evaluate_all-{exp_name}.dat"
)
with open(dat_path, "w+") as f:
    f.writelines(dat_content)

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
sbatch_content += "#SBATCH --account=def-schuurma\n"
sbatch_content += "#SBATCH --time={}\n".format(run_time)
sbatch_content += "#SBATCH --cpus-per-task=1\n"
sbatch_content += "#SBATCH --mem=3G\n"
sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
sbatch_content += (
    "#SBATCH --output=/home/chanb/scratch/run_reports/evaluate_all/%j.out\n".format(
        exp_name
    )
)
sbatch_content += "module load python/3.9\n"
sbatch_content += "module load mujoco\n"
sbatch_content += "source ~/jaxl_env/bin/activate\n"
sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
sbatch_content += " < {}`\n".format(dat_path)
sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
sbatch_content += 'echo "Current working directory is `pwd`"\n'
sbatch_content += 'echo "Running on hostname `hostname`"\n'
sbatch_content += 'echo "Starting run at: `date`"\n'
sbatch_content += "python3 {} \\\n".format(os.path.join(os.path.dirname(os.path.__file__), "run_evaluation.py"))
sbatch_content += "  --config_path=${config_path} \\\n"
sbatch_content += "  --run_seed=${run_seed}\n"
sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

with open(
    os.path.join(
        f"./run_all-evaluate_all-{exp_name}.sh"
    ),
    "w+",
) as f:
    f.writelines(sbatch_content)
