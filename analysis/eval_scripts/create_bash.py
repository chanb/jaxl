import argparse
import os


def main(args):
    learner_path = args.learner_path
    script_path = args.script_path
    save_dir = args.save_dir

    model_path = os.path.join(learner_path, "models")
    checkpoints = sorted(
        [int(model_name.split(".dill")[0]) for model_name in os.listdir(model_path)]
    )

    sh_content = "#!/bin/bash \n"
    sh_content += "source /home/bryan/research/jaxl/.venv/bin/activate \n"
    sh_content += f"mkdir -p {save_dir} \n"
    for checkpoint_i in checkpoints:
        sh_content += (
            "python /home/bryan/research/jaxl/analysis/eval_scripts/evaluate_model.py"
        )
        sh_content += "    --device=cpu"
        sh_content += "    --random"
        sh_content += "    --total_episodes=100"
        sh_content += f"    --learner_path={learner_path}"
        sh_content += f"    --checkpoint_idx={checkpoint_i}"
        sh_content += f"    --save_dir={save_dir} \n \n"

    with open(script_path, "w+") as f:
        f.write(sh_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner_path", type=str, required=True, help="The model to evaluate"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The directory to save the result in",
    )
    parser.add_argument(
        "--script_path", type=str, required=True, help="The script name to save"
    )
    args = parser.parse_args()
    main(args)
