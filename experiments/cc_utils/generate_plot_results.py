import os

from constants import LOG_DIR

plot_dir = "./plots"
os.makedirs(plot_dir, exist_ok=True)

sh_content = ""
for runs_dir in os.listdir(LOG_DIR):
    result_path = os.path.join(LOG_DIR, runs_dir)
    save_path=os.path.join(plot_dir, runs_dir)
    sh_content += "python plot_results.py --results_dir={} --save_path={} --key=accuracies --context=none \n".format(
        result_path,
        save_path,
    )
    sh_content += "python plot_results.py --results_dir={} --save_path={} --key=accuracies --context=last \n".format(
        result_path,
        save_path,
    )
    sh_content += "python plot_results.py --results_dir={} --save_path={} --key=losses --context=none \n".format(
        result_path,
        save_path,
    )
    sh_content += "python plot_results.py --results_dir={} --save_path={} --key=losses --context=last \n".format(
        result_path,
        save_path,
    )

with open(
    "./run_plot_results.sh",
    "w+",
) as f:
    f.writelines(sh_content)
