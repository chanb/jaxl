import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

arch_name = "default_network"
results_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/eval_results/scrambling_10-dataset_seed_17/{}".format(arch_name)

out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

max_num_data = 100000
returns = {}
for result in os.listdir(results_dir):
    result_info = result.split("-")
    num_data = int(result_info[0].split("num_data_")[-1])
    if num_data > max_num_data:
        continue
    returns.setdefault(num_data, [])
    with open(os.path.join(results_dir, f"{result}/result.pkl"), "rb") as f:
        data = pickle.load(f)
        returns[num_data].append(data["episodic_returns"])
num_datas = sorted(list(returns.keys()))

means = []
stds = []
for num_data in num_datas:
    means.append(np.mean(returns[num_data]))
    stds.append(np.std(returns[num_data]))
means = np.array(means)
stds = np.array(stds)

plt.plot(num_datas, means)
plt.fill_between(num_datas, means - stds, means + stds, alpha=0.3)

plt.savefig("{}/test.png".format(out_dir))
