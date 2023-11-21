import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

arch_name = "large_network"
results_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/eval_results/{}".format(arch_name)

out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

max_num_data = 100000
num_datas = []
for result in os.listdir(results_dir):
    num_data = int(result.split("num_data_")[-1])
    if num_data > max_num_data:
        continue
    num_datas.append(num_data)
num_datas = sorted(num_datas)

returns = []
for num_data in num_datas:
    with open(os.path.join(results_dir, f"num_data_{num_data}/result.pkl"), "rb") as f:
        data = pickle.load(f)
        returns.append(data["episodic_returns"])

returns = np.array(returns)

plt.plot(num_datas, np.mean(returns, axis=-1))
plt.savefig("test.png")
