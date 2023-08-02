import _pickle as pickle
import gzip
import numpy as np
import os


# dir_to_filter = "/Users/chanb/research/personal/mtil_results/final_results/data/expert_data/walker_disc"
dir_to_filter = "/home/chanb/scratch/data/expert_data/cartpole_cont"


# cartpole
required_min = 600

# frozenlake
# required_min = 50

# pendulum
# required_min = -500

# cheetah
# required_min = 200

# walker
# required_min = 300

assert os.path.isdir(dir_to_filter)

all_rets = []
for filename in os.listdir(dir_to_filter):
    curr_dataset = os.path.join(dir_to_filter, filename)
    dataset_dict = pickle.load(gzip.open(curr_dataset, "rb"))

    num_episodes = np.sum(dataset_dict["dones"])
    last_done_idx = np.where(dataset_dict["dones"])[0][-1]

    expected_returns = np.sum(dataset_dict["rewards"][:last_done_idx]) / num_episodes

    all_rets.append(expected_returns)

    if expected_returns < required_min:
        print("Removing {}".format(curr_dataset))
        os.remove(curr_dataset)

print(
    "Min/max/mean avg return: {}/{}/{}".format(
        np.min(all_rets), np.max(all_rets), np.mean(all_rets)
    )
)
print(sorted(all_rets))
