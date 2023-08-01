import _pickle as pickle
import gzip
import numpy as np
import os


dir_to_filter = "/Users/chanb/research/personal/mtil_results/final_results/data/expert_data/pendulum_disc"

required_min = -800

assert os.path.isdir(dir_to_filter)

for filename in os.listdir(dir_to_filter):
    curr_dataset = os.path.join(dir_to_filter, filename)
    dataset_dict = pickle.load(gzip.open(curr_dataset, "rb"))

    num_episodes = np.sum(dataset_dict["dones"])
    last_done_idx = np.where(dataset_dict["dones"])[0][-1]

    expected_returns = np.sum(dataset_dict["rewards"][:last_done_idx]) / num_episodes
    
    if expected_returns < required_min:
        print("Removing {}".format(curr_dataset))
        os.remove(curr_dataset)
