import json
import os


num_data_to_check = [100, 250, 500, 1000, 5000]
expert_dataset = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/expert_data/test_metaworld-seed_0.gzip"
bc_template = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/bc_template.json"

for num_data in num_data_to_check:
    assert os.path.isfile(bc_template), f"{bc_template} is not a file"
    with open(bc_template, "r") as f:
        template = json.load(f)
