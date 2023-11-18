import metaworld
import random

from skimage.color import rgb2gray
from skimage.transform import resize

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

task = "box-close-v2"
ml1 = metaworld.ML1(task) # Construct the benchmark, sampling tasks

env = ml1.train_classes[task](render_mode="rgb_array")  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
env.camera_name="corner2"

obs = env.reset(seed=0)  # Reset environment

# a = env.action_space.sample()  # Sample an action
# obs, reward, terminated, truncated, info = env.step(a)  # Step the environment with the sampled random action

import matplotlib.pyplot as plt

# # import ipdb
# # ipdb.set_trace()

img = resize(env.render(), (100, 100))

def crop_center(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# img = crop_center(img, 100, 100)
print(img.shape)

print(env.action_space.shape)
plt.imshow(img[::-1])
plt.show()
