import _pickle as pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

data = pickle.load(
    gzip.open(
        "/Users/chanb/research/personal/jaxl/scripts/mtil/local/rebuttal/test_metaworld.gzip",
        "rb",
    )
)

imgs = [
    Image.fromarray((np.transpose(img, axes=(2, 1, 0)) * 255).astype(np.uint8))
    for img in data["observations"]
]
# duration is the number of milliseconds between frames; this is 40 frames per second
imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
