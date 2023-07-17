import matplotlib as mpl

import matplotlib.pyplot as plt
from scipy.stats import norm

from plot_utils import set_size, pgf_with_latex


# Use the seborn style
plt.style.use('seaborn')
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)


x = norm.rvs(size=100)
y = norm.pdf(x)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679
fig, ax = plt.subplots(1, 1, figsize=set_size(doc_width_pt, 0.49, (1, 1)))
ax.scatter(x, y)
ax.set_ylabel('density')
ax.set_xlabel('$x$')

fig.tight_layout()
fig.savefig('norm_2.pdf', format='pdf', bbox_inches='tight')
