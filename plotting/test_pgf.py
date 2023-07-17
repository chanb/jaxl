import matplotlib as mpl
# Use the pgf backend (must be done before import pyplot interface)
mpl.use('pgf')

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
fig, ax = plt.subplots(1, 2, figsize=set_size(doc_width_pt, 1.0, (1, 2)))
ax[0].scatter(x, y)
ax[0].set_ylabel('density')
ax[0].set_xlabel('$x$')
ax[0].set_label('$x$')

ax[1].scatter(x, y)
ax[1].set_ylabel('density')
ax[1].set_xlabel('$x$')

fig.tight_layout()
fig.savefig('norm.pgf', format='pgf')
