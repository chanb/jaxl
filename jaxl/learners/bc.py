from typing import Any, Dict

import numpy as np
import timeit

from jaxl.constants import *
from jaxl.learners.supervised import SupervisedLearner


"""
Standard Behavioural Cloning.
"""


class BC(SupervisedLearner):
    """
    Behavioural Cloning (BC) algorithm. This extends `SupervisedLearner`.
    """

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the policy.

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """
        tic = timeit.default_timer()
        obss, h_states, acts_e, _, _, _, _, _, _, _ = self._buffer.sample(
            self._config.batch_size
        )
        self.model_dict, aux = self.train_step(self._model_dict, obss, h_states, acts_e)
        update_time = timeit.default_timer() - tic
        assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": aux[CONST_AGG_LOSS],
            f"time/{CONST_UPDATE_TIME}": update_time,
        }
        for loss_key in self._config.losses:
            aux[CONST_LOG][f"losses/{loss_key}"] = aux[loss_key][CONST_LOSS]

        return aux
