# MIT License
#
# Copyright (c) 2018 Haoxintong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Custom loss"""

import numpy as np
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

__all__ = ["RingLoss"]

numeric_types = (float, int, np.generic)


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


class RingLoss(SoftmaxCrossEntropyLoss):
    """Computes the Ring Loss from
    `"Ring loss: Convex Feature Normalization for Face Recognition"
    <https://arxiv.org/abs/1803.00130>`_paper.

    .. math::
        p = \softmax({pred})

        L_SM = -\sum_i \log p_{i,{label}_i}

        L_R = \frac{\lambda}{2m} \sum_{i=1}^{m} (\Vert \mathcal{F}({x}_i)\Vert_2 - R )^2

    Parameters
    ----------
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and ring loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.

    """

    def __init__(self, lamda, weight_initializer=None, dtype='float32',
                 axis=-1, sparse_label=True, weight=None, batch_axis=0, **kwargs):
        super().__init__(axis=axis, sparse_label=sparse_label, weight=weight, batch_axis=batch_axis, **kwargs)

        self._lamda = lamda
        self.R = self.params.get('R', shape=(1,), init=weight_initializer,
                                 dtype=dtype, allow_deferred_init=True)

    def hybrid_forward(self, F, pred, label, embedding, R, sample_weight=None):
        # RingLoss
        emb_norm = F.norm(embedding, axis=1)
        loss_r = F.square(F.broadcast_sub(emb_norm, R))
        loss_r = loss_r * 0.5
        loss_r = _apply_weighting(F, loss_r, self._weight, sample_weight)

        # Softmax
        loss_sm = super().hybrid_forward(F, pred, label, sample_weight)

        return loss_sm + self._lamda * loss_r
