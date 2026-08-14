import numpy as np
import torch


def get_class_weights(
    labels,
    num_classes
):

    counts = np.bincount(
        labels,
        minlength=num_classes
    )


    # Effective number of samples
    beta = 0.999


    effective_num = (
        1.0 -
        np.power(
            beta,
            counts
        )
    )


    weights = np.zeros(
        num_classes,
        dtype=np.float32
    )


    valid = counts > 0


    weights[valid] = (
        (1.0 - beta)
        /
        effective_num[valid]
    )


    # Normalize
    if weights.sum() > 0:

        weights = (
            weights
            /
            weights[valid].mean()
        )


    return torch.tensor(
        weights,
        dtype=torch.float32
    )