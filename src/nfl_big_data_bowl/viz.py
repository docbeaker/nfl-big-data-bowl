import matplotlib.pyplot as plt
import numpy as np


def plot_linear_model_features(tackle_features, model, description):
    xes = np.arange(len(tackle_features))
    cf = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    sidxes = np.abs(cf).argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(xes, cf[sidxes])
    plt.xticks(
        xes, labels=[tackle_features[_i] for _i in sidxes], ha="right", rotation=30
    )
    plt.grid()
    plt.axhline(color="k")
    plt.ylabel(f"feature weight ({description})")
    plt.show()
