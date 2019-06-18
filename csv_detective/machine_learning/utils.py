import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


def visualize_matrices(list_matrices, names):
    assert len(list_matrices) == len(names), "Names and number of matrices should be the same"
    for idx, mat in enumerate(list_matrices):
        plt.subplot(len(names), 1, idx + 1)
        plt.title(names[idx])
        plt.spy(mat, aspect="auto", markersize=0.5)
    plt.show()


# noinspection PyProtectedMember
def visualize_multivariate(X, y):
    assert X.shape[0] == len(y)
    df = pd.DataFrame._from_arrays(X.toarray().T, columns=range(X.shape[1]), index=range(X.shape[0]))
    df["labels"] = y
    if X.shape[1] > 1:
        sns.pairplot(df, hue="labels")
    else:
        df.groupby(["labels"])[0].plot(legend=True)
    plt.show()
