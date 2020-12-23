import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from battleship.datagen import generate_shot_results, probs_given_shot_results


def plot_shots(board, shots, sample_boards):
    """Visualize the results of firing shots"""

    shot_results = generate_shot_results(board, shots)
    probs = probs_given_shot_results(sample_boards, shot_results)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    sns.heatmap(board, square=True, cbar=False, ax=ax1)
    sns.heatmap(shots, square=True, cbar=False, ax=ax2)
    sns.heatmap(shot_results, square=True, cbar=False, ax=ax3)
    sns.heatmap(probs, square=True, cbar=False, ax=ax4)

    ax1.set_title("ships")
    ax2.set_title("shots")
    ax3.set_title("hits & misses")
    ax4.set_title("probs")


def plot_shots_w_model(board, shots, model, sample_boards):
    """Visualize the model predicted results of firing shots"""

    rows, columns = shots.shape

    shot_results = generate_shot_results(board, shots)
    probs = probs_given_shot_results(sample_boards, shot_results)

    model_probs = np.squeeze(
        model.predict(shot_results.reshape(1, rows, columns, 1)))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

    sns.heatmap(board, square=True, cbar=False, ax=ax1)
    sns.heatmap(shot_results, square=True, cbar=False, ax=ax2)
    sns.heatmap(probs, square=True, cbar=False, ax=ax3)
    sns.heatmap(model_probs, square=True, cbar=False, ax=ax4)

    ax1.set_title("ships")
    ax2.set_title("hits & misses")
    ax3.set_title("probs")
    ax4.set_title("model probs")
