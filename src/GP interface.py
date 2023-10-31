from julia.api import Julia

jl = Julia(runtime='/home/maccyz/julia-1.9.3/bin/julia')
from julia import Main

Main.include("./MakeGP.jl")

import numpy as np
from matplotlib import pyplot as plt


def generate_cluster(center, n_points=100, std_dev=0.5):
    """Generate a cluster of points."""
    return np.random.normal(center, std_dev, (n_points, 2))


def gen_data():
    # Generate 2D dataset
    n_points_per_cluster = 100
    left_cluster = generate_cluster(center=[-1, 0], n_points=n_points_per_cluster)
    right_cluster = generate_cluster(center=[1, 0], n_points=n_points_per_cluster)

    # Labels for the clusters
    left_labels = np.ones((n_points_per_cluster, 1))
    right_labels = np.zeros((n_points_per_cluster, 1))

    # Combine the datasets
    X = np.vstack([left_cluster, right_cluster])
    y = np.vstack([left_labels, right_labels]).ravel()
    y += 1
    return X, y


def test_data(n_points):
    x_values = np.linspace(-3, 3, n_points)  # 100 points from -3 to 3 for the x-axis
    y_values = np.linspace(-3, 3, n_points)  # 100 points from -3 to 3 for the y-axis

    # Create a 2D grid of points
    xx, yy = np.meshgrid(x_values, y_values)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    points = np.vstack((x_flat, y_flat)).T

    # Convert to a list of tuples (optional)
    # points_list = [tuple(point) for point in points]
    return points


def main():
    n_points = 50
    X, y = gen_data()
    X_test = test_data(n_points)
    # Make the Gaussian process
    gp = Main.make_GP(X, y, 2, 1., 1.)

    # Train the GP
    vars, inv_lens = Main.train_GP(gp, 500)

    # Print kernel params
    for var, inv_len in zip(vars, inv_lens):
        print(f"variance: {var}, lengthscale: {1 / inv_len}")

    # Predict the labels for the test data
    probs, stds = Main.pred_proba(gp, X_test)

    probs = np.array(probs)

    probs = probs[:, 1]

    # Show the probs
    plt.imshow(probs.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)
    plt.show()


if __name__ == "__main__":
    main()
