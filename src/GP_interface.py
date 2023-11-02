from julia.api import Julia

jl = Julia()
from julia import Main

Main.include("./jl_GP_interface.jl")

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
    n_points = 100
    X, y = gen_data()
    X_test = test_data(n_points)
    n_samples = 1000
    sampler_jl = Main.make_sampler(len(X_test) * n_samples)

    # Make the Gaussian process
    gp_jl = Main.make_GP(X, y, n_class=2, init_sigma=1., init_scale=1., optimiser="ADAM")

    # Train the GP
    vars, inv_lens = Main.train_GP(gp_jl, n_iter=1000)

    # Print kernel params
    for var, inv_len in zip(vars, inv_lens):
        print(f"variance: {var:.3g}, lengthscale: {1 / inv_len :.3g}")

    # Predict the labels for the test data
    #(probs, stds), _ = Main.pred_proba(gp, X_test, diag=True, model_cov=True)
    (probs, p_std), (f_mu, f_var) = Main.pred_proba_sampler(gp_jl, X_test, full_cov=True, model_cov=True, nSamples=n_samples, sampler=sampler_jl)

    f_mu, f_std = np.array(f_mu), np.sqrt(np.abs(np.array(f_var)))
    probs, p_std = np.array(probs), np.array(p_std)


    f_mu, f_std = f_mu[0], f_std[0]
    probs, p_std = probs[:, 0], p_std[:, 0]

    # Show the probs
    plt.subplot(1, 2, 1)
    plt.imshow(f_mu.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)

    plt.subplot(1, 2, 2)
    plt.imshow(probs.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)
    plt.show()

    # Show the stds
    # plt.subplot(1, 2, 1)
    # plt.imshow(f_std.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)

    plt.subplot(1, 2, 2)
    plt.imshow(p_std.reshape((n_points, n_points)), origin="lower", extent=(-3, 3, -3, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4)
    plt.show()

if __name__ == "__main__":
    main()
