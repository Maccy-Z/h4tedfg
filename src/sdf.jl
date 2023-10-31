using Distributions
using LinearAlgebra


function sample_multivariate_normals(means, covariances, n_samples::Int)
    dist = MvNormal(means, covariances)
    return rand(dist, n_samples)

end

function generate_nd_standard_mvn(dim::Int, n_samples::Int)
    μ = zeros(dim)
    Σ = I(dim)
    dist = MvNormal(μ, Σ)
    return rand(dist, n_samples)
end

function transform_samples(samples, μ, Σ) where T
    L = cholesky(Σ).L
    transformed_samples = μ .+ L * samples
    return transformed_samples
end

function compute_mean_covariance(data) where T
    # Number of samples
    n = size(data, 2)

    # Compute mean
    μ = mean(data, dims=2)

    # Center the data
    centered_data = data .- μ

    # Compute covariance
    Σ = (centered_data * centered_data') / (n - 1)

    return μ, Σ
end

# Example usage
μs = [0.0, 0.6]
Σs_full = [1.0 0.23; 0.23 2.0] # Full covariances

samples_true = sample_multivariate_normals(μs, Σs_full, 1000)

std_norm = generate_nd_standard_mvn(2, 1000)
samples_regen = transform_samples(std_norm, μs, Σs_full)

true_mean_cov = compute_mean_covariance(samples_true)
regen_mean_cov = compute_mean_covariance(samples_regen)

println("True mean: ", true_mean_cov[1])
println("True covariance: ", true_mean_cov[2])

println("Regenerated mean: ", regen_mean_cov[1])
println("Regenerated covariance: ", regen_mean_cov[2])

struct MultiGaussian
    std_norm_gaussian
end

function make_gaussian(dim::Int, n_precompute::Int)
    return MultiGaussian(generate_nd_standard_mvn(dim, n_precompute))
end

function show_struct(s::MultiGaussian)
    println("MultiGaussian struct")
    println("std_norm_gaussian: ", s.std_norm_gaussian)
end
