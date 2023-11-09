import Distributions as Dist
import LinearAlgebra as LA
import Statistics as Stat
import Random as Rand


# Cache "random" numbers for fast sampling
struct FastGP
    norm_numbers    # length > dim * n_samples
    length::Int
end

function MakeFastGP(;n_samples::Int)
    norm_samples = Rand.randn(n_samples + 100)
    return FastGP(norm_samples, n_samples)
end

# Draw a "random" normal distribution from selection
function draw_normals(GP_gen::FastGP, n::Int)
    k = GP_gen.length
    norm_numbers = GP_gen.norm_numbers

    # Start point
    i = rand(1:k)

    indices = [(i + j - 1) % k + 1 for j in 0:(n-1)]
    return norm_numbers[indices]
end

# Using FastGP, quicky convert to normal distribution
function sample_multivariate_normals(GP_gen::FastGP, μs, Σs; n_repeats::Int)
    # μs.shape = (n_classes, n_points)
    # Return shape: (n_classes, n_repeats, n_points)
    n_points = size(μs[1], 1)
    n_gaussains = n_points * n_repeats

    multi_gaussians = []
    for (μ, Σ) in zip(μs, Σs)
        # Draw Gaussian
        samples = draw_normals(GP_gen, n_gaussains)
        samples = reshape(samples, (n_points, n_repeats))

        # Get std. matrix
        L = LA.cholesky(Σ).L

        transformed_samples = μ .+ L * samples        # Shape (n_points, n_samples)

        push!(multi_gaussians, transformed_samples)
    end
    return multi_gaussians
end

GP_gen = MakeFastGP(n_samples=1000*200)

μs = [[1, 2], [3., 4.], [1.,4.]]
Σs_full =[[1.0 0.2; 0.2 3.0], [1.0 0.5; 0.5 1.0], [1.0 0.; 0. 1.0]]

samples_true = sample_multivariate_normals(GP_gen, μs, Σs_full, n_repeats=1000)
samples_true = cat(samples_true..., dims=3)

println(size(samples_true))
println(size(samples_true[:, :, 1]))

function compute_mean_covariance(data)
    # data = permutedims(data, (2, 1))

    # Number of samples
    n = size(data, 2)

    # Compute mean
    μ = Stat.mean(data, dims=2)

    # Center the data
    centered_data = data .- μ

    # Compute covariance
    Σ = (centered_data * centered_data') / (n - 1)

    return μ, Σ
end

println(compute_mean_covariance(samples_true[:, :, 1]))
# println(compute_mean_covariance(samples_true[2, :, :]))
