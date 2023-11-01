import Distributions as Dist
import LinearAlgebra as LA
import Statistics as Stat

struct FastGP
    norm_samples    # shape = (dim, n_samples)
    n_samples::Int
end

function MakeFastGP(;dim::Int, n_samples::Int)
    μ = zeros(dim)
    Σ = LA.I(dim)
    dist = Dist.MvNormal(μ, Σ)
    norm_samples = rand(dist, n_samples)

    return FastGP(norm_samples, n_samples)
end

# Using FastGP, quicky convert to normal distribution
function make_normals(GP_gen::FastGP, μ, Σ)

    # Get std. matrix
    L = LA.cholesky(Σ).L

    samples = GP_gen.norm_samples
    transformed_samples = μ .+ L * samples
    return transformed_samples
end
#
# GP_gen = MakeFastGP(dim=2, n_samples=1000)
#
# μs = [0.0, 0.6]
# Σs_full = [1.0 0.23; 0.23 2.0]
#
# samples_true = make_normals(GP_gen, μs, Σs_full)
#
#
# function compute_mean_covariance(data)
#     # Number of samples
#     n = size(data, 2)
#
#     # Compute mean
#     μ = Stat.mean(data, dims=2)
#
#     # Center the data
#     centered_data = data .- μ
#
#     # Compute covariance
#     Σ = (centered_data * centered_data') / (n - 1)
#
#     return μ, Σ
# end
#
# println(compute_mean_covariance(samples_true))
