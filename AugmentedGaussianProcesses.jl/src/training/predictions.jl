import Random as Rand
import LinearAlgebra as LA

## Nodes and weights for predictions based on quadrature
# const pred_nodes, pred_weights = (x -> (x[1] .* sqrt2, x[2] ./ sqrtπ))(gausshermite(100))


# Cache "random" numbers for fast sampling
struct FastGP
    norm_numbers::Vector{Float64}    # length > dim * n_samples
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
function sample_multivariate_normals(GP_gen::FastGP, μs:: Tuple{Vararg{Vector{Float64}}},
    Σs::Tuple{Vararg{LinearAlgebra.Symmetric{Float64, Matrix{Float64}}}}; n_repeats::Int)
    # μs.shape = (n_classes, n_points)
    # Return shape: (n_classes, n_repeats, n_points)
    println("Type of Σs:", typeof(Σs))

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

function sample_multivariate_normals(::Nothing, means:: Tuple{Vararg{Vector{Float64}}}, covariances, n_repeats::Int)
    all_samples = []

    # Iterate over each mean-covariance pair
    for i in 1:length(means)
        dist = MvNormal(means[i], covariances[i])
        samples = rand(dist, n_samples)
        push!(all_samples, samples)
    end

    return all_samples
end


@traitfn function _predict_f(
    m::TGP, X_test::AbstractArray; cov::Bool=true, diag::Bool=true
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}

    Ks = if isnothing(m.final_Ks)
        compute_K.(m.f, Zviews(m), T(jitt))
    else
        getproperty.(m.final_Ks.kernel_matrices, :K)
        #m.final_Ks
    end


    k_star = kernelmatrix.(kernels(m), (X_test,), Zviews(m))
    μf = k_star .* (Ks .\ means(m))
    if !cov
        return (μf, nothing)
    end
    A = Ks .\ (Ref(I) .- covs(m) ./ Ks)
    if diag
        k_starstar =
            kernelmatrix_diag.(kernels(m), Ref(X_test)) .+
            Ref(T(jitt) * ones(T, size(X_test, 1)))
        σ²f = k_starstar .- diag_ABt.(k_star .* A, k_star)
        return (μf, σ²f)
    else
        k_starstar = kernelmatrix.(kernels(m), Ref(X_test)) .+ T(jitt) * [I]
        Σf = Symmetric.(k_starstar .- k_star .* A .* transpose.(k_star))
        Σf = Tuple(Σf)
        return (μf, Σf)
    end
end



function predict_f(
    model::AbstractGPModel,
    X_test::AbstractVector,
    cov::Bool=false,
    diag::Bool=true,
)
    return _predict_f(model, X_test; cov, diag)
end


function predict_y(
    model::AbstractGPModel, X_test::AbstractVector)
    return predict_y(likelihood(model), only(_predict_f(model, X_test; cov=false)))
end


function proba_y(
    model::AbstractGPModel, X_test::AbstractVector; model_cov::Bool=true, diag::Bool=false, nSamples::Int=1000, sampler::Union{FastGP, Nothing}=nothing)
    μ_f, Σ_f = _predict_f(model, X_test; cov=model_cov, diag=diag)

    if model_cov
        return (compute_proba_full(model.likelihood, μ_f, Σ_f, nSamples, sampler=sampler), (μ_f, Σ_f))
    else
        means = compute_proba_mean(model.likelihood, μ_f)
        return ((means, zeros(size(means))), (μ_f, Σ_f))
    end
end


function StatsBase.mean_and_var(lik::AbstractLikelihood, fs::AbstractMatrix)
    vals = lik.(eachcol(fs))
    return StatsBase.mean(vals), StatsBase.var(vals)
end



function compute_proba_f(l::AbstractLikelihood, f::AbstractVector{<:Real})
    return compute_proba.(l, f)
end


