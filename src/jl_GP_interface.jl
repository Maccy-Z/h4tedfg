using AugmentedGaussianProcesses
using Pkg
#Pkg.add("Distributions")
import Distributions as Dist
import LinearAlgebra as LA
import Statistics as Stat

function make_sampler(n_samples)
    return MakeFastGP(n_samples=n_samples)
end


function make_GP(X::Matrix{Float64}, y::Vector{Float64}; n_class::Int, init_sigma::Real, init_scale::Real, optimiser::String=nothing)
    #kernel = init_sigma * SqExponentialKernel() ∘ ScaleTransform(init_scale)
    kernel = init_sigma * Matern32Kernel() ∘ ScaleTransform(init_scale)

    if optimiser == "ADAM"
        optimiser = Optimisers.ADAM(0.01)
    elseif optimiser == "SGD"
        optimiser = Optimisers.Descent(0.01)
    else
        optimiser = nothing
    end

    m = VGP(
        X,
        y,
        kernel,
        LogisticSoftMaxLikelihood(n_class),
        AnalyticVI(),
        optimiser=optimiser,
        obsdim=1,
        length_bounds=(0.2, 10),
        var_bounds=(0.1, 100.),
    )
    return m
end

function train_GP(m::AbstractGPModel; n_iter::Int)
    train!(m, n_iter)

    # Print out kernel parameters
    vars, inv_lens = [], []
    for (phase_no, e) in enumerate(m.f)
        var = e.prior.kernel.kernel.σ²[1]
        inv_len = e.prior.kernel.transform.s[1]

        push!(vars, var)
        push!(inv_lens, inv_len)
    end

    return (vars, inv_lens)
end

function pred_proba(m, X; full_cov, model_cov, nSamples::Int=100)
    # X must be a vector of vectors, not matrix
    X = [row for row in eachrow(X)]

    probs, fs = proba_y(m, X, diag=!full_cov, model_cov=model_cov, nSamples=nSamples)
    return (probs, fs)
end

function pred_proba_sampler(m, X; full_cov, model_cov, nSamples=100, sampler=nothing)


    # X must be a vector of vectors, not matrix
    X = [row for row in eachrow(X)]

    probs, fs = proba_y(m, X, diag=!full_cov, model_cov=model_cov, nSamples=nSamples, sampler=sampler)
    return (probs, fs)
end

