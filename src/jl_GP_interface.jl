using AugmentedGaussianProcesses

function make_GP(X, y; n_class::Int, init_sigma::Real, init_scale::Real)
    #kernel = init_sigma * SqExponentialKernel() ∘ ScaleTransform(init_scale)
    kernel = init_sigma * Matern32Kernel() ∘ ScaleTransform(init_scale)
    m = VGP(
        X,
        y,
        kernel,
        LogisticSoftMaxLikelihood(n_class),
        AnalyticVI(),
        optimiser=Optimisers.Descent(0.005),
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

    return vars, inv_lens
end

function pred_proba(m, X; diag, model_cov)
    # X must be a vector of vectors, not matrix
    X = [row for row in eachrow(X)]

    probs, fs = proba_y(m, X, diag=diag, model_cov=model_cov)
    return probs, fs
end

# Fast normal sampling
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
