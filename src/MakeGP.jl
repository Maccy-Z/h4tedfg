using AugmentedGaussianProcesses

function make_GP(X, y, n_class::Int, init_sigma::Real, init_scale::Real)
    #kernel = init_sigma * SqExponentialKernel() ∘ ScaleTransform(init_scale)
    kernel = init_sigma * Matern52Kernel() ∘ ScaleTransform(init_scale)
    m = VGP(
        X,
        y,
        kernel,
        LogisticSoftMaxLikelihood(n_class),
        AnalyticVI(),
        optimiser=Optimisers.Descent(0.01),
        obsdim=1
    )
    return m
end

function train_GP(m::AbstractGPModel, n_iter::Int)
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

function pred_proba(m, X)
    # X must be a vector of vectors, not matrix
    X = [row for row in eachrow(X)]

    probs, stds = proba_y(m, X)
    return probs, stds
end

