# File treating all the prediction functions

## Nodes and weights for predictions based on quadrature
const pred_nodes, pred_weights = (x -> (x[1] .* sqrt2, x[2] ./ sqrtπ))(gausshermite(100))


@traitfn function _predict_f(
    m::TGP, X_test::AbstractArray, state=nothing; cov::Bool=true, diag::Bool=true
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}

    Ks = if isnothing(state)
        compute_K.(m.f, Zviews(m), T(jitt))
    else
        getproperty.(state.kernel_matrices, :K)
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
    state=nothing;
    cov::Bool=false,
    diag::Bool=true,
)
    return _predict_f(model, X_test, state; cov, diag)
end


function predict_y(
    model::AbstractGPModel, X_test::AbstractVector, state=nothing)
    return predict_y(likelihood(model), only(_predict_f(model, X_test, state; cov=false)))
end


function proba_y(
    model::AbstractGPModel, X_test::AbstractVector; state=nothing, diag=false, nSamples=1000, model_cov=true)
    μ_f, Σ_f = _predict_f(model, X_test, state; cov=model_cov, diag=diag)

    if model_cov
        return (compute_proba_full(model.likelihood, μ_f, Σ_f, nSamples), (μ_f, Σ_f))
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
