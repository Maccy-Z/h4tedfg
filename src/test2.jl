using AbstractGPs
using ApproximateGPs
using ArraysOfArrays
using AugmentedGPLikelihoods
using Distributions
using LinearAlgebra
using SplitApplyCombine
using Plots

N = 200
Nclass = 3
x = collect(range(-10, 10; length=N));
X = MOInput(x, Nclass - 1);

liks = CategoricalLikelihood(LogisticSoftMaxLink(Nclass))
AugmentedGPLikelihoods.nlatent(::CategoricalLikelihood{<:LogisticSoftMaxLink}) = Nclass
SplitApplyCombine.invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')

# kernel function
kernel = 5.0 * with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)
fz = gp(x, 1e-8);

# Generate random targets
y = rand(1:3, N)
Ys = nestedview(sort(unique(y))[1:nlatent(liks)] .== permutedims(y))

# Plot sampled data
plt = plot(; title="Logistic-softmax")
plt = scatter!(plt, x, y; group=y, label=[1 2 3 4], msw=0.0)
    # plot!(plt, x, vcat(fs, [zeros(N)]); color=[1 2 3 4], label="", lw=3.0)

# CVAI Training code
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(copy(m), S)))
end

function cavi!(fz::AbstractGPs.FiniteGP, lik, x, Y, ms, Ss, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        posts_u = u_posterior.(Ref(fz), ms, Ss)
        posts_fs = marginals.([p_u(x) for p_u in posts_u])
        aux_posterior!(qΩ, lik, Y, SplitApplyCombine.invert(posts_fs))
        Ss .=
            inv.(
                Symmetric.(Ref(inv(K)) .+ Diagonal.(expected_auglik_precision(lik, qΩ, Y)))
            )
        ms .= Ss .* (expected_auglik_potential(lik, qΩ, Y) .- Ref(K \ mean(fz)))
    end
    return ms, Ss
end

println(kernel)

m = nestedview(zeros(N, nlatent(liks)))
S = [Matrix{Float64}(I(N)) for _ in 1:nlatent(liks)]
qΩ = init_aux_posterior(liks, N)
println("qΩ")
println(qΩ)
fz = gp(x, 1e-8)
m, S = cavi!(fz, liks, x, Ys, m, S, qΩ; niter=20)

println(size(m))


x_te = -10:0.01:10

for j in 1:nlatent(liks)
    plot!(
        plt,
        x_te,
        u_posterior(fz, m[j], S[j]);
        color=j,
        alpha=0.3,
        lw=3.0,
        label="",
    )
end


p_plts =plot()
scatter!(p_plts, x, y / Nclass; group=y, label=[1 2 3 4], msw=0.0)

lik_pred =
    liks.(
        invert(
            mean.([
                u_post(x_te) for u_post in u_posterior.(Ref(fz), m, S)
            ]),
        ),
    )
ps = getproperty.(lik_pred, :p)
lik_pred_x =
    liks.(
        invert(
            mean.([
                u_post(x) for u_post in u_posterior.(Ref(fz), m, S)
            ]),
        ),
    )
ps_x = getproperty.(lik_pred_x, :p)
# ps_true = getproperty.(lik_true.v, :p)
for k in 1:Nclass
    # plot!(p_plts[1], x, invert(ps_true)[k]; color=k, lw=2.0, label="")
    plot!(p_plts, x_te, invert(ps)[k]; color=k, lw=2.0, label="", ls=:dash)

end


# Gibbs sampling
function gibbs_sample(fz, lik, Y, fs, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fz)
    Σ = [zeros(N, N) for _ in 1:nlatent(lik)]
    μ = [zeros(N) for _ in 1:nlatent(lik)]
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, Y, invert(fs))
        Σ .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(auglik_precision(lik, Ω, Y))))
        μ .= Σ .* (auglik_potential(lik, Ω, Y) .- Ref(K \ mean(fz)))
        rand!.(MvNormal.(μ, Σ), fs)
        return copy(fs)
    end
end;


fs_init = nestedview(randn(N, nlatent(liks)))
Ω = init_aux_variables(liks, N)
# Run the sampling for default number of iterations (200)
samples =  gibbs_sample(fz, liks, Ys, fs_init, Ω)
# And visualize the samples overlapped to the variational posterior
# that we found earlier.



for fs in samples
    for j in 1:nlatent(liks)
        plot!(plt, x, fs[j]; color=j, alpha=0.07, label="")
    end
end

plot(plt)

display(current())


