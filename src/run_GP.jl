using Plots
using Distributions
using AugmentedGaussianProcesses
# using Flux
using Printf
#
Base.show(io::IO, f::Float64) = @printf(io, "%1.2f", f)


# Make data
n_data = 100
n_dim = 2
n_grid = 100
minx = -2.5;
maxx = 3.5;
σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
n_class = n_dim + 1;

function generate_mixture_data(σ)
    centers = zeros(n_class, n_dim)
    # Create equidistant centers
    for i in 1:n_dim
        centers[i, i] = 1.0
    end
    centers[end, :] .= (1 + sqrt(n_class)) / n_dim
    centers ./= sqrt(n_dim)
    # Generate distributions with desired noise
    distr = [MvNormal(centers[i, :], σ) for i in 1:n_class]
    X = zeros(Float64, n_data, n_dim)
    y = zeros(Int64, n_data)
    for i in eachindex(y)
        y[i] = rand(1:n_class)
        X[i, :] = rand(distr[y[i]])
    end
    return X, y
end

function plot_data(X, Y, σ)
    p = Plots.plot(size(300, 500); lab="", title="sigma = $σ")
    Plots.scatter!(eachcol(X)...; group=Y, msw=0.0, lab="")
    return p
end

plot([plot_data(generate_mixture_data(σ)..., σ) for σ in σs]...)

# Make and train model 
models = Vector{AbstractGPModel}(undef, length(σs))
kernel = 1.0 * SqExponentialKernel() ∘ ScaleTransform(0.5)
for (i, σ) in enumerate(σs)
    @info "Training with data with noise $σ"
    X, y = generate_mixture_data(σ)

	println(y[1])

    m = VGP(
	    X, 
	    y,
        kernel,
        LogisticSoftMaxLikelihood(n_class),
        AnalyticVI(),
        optimiser=Optimisers.Descent(0.01),
		obsdim=1
    )

    m2, s2 = train!(m, 25)
    models[i] = m
    @warn "Main code finished"

    for e in m.f
        println()
		println("σ² = ", e.prior.kernel.kernel.σ²)
		println("1/r = ", e.prior.kernel.transform.s)
		# println(e.prior.kernel)
 		if e.prior.kernel.transform.s[1] > 10.
			e.prior.kernel.transform.s[1] = 10.
		end
		if e.prior.kernel.kernel.σ²[1] > 10.
			e.prior.kernel.kernel.σ²[1] = 10.
		end
		e.prior.kernel.kernel.σ²[1] = 10.
 		#e.prior.kernel.kernel.σ²[1] = 100
		# println(kernelmatrix(e.prior.kernel, X; obsdim=1))
	end
end

function compute_grid(model, n_grid=50)
    xlin = range(minx, maxx; length=n_grid)
    ylin = range(minx, maxx; length=n_grid)
    x_grid = Iterators.product(xlin, ylin)
	@info "Computing grid"
	a = vec(collect.(x_grid))
	println(size(a))
	println(size(a[1]))
    y_p, _ = proba_y(model, vec(collect.(x_grid)))
    y = predict_y(model, vec(collect.(x_grid)))
    return y_p, y, xlin, ylin
end;

function plot_contour(model, σ)
	@info "Plotting"
    n_grid = 50
    pred_proba, pred, x, y = compute_grid(model, n_grid)
	println(size(pred_proba))
	pred = reshape(pred, n_grid, n_grid)

    colors = [
            RGB([pred_proba[i, model.likelihood.ind_mapping[j]] for j in 1:n_class]...) for
            i in 1:(n_grid^2)
	]
	colors = reshape(colors,
        n_grid,
        n_grid,
    ) # Convert the predictions into an RGB array
	println(colors[1])
    Plots.contour(
        x,
        y,
        colors;
        cbar=false,
        fill=false,
        color=:black,
        linewidth=2.0,
        title="sigma = $σ",
    )
    return Plots.contour!(
        x,
        y,
        pred;
        clims=(0, 100),
        colorbar=false,
        color=:gray,
        levels=10,
    )
end;

h = Plots.plot(plot_contour.(models, σs)...)
display(h)