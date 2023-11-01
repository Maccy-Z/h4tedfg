using KernelFunctions

function evaluate_rbf_kernel(x, y, kernel)
    return kernel(x, y)
end

function main()
    # Define the RBF kernel with a given length scale
    sigma = 1.0
    k = sigma * with_lengthscale(SqExponentialKernel(), 0.2)
    println(fieldnames(typeof(k.kernel)))
    println()
    println(k.σ²)
    println(k.kernel.transform.s)
    println(" ")
    # 2-dimensional test values
    x_vals = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    y_vals = [[1., 1.], [1., 1.], [1., 1.]]

    for (x, y) in zip(x_vals, y_vals)
        result = evaluate_rbf_kernel(x, y, k)
        println("k($(x[1]), $(x[2])), ($(y[1]), $(y[2]))) = $result")
    end
end

main()