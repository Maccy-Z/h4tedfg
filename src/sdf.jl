# using Zygote
#
# # Define a mutable struct with some fields
#
#
# mutable struct MyParams
#     x::Float64
#     y::Float64
# end
#
# struct OuterStruct
#     inner_struct::MyParams
# end
#
# # Define a function that does some computation using the fields of MyParams
# function my_function(params)
#     # Some arbitrary computation involving params' fields
#     return 3 * params.inner_struct.x + 4 * params.inner_struct.y^2
# end
#
# # Now, we'll compute the gradient of 'my_function' with respect to the fields of 'params'
# params = OuterStruct(MyParams(2.0, 3.0))  # Our initial parameters
#
# # Compute the gradient
# grads = Zygote.gradient(params -> my_function(params), params)
#
# # 'grads' is a tuple containing a struct of the same type as 'params'
# # The struct contains the gradients of the function with respect to each field in 'params'
# println(grads)  # Should display something like: MyParams(3.0, 24.0)#=
#

a = (0, 1)
b = (0, 1, 2)

for (c, d) in zip(a, b)
    println(c, d)
end
