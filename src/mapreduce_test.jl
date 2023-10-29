#=
mapreduce_test:
- Julia version: 
- Author: maccyz
- Date: 2023-10-29
=#
function add(x, y)
    return x + y
end

function mapreduce_test()
    x = [1, 2, 3]
    y = [4, 5]
    z = mapreduce(add, +, x, y, )
    println(z)
end

mapreduce_test()