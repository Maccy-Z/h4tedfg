
from julia.api import Julia

jl = Julia(runtime='/home/maccyz/julia-1.9.3/bin/julia')
from julia import Main

Main.include("sdf.jl")
x = Main.make_gaussian(2, 3)

Main.show_struct(x)
# result = Main.sdf(1, 2, 3)0
