using Plots

# Generate some data
x = 1:10  # This creates an iterable for numbers 1 through 10.
y = rand(10)  # This creates an array of 10 random numbers.

# Create a simple line plot
plot(x, y, title="Sample Plot", xlabel="x", ylabel="y", linewidth=2, legend=false)

# Show the plot
# Depending on your environment (Jupyter, REPL, script), this might be optional or necessary.
display(plot(x, y))