
i = 12345
j = 250000
k = 200000

for i in 1:5
    @time [(i + j - 1) % k + 1 for j in 0:(n-1)]
end