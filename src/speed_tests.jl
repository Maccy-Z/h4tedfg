using Random

n = 40000

k = 100000

glob_idx = 1

test_arr = rand(k)

println("Fast resample")

for b in 1:10
    i = rand(1:k)

    @time begin
        arr = mod.(i:i+n-1, k+1)
        b = test_arr[arr]
        b = b .+ 1.
    end

end

println("Shuffle")

for b in 1:10

    @time begin
        b = Random.shuffle(test_arr)
        b = b .+ 1.
    end

end
