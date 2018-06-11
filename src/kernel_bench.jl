include("kernels.jl)
a = CuArray(rand(Float32, (3, 4)))
b = CuArray(zeros(Float32, (3, 4)))

display(a)
softmax!(a, b)
display(b)

a = CuArray(rand(Float32, (16384, 64*32)));
b = CuArray(zeros(Float32, (16384, 64*32)));
@btime begin softmax!(a, b); synchronize() end
# 795.205 μs (24 allocations: 704 bytes)
