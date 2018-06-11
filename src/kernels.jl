using CUDAdrv
using CUDAnative
using BenchmarkTools

const MAX_THREADS = 256 # seems to work best (1024 max)
const MAX_BLOCKS = 2^31 - 1 # benchmark only exercises 2048

macro reduce_block(f, ident, accid=1, g=:(val))
    quote
        acc = @cuDynamicSharedMem(
            T, blockDim().x, ($accid-1) * blockDim().x * sizeof(T))
        acc[threadIdx().x] = $ident
        # serial reduction by rows ÷ threads
        for rowbase in 0:blockDim().x:(rows-1)
            row = rowbase + threadIdx().x
            row > rows && return
            idx = row + (col-1) * rows
            mask_val = T(1)
            if mask !== nothing
                mask_idx = idx
                if broadcast
                    mask_idx = Broadcast.newindex(
                        mask, CartesianIndices(arr)[mask_idx])
                end
                mask_val = mask[mask_idx]
            end
            val = arr[idx]
            val = ifelse(mask_val == T(0), $ident, $g)
            acc[threadIdx().x] = $f(acc[threadIdx().x], val)
        end
        # block-parallel reduction by threads
        sync_threads()
        len = blockDim().x
        while len != 1
            sync_threads()
            skip = (len + 1) >> 1
            if threadIdx().x <= (len >> 1)
                acc[threadIdx().x] = $f(acc[threadIdx().x],
                                        acc[threadIdx().x + skip])
            end
            len = (len + 1) >> 1
        end
        sync_threads()
        acc[1]
    end |> esc
end

macro colwise(kernel)
    quote
        rows = size(out, 1) # "rows" is the first dimension
        cols = length(out) ÷ rows # "cols" are dimensions 2:end
        broadcast = (mask !== nothing && size(out) != size(mask))
        for colbase in 0:gridDim().x:(cols-1)
            col = colbase + blockIdx().x
            col > cols && return
            $kernel
        end
    end |> esc
end

macro elwise(f)
    quote
        for rowbase in 0:blockDim().x:(rows-1)
            row = rowbase + threadIdx().x
            row > rows && return
            idx = row + (col-1) * rows
            val = arr[idx]
            out[idx] = $f
        end
    end |> esc
end

function softmax_kernel(arr::CuDeviceArray{T}, out::CuDeviceArray{T},
                        mask::Union{CuDeviceArray{T}, Nothing}) where {T}
    @colwise begin
        colmax = @reduce_block(max, -typemax(T), 1)
        colsum = @reduce_block(
            +, T(0), 2, CUDAnative.exp_fast(val - colmax))
        @elwise CUDAnative.exp_fast(val - colmax) / colsum
    end
end

function normalize_kernel(arr::CuDeviceArray{T}, out::CuDeviceArray{T},
                          mask::Union{CuDeviceArray{T}, Nothing}) where {T}
    @colwise begin
        colmean = @reduce_block(+, T(0), 1) / rows
        colstd = CUDAnative.sqrt(
            @reduce_block(+, T(0), 2, (val - colmean)^2) / rows)
        @elwise (val - colmean) / (colstd + eps)
    end
end

function softmax!(arr::CuArray{T}, out::CuArray{T};
                  mask::Union{CuArray{T}, Nothing}=nothing) where {T}
    #device!(out.device)
    rows = size(out, 1)
    cols = length(out) ÷ rows
    blks = min(MAX_BLOCKS, cols)
    thrds = min(MAX_THREADS, rows)
    shared = sizeof(T) * thrds * 2
    @cuda blocks=blks threads=thrds shmem=shared softmax_kernel(
        arr, out, mask)
end

softmax(arr; mask) = softmax!(arr, similar(arr); mask=mask)
