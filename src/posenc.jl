using GPUArrays
using Minibatch

function posenc(timestep::Integer, channel::Integer, nchannels::Integer)
    if iseven(channel)
        return sin(timestep/(10000^(channel/nchannels)))
    else
        return cos(timestep/(10000^((channel-1)/nchannels)))
    end
end

function posenc(idx::CartesianIndex, nchannels::Integer)
    return posenc(idx[2], idx[1], nchannels)
end

function posenc!(A::GPUArray, state, nchannels::Integer)
    idx = @cartesianidx A state
    @inbounds A[idx] += posenc(idx, nchannels)
    return A
end

function posenc!(A::GPUArray)
    nchannels = size(A, 1)
    gpucall(posenc, A, (nchannels,))
    return A
end

function posenc!(A::AbstractArray)
    nchannels = size(A, 1)
    for idx in CartesianRange(size(A))
        # @inbounds
        A[idx] += posenc(idx, nchannels)
    end
    return A
end

function posenc!(B::MaskedBatch)
    posenc!(B.data)
    B.data = B.data .* B.mask # TODO in-place
    return B
end

function posenc!(A::Flux.TrackedArray)
    return posenc!(Flux.Tracker.data(A))
end
