module Transformer

include("posenc.jl")
include("layers.jl")

# 1. first try EncoderLayer on its own:
d = 4 # model size
t = 5 # timesteps
layer = EncoderLayer(d, 2d)
layer = Flux.mapleaves(Flux.Tracker.data, layer) # disable Tracker
x = rand(d, t)
layer(x)

# 2. then try Encoder
v = 10 # vocab size
n = 3 # number of layers
encoder = Encoder(d, 2d, n, v)
encoder = Flux.mapleaves(Flux.Tracker.data, encoder)
x = rand(1:v, t)
encoder(x)

# 3. try with real data and Minibatch.jl
include("data.jl")
encoder = Encoder(d, 2d, n, length(vocab_en))
encoder = Flux.mapleaves(Flux.Tracker.data, encoder)
x = en[1]
encoder(x)

end # module
