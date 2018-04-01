using Flux
using NNlib

struct Embedding
    W
end
Embedding(vocabsize, embedsize) = Embedding(param(randn(embedsize, vocabsize)))
(embedding::Embedding)(x) = embedding.W[:, x]
Flux.treelike(Embedding)

FeedForward(d_model, d_hidden) = Chain(Dense(d_model, d_hidden, relu), Dense(d_hidden, d_model))

struct ResidualBlock
    l
    norm
    # TODO dropout
end
ResidualBlock(l, d_model::Integer) = ResidualBlock(l, LayerNorm(d_model))
(l::ResidualBlock)(x...) = x[1] .+ l.norm(l.l(x...))
Flux.treelike(ResidualBlock)

struct Attention
    scale
    # TODO dropout
    # TODO causal
end
Attention(d_key::Integer, causal=false) = Attention(sqrt(d_key))
function (l::Attention)(q, k, v)
    alpha = k'*q
    # TODO causal mask
    return v*softmax(alpha, 1)
end
Flux.mapchildren(f, l::Attention) = Attention(l.scale)

# TODO multihead

struct EncoderLayer
    selfattn
    feedforward
end
EncoderLayer(d_model::Integer, d_hidden::Integer) = EncoderLayer(
    ResidualBlock(Attention(d_model), d_model),
    ResidualBlock(FeedForward(d_model, d_hidden), d_model))
(l::EncoderLayer)(x) = l.feedforward(l.selfattn(x, x, x))
Flux.treelike(EncoderLayer)

struct Encoder
    embed
    layers
end
Encoder(d_model, d_hidden, n_layers, vocabsize) = Encoder(
    Embedding(vocabsize, d_model),
    Chain((EncoderLayer(d_model, d_hidden) for _ in 1:n_layers)...))
function (l::Encoder)(x)
    x = posenc!(l.embed(x))
    encoding = [x = l(x) for l in l.layers.layers]
    return encoding
end
Flux.treelike(Encoder)
