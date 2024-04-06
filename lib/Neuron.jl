module Neuron
    using Test

    mutable struct Layer
        neurons :: Vector{Int8}
        weights :: Matrix{Int8}

        Layer(len :: Int) = new(Vector{Int8}(undef, len), Matrix{Int8}(undef, len, 1))
        Layer(neurons) = new(neurons, Matrix{Int8}(undef, length(neurons), 1))
        Layer(neurons, weights) = begin 
            nrows, _ = size(weights)
            @test nrows == length(neurons)

            new(neurons, weights)
        end
    end

    struct Network
        layers :: Array{Layer}

        Network(layers) = new(layers)
    end
end
