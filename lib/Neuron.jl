module Neuron
    struct Layer
        neurons :: Matrix{Int8}
        weights :: Matrix{Int8}

        Layer(nrows :: Int, ncols :: Int) = new(Matrix{Int8}(undef, nrows, ncols),
                                                Matrix{Int8}(undef, ncols, nrows))

        Layer(neurons, weights) = new(neurons, weights)
        Layer(neurons) = begin
            nrows, ncols = size(neurons)
            new(neurons, Matrix{Int8}(undef, ncols, nrows))
        end
    end

    struct Network
        layers :: Array{Layer}

        Network(layers) = new(layers)
    end
end
