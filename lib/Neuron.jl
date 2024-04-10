module Neuron
    import Statistics
    using Test

    mutable struct Layer
        neurons :: Vector{Float64}
        weights :: Matrix{Float64}

        Layer(len :: Int) = new(Vector{Float64}(undef, len), Matrix{Float64}(undef, len, 1))
        Layer(len :: Int, weights) = begin
            nrows, _ = size(weights)
            @test nrows == len

            new(Vector{Float64}(undef, len), weights)
        end

        Layer(neurons) = new(neurons, Matrix{Float64}(undef, length(neurons), 1))
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

    # calculate loss using the Mean Squared Error (MSE) method
    function loss(nn :: Network, expected_output :: Vector{Int})
        output_neurons_len = length(last(nn.layers).neurons)
        @test length(expected_output) == output_neurons_len

        losses = map((neuron, expected) -> (expected - neuron) ^ 2,
                    last(nn.layers).neurons, expected_output)

        return (1 / output_neurons_len) * sum(losses)
    end

    function forward(nn :: Network, activation_fn :: Function = x -> x)
        for i in 2:length(nn.layers)
            product = reshape(nn.layers[i - 1].neurons, (1, :)) * nn.layers[i - 1].weights
            nn.layers[i].neurons = map(activation_fn, vec(product))
        end
    end

    function adjust(nn :: Network, activation_derivative_fn :: Function = x -> x)
    end

    # Randomizes weights between -1, 1
    function randomize_weights(nn :: Network)
        for i in 1:(length(nn.layers) - 1)
            nn.layers[i].weights = 
                rand(Float64,
                     (length(nn.layers[i].neurons),
                      length(nn.layers[i + 1].neurons))) * 2 .- 1
        end
    end
end
