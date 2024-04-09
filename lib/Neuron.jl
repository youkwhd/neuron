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

    DESIRED_OUTPUT = 1

    function calculate_loss(nn :: Network)
        losses = map(neuron -> -(log(neuron) * DESIRED_OUTPUT), last(nn.layers).neurons)
        loss_avg = Statistics.mean(losses)
        println("Loss: $loss_avg")
    end

    function forward_prop(nn :: Network, activation_fn :: Function = x -> x)
        for i in 2:length(nn.layers)
            product = reshape(nn.layers[i - 1].neurons, (1, :)) * nn.layers[i - 1].weights
            nn.layers[i].neurons = map(activation_fn, vec(product))
        end
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
