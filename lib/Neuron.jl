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

        Network(layers) = begin 
            @test length(layers) >= 2
            new(layers)
        end
    end

    function train(nn :: Network, input :: Vector{T}, expected_output :: Vector{T}, activation_fn :: Function, activation_derivative_fn :: Function) where T<:Number
        first(nn.layers).neurons = input

        forward(nn, activation_fn)
        __loss = loss(nn, expected_output)
        adjust(nn, activation_derivative_fn, expected_output)

        return __loss
    end

    # calculate loss using the Mean Squared Error (MSE) method
    function loss(nn :: Network, expected_output :: Vector{T}) where T<:Number
        expected_output_length = length(expected_output)
        @test expected_output_length == length(last(nn.layers).neurons)

        losses = map((neuron, expected) -> (expected - neuron) ^ 2, last(nn.layers).neurons, expected_output)
        return sum(losses)
    end

    function forward(nn :: Network, activation_fn :: Function)
        for i in 2:length(nn.layers)
            product = reshape(nn.layers[i - 1].neurons, (1, :)) * nn.layers[i - 1].weights
            nn.layers[i].neurons = map(activation_fn, vec(product))
        end
    end

    function adjust(nn :: Network, activation_derivative_fn :: Function, expected_output :: Vector{T}) where T<:Number
        # TODO: refactor learning_rate
        __LEARNING_RATE = 0.1

        der_losses = map((neuron, expected) -> 2 * (neuron - expected), last(nn.layers).neurons, expected_output)
        der_loss = sum(der_losses)

        for i in range(length(nn.layers), 2, step=-1)
            der_act = sum(map(activation_derivative_fn, nn.layers[i].neurons))
            learn_by = der_act * der_loss

            nn.layers[i - 1].weights = map(weight -> weight - (learn_by * __LEARNING_RATE), nn.layers[i - 1].weights)
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
