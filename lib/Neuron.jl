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

        predict(nn, activation_fn)
        __loss = sum(map(Loss.MSE, last(nn.layers).neurons, expected_output))
        adjust(nn, activation_derivative_fn, expected_output)

        return __loss
    end

    module Loss
        function MSE(output :: Float64, expected :: T) where T<:Number
            return (output - expected) ^ 2
        end

        function MSE_derivative(output :: Float64, expected :: T) where T<:Number
            return 2 * (output - expected)
        end
    end

    function predict(nn :: Network, activation_fn :: Function)
        for i in 2:length(nn.layers)
            product = reshape(nn.layers[i - 1].neurons, (1, :)) * nn.layers[i - 1].weights
            nn.layers[i].neurons = map(activation_fn, vec(product))
        end
    end

    function adjust(nn :: Network, activation_derivative_fn :: Function, expected_output :: Vector{T}) where T<:Number
        # TODO: refactor learning_rate
        __LEARNING_RATE = 0.1

        for i in range(length(nn.layers), 2, step=-1)
            for j in 1:length(nn.layers[i].neurons)
                __loss = sum(map(Loss.MSE_derivative, last(nn.layers).neurons, expected_output))
                learn_by = __loss * activation_derivative_fn(nn.layers[i].neurons[j])

                nn.layers[i - 1].weights[j, :] = map(weight -> weight - (__LEARNING_RATE * learn_by), nn.layers[i - 1].weights[j, :])
            end
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
