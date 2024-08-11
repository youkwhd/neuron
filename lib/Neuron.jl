module Neuron
    import Statistics
    using Test

    ReLU(x) = max(0, x)

    # x == 0 should be an anomaly,
    # but for the sake of simplicty just interpret 0 to be the result
    # 
    # see: https://stats.stackexchange.com/a/333400
    ReLU_derivative(x) = Int(x > 0)

    # Leaky ReLU
    LReLU(x) = max(0.01, x)
    LReLU_derivative(x) = x > 0 ? 1 : 0.01

    sigmoid(x) = 1 / (1 + exp(-x))
    σ = sigmoid

    sigmoid_derivative(x) = sigmoid_derivativef(sigmoid(x))
    sigmoid_derivativef(x) = x * (1 - x)
    σd = sigmoid_derivative
    σdf = sigmoid_derivativef

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

    mutable struct Network
        layers :: Array{Layer}

        Network(layers) = begin 
            @test length(layers) >= 2
            network = new(layers)
            randomize_weights(network)
            return network
        end
    end

    # function train(nn :: Network, input :: Vector{T}, expected_output :: Vector{T}, activation_fn :: Function, activation_derivative_fn :: Function) where T<:Number
    #     first(nn.layers).neurons = input

    #     predict(nn, activation_fn)
    #     adjust(nn, activation_derivative_fn, expected_output)

    #     __loss = sum(map((neuron, expected) -> 1 / length(last(nn.layers).neurons) * Loss.MSE(neuron, expected), last(nn.layers).neurons, expected_output))
    #     return __loss
    # end

    function train(nn :: Network, dataset :: Vector{Vector{Vector{Int64}}}; epoch=4000)
        mutated = deepcopy(nn)
        randomize_weights(mutated)
        best_loss = 0

        for data in dataset
            input = data[1]
            expected = data[2]

            predict(mutated, convert(Vector{Float64}, input))
            best_loss += expected != convert(Vector{Int64}, map(round, last(mutated.layers).neurons))
        end

        # for _ = 1:epoch
        while best_loss != 0
            loss = 0
            next = deepcopy(mutated)
            randomize_weights(next)

            for data in dataset
                input = data[1]
                expected = data[2]

                predict(next, convert(Vector{Float64}, input))
                loss += expected != convert(Vector{Int64}, map(round, last(next.layers).neurons))
            end

            if loss < best_loss
                best_loss = loss
                mutated = next
            end
        end

        nn.layers = mutated.layers
    end

    module Loss
        function MSE(output :: Float64, expected :: T) where T<:Number
            return (output - expected) ^ 2
        end

        function MSE_derivative(output :: Float64, expected :: T) where T<:Number
            return 2 * (output - expected)
        end
    end

    function predict(nn :: Network, input :: Vector{Float64}; activation_fn :: Function = ReLU)
        first(nn.layers).neurons = input

        for i in 2:length(nn.layers)
            product = reshape(nn.layers[i - 1].neurons, (1, :)) * nn.layers[i - 1].weights
            nn.layers[i].neurons = map(activation_fn, vec(product))
        end

        return last(nn.layers).neurons
    end

    function adjust(nn :: Network, activation_derivative_fn :: Function, expected_output :: Vector{T}) where T<:Number
        # TODO: refactor learning_rate
        __LEARNING_RATE = 0.1
        __loss = sum(map(Loss.MSE_derivative, last(nn.layers).neurons, expected_output))

        for i in range(length(nn.layers), 2, step=-1)
            for j in 1:length(nn.layers[i].neurons)
                learn_by = __loss * activation_derivative_fn(nn.layers[i].neurons[j])

                for k in 1:length(nn.layers[i - 1].neurons)
                    nn.layers[i - 1].weights[k, :] = map(weight -> weight - (__LEARNING_RATE * learn_by), nn.layers[i - 1].weights[k, :])
                end
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
