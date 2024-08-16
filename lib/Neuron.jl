module Neuron
    using Activation
    using Test

    export Layer, Network, train, predict, VERBOSE

    VERBOSE = Ref(false)

    mutable struct Layer
        neurons :: Vector{Float64}
        biases :: Vector{Float64}
        weights :: Matrix{Float64}

        Layer(len :: Int) = new(Vector{Float64}(undef, len), Vector{Float64}(undef, len), Matrix{Float64}(undef, len, 1))
        Layer(len :: Int, weights) = begin
            nrows, _ = size(weights)
            @test nrows == len

            new(Vector{Float64}(undef, len), Vector{Float64}(undef, len), weights)
        end

        Layer(neurons) = new(neurons, Vector{Float64}(undef, length(neurons)), Matrix{Float64}(undef, length(neurons), 1))
        Layer(neurons, weights) = begin 
            nrows, _ = size(weights)
            @test nrows == length(neurons)

            new(neurons, Vector{Float64}(undef, length(neurons)), weights)
        end
    end

    mutable struct Network
        layers :: Array{Layer}

        Network(layers) = begin 
            @test length(layers) >= 2
            network = new(layers)
            randomize(network)
            return network
        end
    end

    function train(nn :: Network, dataset :: Vector{Vector{Vector{Int64}}}; epoch=30000)
        dataset = convert(Vector{Vector{Vector{Float64}}}, dataset)

        for _ = 1:epoch
            adjust(nn, dataset)

            if VERBOSE[]
                println("Cost: ", cost(nn, dataset))
            end
        end

        if cost(nn, dataset) > 0.005
            if VERBOSE[]
                println("[WARN ::] ", "cost is too big, retraining..")
            end

            if cost(nn, dataset) >= 0.01
                if VERBOSE[]
                    println("[WARN ::] ", "randomizing weights and biases..")
                end

                randomize(nn)
            end

            train(nn, convert(Vector{Vector{Vector{Int64}}}, dataset))
        end
    end

    function predict(nn :: Network, input :: Vector{Float64})
        convert(Vector{Int}, map(round, forward(nn, input)))
    end

    function cost(nn :: Network, dataset :: Vector{Vector{Vector{Float64}}})
        result = 0

        for data in dataset
            input = data[1]
            expected = data[2]

            pred = forward(nn, input)
            distance = sum(pred - expected)
            result += distance * distance
        end

        result /= length(dataset)
        return result
    end

    function forward(nn :: Network, input :: Vector{Float64}; activation_fn :: Function = Activation.σ)
        first(nn.layers).neurons = input

        for i = 2:length(nn.layers)
            product = reshape(nn.layers[i - 1].neurons, (1, :)) * nn.layers[i - 1].weights
            product = vec(product) + nn.layers[i].biases
            nn.layers[i].neurons = map(activation_fn, product)
        end

        return last(nn.layers).neurons
    end

    function adjust(nn :: Network, dataset :: Vector{Vector{Vector{Float64}}}; learning_rate=0.10)
        slide = 0.1
        old_cost = cost(nn, dataset)
        mutated = deepcopy(nn)

        for i = 2:length(nn.layers)
            for j = 1:ndims(nn.layers[i - 1].weights)
                for k = 1:length(nn.layers[i - 1].weights[j, :])
                    temp = nn.layers[i - 1].weights[j, k]
                    nn.layers[i - 1].weights[j, k] += slide
                    mutated.layers[i - 1].weights[j, k] = (cost(nn, dataset) - old_cost) / slide
                    nn.layers[i - 1].weights[j, k] = temp
                end
            end

            for j = 1:length(nn.layers[i].biases)
                temp = nn.layers[i].biases[j]
                nn.layers[i].biases[j] += slide
                mutated.layers[i].biases[j] = (cost(nn, dataset) - old_cost) / slide
                nn.layers[i].biases[j] = temp
            end
        end

        for i = 2:length(nn.layers)
            for j = 1:ndims(nn.layers[i - 1].weights)
                for k = 1:length(nn.layers[i - 1].weights[j, :])
                    nn.layers[i - 1].weights[j, k] -= learning_rate * mutated.layers[i - 1].weights[j, k]
                end
            end

            for j = 1:length(nn.layers[i].biases)
                nn.layers[i].biases[j] -= learning_rate * mutated.layers[i].biases[j]
            end
        end
    end

    function randomize(nn :: Network)
        for i = 1:(length(nn.layers) - 1)
            # nn.layers[i].weights = 
            #     rand(-5:5,
            #          (length(nn.layers[i].neurons),
            #           length(nn.layers[i + 1].neurons)))

            nn.layers[i].weights = 
                rand(Float64,
                     (length(nn.layers[i].neurons),
                      length(nn.layers[i + 1].neurons))) * 2 .- 1
        end

        for i = 2:length(nn.layers)
            # nn.layers[i].biases = rand(-1:1, length(nn.layers[i].neurons))
            nn.layers[i].biases = rand(Float64, length(nn.layers[i].neurons)) * 2 .- 1
        end
    end
end
