module Math

sigmoid(x) = 1 / (1 + exp(-x))
σ = sigmoid

sigmoid_derivative(x) = sigmoid_derivativef(sigmoid(x))
sigmoid_derivativef(x) = x * (1 - x)
σd = sigmoid_derivative
σdf = sigmoid_derivativef

end
