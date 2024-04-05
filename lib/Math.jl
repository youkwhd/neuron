module Math

sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_derivative(x) = sigmoid_derivativef(sigmoid(x))
sigmoid_derivativef(x) = x * (1 - x)

end
