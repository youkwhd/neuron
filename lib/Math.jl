module Math

ReLU(x) = max(0, x)

# x == 0 should be an anomaly,
# but for the sake of simplicty just interpret 0 to be the result
# 
# see: https://stats.stackexchange.com/a/333400
ReLU_derivative(x) = Int(x > 0)

sigmoid(x) = 1 / (1 + exp(-x))
σ = sigmoid

sigmoid_derivative(x) = sigmoid_derivativef(sigmoid(x))
sigmoid_derivativef(x) = x * (1 - x)
σd = sigmoid_derivative
σdf = sigmoid_derivativef

end
