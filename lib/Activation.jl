module Activation

export ReLU, ReLU_derivative, LReLU,
       LReLU_derivative, sigmoid, σ,
       sigmoid_derivative, sigmoid_derivativef,
       σd, σdf 

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

end # module
