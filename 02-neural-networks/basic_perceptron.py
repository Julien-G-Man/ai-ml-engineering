import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perceptron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    output = sigmoid(weighted_sum)
    return output


def main():
    # Features: [hoursof_sleep, free_time, energy]
    person_features = np.array([7, 1, 0.4])
    weights = np.array([0.3, 0.5, 0.7])
    bias = -0.5

    prediction = perceptron(person_features, weights, bias)
    print(f"Probability: {prediction:.2f}")
    

if __name__ == "__main__":
    main()