# nn-assignment-2
# Neural Network Backpropagation Implementation

## Overview
This repository contains an implementation of a simple **Neural Network (NN) with Backpropagation** using PyTorch. The network consists of:
- An **input layer** with 2 neurons
- A **hidden layer** with 2 neurons
- An **output layer** with 2 neurons
- Sigmoid activation function

The network is initialized with predefined weights and biases and performs a **single iteration** of backpropagation to update the parameters.

## Files
- **`nn_backprop.py`**: Python script implementing the forward and backward pass of the neural network.
- **`output.txt`**: The recorded output from running the script.
- **`README.md`**: This file, explaining the repository contents and functionality.

## Implementation Details
- The model is implemented using `torch.nn.Module`.
- The forward pass applies **sigmoid activation** after each layer.
- The loss function used is **Mean Squared Error (MSE)**.
- **Manual weight updates** are performed using the backpropagation algorithm.

## Setup & Execution
### Prerequisites
Ensure you have Python and PyTorch installed. You can install PyTorch using:
```bash
pip install torch
```

### Running the Script
To execute the neural network and observe the output:
```bash
python nn_backprop.py
```

The output will display the loss before and after weight updates, along with the updated weights and biases.

## Example Output
```
Initial Loss: 0.2983
Updated Hidden Layer Weights:
[[0.1490, 0.1990],
 [0.2495, 0.2995]]
Updated Hidden Layer Biases:
[0.3490, 0.3490]
Updated Output Layer Weights:
[[0.3990, 0.4490],
 [0.4995, 0.5495]]
Updated Output Layer Biases:
[0.5990, 0.5990]
```

## Contact
If you have any questions or need further assistance, feel free to reach out!

---
🚀 **Happy Coding**

