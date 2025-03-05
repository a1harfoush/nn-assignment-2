import torch
import torch.nn as nn

class SecondAssignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.hidden_layer.weight.copy_(torch.tensor([[0.15, 0.20], [0.25, 0.30]]))
            self.hidden_layer.bias.copy_(torch.tensor([0.35, 0.35]))
            self.output_layer.weight.copy_(torch.tensor([[0.40, 0.45], [0.50, 0.55]]))
            self.output_layer.bias.copy_(torch.tensor([0.60, 0.60]))

    def forward(self, x):
        self.hidden_input = self.hidden_layer(x)
        self.hidden_output = torch.sigmoid(self.hidden_input)
        self.final_input = self.output_layer(self.hidden_output)
        self.final_output = torch.sigmoid(self.final_input)
        return self.final_output

model = SecondAssignment()
input_tensor = torch.tensor([0.05, 0.10])
target = torch.tensor([0.01, 0.99])
output = model(input_tensor)
loss = 0.5 * torch.sum((target - output) ** 2)
print(loss.item())

learning_rate = 0.5
delta_output = (output - target) * (output * (1 - output))

with torch.no_grad():
    for i in range(2):
        for j in range(2):
            model.output_layer.weight[i][j] -= learning_rate * delta_output[i] * model.hidden_output[j]
        model.output_layer.bias[i] -= learning_rate * delta_output[i]

delta_hidden = torch.zeros(2)
for i in range(2):
    delta_hidden[i] = (delta_output[0] * model.output_layer.weight[0][i] +
                       delta_output[1] * model.output_layer.weight[1][i]) * (model.hidden_output[i] * (1 - model.hidden_output[i]))

with torch.no_grad():
    for i in range(2):
        for j in range(2):
            model.hidden_layer.weight[i][j] -= learning_rate * delta_hidden[i] * input_tensor[j]
        model.hidden_layer.bias[i] -= learning_rate * delta_hidden[i]

print(model.hidden_layer.weight)
print(model.hidden_layer.bias)
print(model.output_layer.weight)
print(model.output_layer.bias)