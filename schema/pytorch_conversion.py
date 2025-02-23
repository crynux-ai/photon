import torch

# Load model definition
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Load the `.pth` checkpoint
checkpoint = torch.load("models/model.pth", map_location="cpu")
print(checkpoint.keys())
print(checkpoint["model_state_dict"])

# Initialize the model and load weights
#model = MyModel()
#model.load_state_dict(checkpoint["model_state_dict"])  # Adjust key based on checkpoint structure
#model.eval()

# Convert to TorchScript
#scripted_model = torch.jit.script(model)  # or use `torch.jit.trace(model, example_input)`
#scripted_model.save("model.pt")  # Save as TorchScript format
