import torch
import h5py

# Dummy PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

model = MyModel()
dummy_input = torch.randn(10)
output = model(dummy_input)

# Save weights to HDF5
with h5py.File("data/model_weights.h5", "w") as hf:
    for name, param in model.state_dict().items():
        hf.create_dataset(name, data=param.cpu().numpy())
