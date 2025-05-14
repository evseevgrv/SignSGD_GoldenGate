import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt
from parameter_free_signsgd import AIDsignSGD
import functools
from utils import training_utils

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = "cuda"

# Generate synthetic data
def generate_synthetic_data(n_samples=1000):
    # Generate random input features
    X = np.random.randn(n_samples, 2)
    
    # Generate target values (simple nonlinear function)
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(np.float32)
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, device=device, dtype=torch.float)
    y = torch.tensor(y, device=device, dtype=torch.float).reshape(-1, 1)
    
    return X, y

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Training function
def train_model(model, X, y, optimizer, criterion, n_epochs=100):
    losses = []
    lrs = []
    
    for epoch in range(n_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        closure = None
        if not epoch:
            closure = functools.partial(optimizer._get_loss, loss=loss)
        optimizer.step(closure)
        with torch.no_grad():
            state = optimizer.state[next(model.parameters())]
            print("epoch: ", epoch, state["prev_grad"].flatten()[:3], state["prev_param"].flatten()[:3], optimizer.param_groups[0]["prev_gamma"])
        # Store loss
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
            
    return losses, lrs

# Generate data
X, y = generate_synthetic_data()

# Initialize model
model = SimpleNN().to(device)

max_lr = 1e-2

l_inf = 100.0
lower_bound=0.0
d_0=None
weight_decay=0.0
clamp_level=max_lr
update_gap = 1
num_training_steps=1000
warmup_steps=int(0.1*num_training_steps)
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = AIDsignSGD(
    model.parameters(), 
    L_inf=l_inf,
    lower_bound=lower_bound,
    d_0=d_0,
    weight_decay = weight_decay,
    clamp_level=clamp_level,
    update_gap=update_gap,
    foreach=False,
    fused=False,
    differentiable=False,

    lr=max_lr,
    warmup_steps=warmup_steps,
)

scheduler = training_utils.get_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    num_training_steps=num_training_steps,
    warmup_steps=warmup_steps,
    min_lr_ratio=0.1,
    cycle_length=num_training_steps,
)

# Train the model
losses, lrs = train_model(model, X, y, optimizer, criterion, n_epochs=num_training_steps)

import matplotlib.pyplot as plt
plt.plot(torch.stack(lrs).cpu().numpy())
plt.savefig('lrs.pdf')
plt.close()