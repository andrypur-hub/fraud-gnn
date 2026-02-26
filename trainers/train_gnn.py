import torch
import torch.nn.functional as F

def train_gnn(model, data, train_mask, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    y = data.y.float()

    pos_weight = (y[train_mask]==0).sum()/(y[train_mask]==1).sum()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        loss = F.binary_cross_entropy_with_logits(
            out[train_mask],
            y[train_mask],
            pos_weight=pos_weight
        )

        loss.backward()
        optimizer.step()

    return model
