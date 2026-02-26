import torch
import torch.nn.functional as F


def train_gnn(model, data, train_mask, epochs=50, lr=0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # class imbalance weighting
    labels = data.y[train_mask]
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()

    weight = torch.tensor([1.0, num_neg / num_pos])  # fraud lebih berat
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        out = model(data)  # [N,2]

        loss = criterion(out[train_mask], data.y[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model
