# You are not allowed to import any other libraries or modules.

import torch
import torch.nn as nn
import numpy as np

def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    model.to(device)
    avg_train_loss, avg_train_acc = [], []

    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))

        print(f'\nEpoch [{epoch}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model

def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    batch_train_loss, batch_train_acc = [], []

    for batch in train_dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Compute accuracy
        predicted = outputs.argmax(dim=1)
        true_labels = targets.argmax(dim=1)
        correct = (predicted == true_labels).sum().item()
        accuracy = correct / targets.size(0)

        batch_train_loss.append(loss.item())
        batch_train_acc.append(accuracy)

        # Compute gradient of loss w.r.t. outputs
        delta = 2 * (outputs - targets) / outputs.size(0)  # Gradient of MSELoss

        # Backward pass
        for layer in reversed(list(model.children())):
            delta = layer.backward(delta)

        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

    return batch_train_loss, batch_train_acc

def test(model, test_dataloader, device):
    model.to(device)
    model.eval()
    batch_test_acc = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            predicted = outputs.argmax(dim=1)
            true_labels = targets.argmax(dim=1)
            correct = (predicted == true_labels).sum().item()
            accuracy = correct / targets.size(0)

            batch_test_acc.append(accuracy)

    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")