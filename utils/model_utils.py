from torch import no_grad
from torch.nn import BCELoss
from torch.optim import Adam


def get_accuracy(model, dataloader, device) -> float:
    model.eval()

    n_correct = 0
    n_samples = 0

    with no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            pred = model(data).max(1)[1]
            n_correct += (pred == labels).sum().item()
            n_samples += pred.shape[0]

    return n_correct / n_samples


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs: int,
    lr: float,
    print_on_batch: int = 50,
) -> None:
    # define an optimizer and a loss function
    optim = Adam(model.parameters(), lr=lr)
    loss_func = BCELoss()

    for epoch in range(num_epochs):
        print(f"Epoch #{epoch + 1}")

        model.train()

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optim.zero_grad()
            output = model(data)
            loss = loss_func(output, labels)
            loss.backward()
            optim.step()

            if (batch_idx + 1) % print_on_batch == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}")
                print(f"Loss={loss.item()}")

        train_acc = get_accuracy(model, train_loader, device)
        test_acc = get_accuracy(model, test_loader, device)

        print(f"\nAccuracy on train: {100*train_acc:.2f}")
        print(f"Accuracy on test: {100*test_acc:.2f}\n")
