import torch
from tqdm import tqdm
from .config import LR, EPOCHS, SAVE_PATH
from .model import get_model
from .data_loader import get_dataloaders

def train():
    train_loader, _, train_dataset, _ = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=len(train_dataset.classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        print(f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {correct/total:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved at {SAVE_PATH}")
