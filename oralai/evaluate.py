import torch
from .data_loader import get_dataloaders
from .model import get_model
from .config import SAVE_PATH

def evaluate():
    _, test_loader, train_dataset, _ = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=len(train_dataset.classes)).to(device)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, "
          f"Test Accuracy: {correct/total:.4f}")
