import random
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from .data_loader import get_dataloaders
from .model import get_model
from .config import SAVE_PATH

def test_random_image():
    _, _, train_dataset, test_dataset = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=len(train_dataset.classes)).to(device)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    idx = random.randint(0, len(test_dataset)-1)
    image, true_label = test_dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)

    pil_image = ToPILImage()(image.cpu())
    class_names = test_dataset.classes

    plt.imshow(pil_image)
    plt.title(f"True: {class_names[true_label]} | Pred: {class_names[predicted_label.item()]}")
    plt.axis("off")
    plt.show()
