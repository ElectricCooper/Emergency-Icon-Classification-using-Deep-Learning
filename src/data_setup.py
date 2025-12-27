""" Module for setting up data loaders for our image classification task"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def create_dataloaders(data_dir, batch_size=32):
    """
    Loads and splits the dataset. Returns the dataloaders
    """

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(
                    root=data_dir,
                    transform=data_transform
                )

    class_names = full_dataset.classes

    total_count = len(full_dataset)
    train_count = int(0.8 * total_count)
    val_count = int(0.1 * total_count)
    test_count = total_count - train_count - val_count

    train_data, val_data, test_data = random_split(
                                        full_dataset,
                                        [train_count, val_count, test_count]
                                    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names
