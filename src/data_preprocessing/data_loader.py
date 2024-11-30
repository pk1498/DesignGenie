from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


def load_sun397_dataset(batch_size=32):
    dataset = load_dataset("1aurent/SUN397", split="train")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit models like ResNet
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize as per ImageNet
    ])

    def preprocess(example):
        image = Image.open(example["file"]).convert("RGB")
        example["image"] = transform(image)
        return example

    dataset = dataset.map(preprocess, remove_columns=["file"], batched=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader