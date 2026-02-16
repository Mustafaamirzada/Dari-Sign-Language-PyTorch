from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataFolder(Dataset):
    """
    Docstring for CustomDataFolder
    This class is for creating our own custom Dataset ImageFolder class.
    It extends from the torch.utils.data.Dataset class which has all properities of that class
    
    Args:
      root_data_dir: Path to dataset directory.
      transform: a torchvision.transforms.v2 or torchvision.transforms which transform the dataset.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = sorted(
            [d.name for d in os.scandir(root) if d.is_dir()],
            key=lambda x: int(x)
        )
        # sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes) }

        for cls in self.classes:
            folder = os.path.join(root, cls)
            for file in os.listdir(folder):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(folder, file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label
