"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import torch
import torchvision.transforms.v2 as T
from timeit import default_timer as timer 
from colorama import Back, init, Style
from datetime import datetime
from pathlib import Path
import engine, model_builder
from utils import create_writer, save_model
from data.data_setup import create_dataloaders

# Setup hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Setup directories
# train_dir = "Dataset/Dataset_A150_64x64_split/train"
# test_dir = "Dataset/test"
# val_dir = "Dataset/val"

train_dir = Path("../Dataset/Images_100_64x64/train")
test_dir = Path("../Dataset/Images_100_64x64/test")
val_dir = Path("../Dataset/Images_100_64x64/val")

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = T.Compose([
    T.Resize((64, 64)),
    # T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    T.CenterCrop(64),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create DataLoader's and get class_names
train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir, # type: ignore
                                                                               test_dir=test_dir, # type: ignore
                                                                               val_dir=val_dir, # type: ignore
                                                                               transform=data_transform, # type: ignore
                                                                               batch_size=BATCH_SIZE)


# Create model
model = model_builder.SignMobileNet(num_classes=35).to(device)
# or with torchvision.models.MobileNetV2
# model = MobileNetV2(num_classes=35)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             writer = create_writer(experment_name='MobileNet',
                                           model_name='MobileNet',
                                           extra=f"{NUM_EPOCHS}_epoch"),
                                           device=device)

# End the timer and print out how long it took
end_time = timer()
print(Back.GREEN + '[INFO]' + Style.RESET_ALL + f" Total training time: {end_time-start_time:.3f} seconds")

model_path = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# Save the model to file
save_model(model=model,
                 target_dir="results",
                 model_name="{model_path}.pth")