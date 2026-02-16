"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm import tqdm
from typing import Dict, List
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import pkbar
import time
from colorama import Back, init, Style
from torch.utils.tensorboard import SummaryWriter

# Basic setup for early stopping criteria
patience = 3  # epochs to wait after no improvement
delta = 0.01  # minimum change in the monitored metric
best_val_loss = float("inf")  # best validation loss to compare against
no_improvement_count = 0  # count of epochs with no improvement

device = "cuda" if torch.cuda.is_available else "cpu"
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(Back.GREEN + '[INFO] ' + Style.RESET_ALL + "Stopping early as no improvement has been observed.")
                    

# Initialize early stopping
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

# Create train_step()
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device=device):
    
    # Put the model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # pbar = pkbar.Pbar("Trainig Step...", target=len(dataloader))
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # time.sleep(0.1)
        # pbar.update(batch)

        # Send data to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X) # output model logits

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Create a test step
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0,  0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


# 1. Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          writer: SummaryWriter, # this parameter take the writer
          device=device) -> Dict[str, List]:


    # 2. Create empty results dictionary
    results = {"train_loss": [],
              "train_accuracy": [],
              "test_loss": [],
              "test_accuracy": []}

    pbar = pkbar.Pbar("Start Trainig...", target=len(train_dataloader))
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        for i in range(len(train_dataloader)):
            time.sleep(0.1)
            pbar.update(i)
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # 4. Print out what's happening 
        print(colored(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}", "green", attrs=["bold"]))
        
        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)

        ### Use the writer parameter to track the experments ###
        # See if there's a writer, if so, log to it
        
        if writer:
            #
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={'train_loss': train_loss,
                                                'test_loss': test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag='Accuracy',
                               tag_scalar_dict={'train_accuracy': train_acc,
                                                'test_accuracy': test_acc},
                               global_step=epoch)
            
            # # Track the PyTorch model architecture
            # writer.add_graph(model=model, 
            #              # Pass in an example input
            #              input_to_model=torch.randn(64, 3, 64, 64).to(device))
    
            # Close the Writer
            writer.close()
        else:
            pass
        
        ### End Tracking ###
        
        # Check early stopping condition
        early_stopping.check_early_stop(test_loss)
        
        if early_stopping.stop_training:
          print(colored(f"Early stopping at epoch... {epoch}", "green", attrs=["bold"]))
          break
    
    # 6. Return the filled results at the end of the epochs
    return results
