"""
File containing various utility functions for PyTorch model training.
""" 
import torch
from colorama import Back, init, Style
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def create_writer(
        experment_name: str,
        model_name: str,
        extra: str=None # type: ignore
    ) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    # Get timestamp of current date ()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # returns current date in YYYY-MM-DD format

    if extra:
        log_dir = os.path.join("runs", timestamp, experment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experment_name, model_name)

    print(Back.BLUE + '[INFO]' + Style.RESET_ALL + f" Created SummaryWriter. Saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(Back.BLUE + '[INFO]' + Style.RESET_ALL + f"Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
