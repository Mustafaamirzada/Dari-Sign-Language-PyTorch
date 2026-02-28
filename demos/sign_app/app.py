### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
import torchvision
from models import SignMobileNet
import torchvision.transforms.v2 as T

from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['آ','ا','ب','پ', 'ت','ث','ج','چ','ح','خ',
               'د','ذ','ر','ز','ژ','س','ش','ص','ض','ط',
               'ظ','ع','غ','ف','ق','ک','گ','ل','لا','م',
               'ن','و','ها','ء','ی'
              ]

### 2. Model and transforms preparation ###
data_transform = T.Compose([
    # Resize the image to 64x64 or 224x224
    T.Resize((224, 224)),
    # Flip the images randomly on the horizontal
    T.RandomHorizontalFlip(p=0.5),
    # Resize and Crop the images to 64x64 randomly
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    # Crop the images from Center with size 64x64
    T.CenterCrop(224),
    # convert the images to PIL objects
    T.ToImage(),
    # Turn the image into a torch.Tensor
    T.ToDtype(torch.float32, scale=True),
])

# Create EffNetB2 model
# effnetb2, effnetb2_transforms = create_effnetb2_model(
#     num_classes=3, # len(class_names) would also work
# )

# Load saved weights
# effnet_b2 = torch.load('Pytorch_Efficient_B2_model.pth', weights_only=False, map_location='cpu')
# effnetb2.load_state_dict(
#     torch.load(
#         f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
#         map_location=torch.device("cpu"),  # load to CPU
#     )
# )
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
transforms = weights.transforms()

#  
# model = SignMobileNet(num_classes=35)
# PATH = "Pytorch_MyMobile_Net_Original_model.pt"
PATH = "Script_MyMobile_Net_Original_model_224"
model = torch.jit.load(PATH, map_location='cpu')
# model.load_state_dict(state)
model.eval()

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = data_transform(img).unsqueeze(0)
    # img = transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    # effnet_b2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        # pred_probs = torch.softmax(effnet_b2(img), dim=1)
        pred_probs = torch.softmax(model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Sign Language 🫣🤚🫲🖐️"
description = "A MobileNet feature extractor computer vision model to classify images of Sign Language"


# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    )

# Launch the demo!
demo.launch(pwa=True)
