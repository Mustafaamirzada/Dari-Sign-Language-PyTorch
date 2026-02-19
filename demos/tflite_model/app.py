import gradio as gr
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import time
from typing import Tuple, Dict

# Class names
class_names = ['آ','ا','ب','پ', 'ت','ث','ج','چ','ح','خ',
               'د','ذ','ر','ز','ژ','س','ش','ص','ض','ط',
               'ظ','ع','غ','ف','ق','ک','گ','ل','لا','م',
               'ن','و','ء','ها','ی'
              ]

# Load TFLite model
model_path = Path("MyMobile_Net_model.tflite")
if os.path.exists(model_path):
    print(f"✅ Model found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
else:
    print(f"❌ Model not found: {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details)

input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2]) if len(input_shape) > 2 else (224, 224)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for TFLite model"""
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(input_size)
    
    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    imgInput = np.transpose(imgInput, (0, 3, 1, 2))
    
    return imgInput

def predict(img) -> Tuple[Dict, float]:
    """Run inference with TFLite model"""
    start_time = time.time()
    
    try:
        # Preprocess
        input_data = preprocess_image(img)
        print(input_data)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predictions
        predictions = output_data[0] if len(output_data.shape) > 1 else output_data
        
        # Apply softmax if needed
        if predictions.max() > 1:
            exp_preds = np.exp(predictions - np.max(predictions))
            predictions = exp_preds / exp_preds.sum()
        
        # Create output dictionary
        pred_dict = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
        pred_dict = dict(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True))
        
        pred_time = round(time.time() - start_time, 5)
        
        return pred_dict, pred_time
        
    except Exception as e:
        return {f"Error: {str(e)}": 1.0}, round(time.time() - start_time, 5)

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top 5 Predictions"),
        gr.Number(label="Prediction Time (s)")
    ],
    title="Sign Language Recognition 🤟 (TFLite)",
    description="TensorFlow Lite model for Persian/Arabic sign language recognition",
    examples=[[f"examples/{f}"] for f in os.listdir("examples")] if os.path.exists("examples") else None,
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(debug=True)