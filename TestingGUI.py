import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import torch
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import json
import easyocr


"""FIRST WE MUST RUN THE TRAINING FILE THEN RUN THIS FILE BECAUSE HERE WE USE TRAINED MODEL"""
# Function to scale bounding box coordinates
def scale_bounding_box(box, width_scale, height_scale):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]

# Initialize LayoutLMv3 components
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

# Define the model class
class ModelModule(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )

# Function to predict document class
def predict_document_class(im_path, model, processor, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image = Image.open(im_path).convert('RGB')
    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height

    json_path = im_path.with_suffix('.json')
    if json_path.exists():
        with json_path.open('r') as f:
            ocr_result = json.load(f)
        words = [row['word'] for row in ocr_result]
        boxes = [scale_bounding_box(row['bbox'], width_scale, height_scale) for row in ocr_result]
    else:
        # Use EasyOCR to perform OCR
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(im_path)
        words = [item[1] for item in ocr_result]
        boxes = [scale_bounding_box(item[0], width_scale, height_scale) for item in ocr_result]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            bbox=encoding['bbox'].to(device),
            pixel_values=encoding['pixel_values'].to(device)
        )

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

# Load your trained model
model_path = "D:\\Documentation classification\\documodel\\documodel\\best_model2.pth"
model = ModelModule(n_classes=10)  # Replace with your number of classes
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define your class names
class_names = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

# Function to handle file upload and prediction
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            # Predict document class
            predicted_class_name = predict_document_class(Path(file_path), model, processor, class_names)
            result_label.config(text=f"Predicted Class: {predicted_class_name}")

            # Display image
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img)
            img_label.config(image=img)
            img_label.image = img

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")

# Create main window
window = tk.Tk()
window.title("Document Classification")
window.geometry("600x400")
window.config(bg='#f0f0f0')

# Title Label
title_label = tk.Label(window, text="Document Classification📃📄 ", font=("Helvetica", 24, "bold"), bg='#f0f0f0', fg='#333')
title_label.pack(pady=20)

# Upload Button
upload_button = tk.Button(window, text="Choose File", command=upload_image, font=("Helvetica", 16), bg='#4CAF50', fg='white', bd=0, padx=10, pady=10)
upload_button.pack(pady=20)

# Result Label
result_label = tk.Label(window, text="", font=("Helvetica", 18), bg='#f0f0f0', fg='#333')
result_label.pack(pady=10)

# Image Label
img_label = tk.Label(window, bg='#f0f0f0')
img_label.pack(pady=10)

# Run the GUI
window.mainloop()
