import os
import gc
import torch
import librosa
import librosa.display
import pandas as pd
import timm
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import dearpygui.dearpygui as dpg
import csv
import tkinter as tk
from tkinter import filedialog
import sys
import threading
import easygui



# Paths (Do NOT change)
MODEL_PATH = "/home/pujal/Documents/NodeEditor/model/efficientnet_bird.pth"
METADATA_PATH = "/home/pujal/Documents/NodeEditor/train_metadata.csv"
TEMP_SPECTROGRAM = "/home/pujal/Documents/NodeEditor/spectrogram.png"
AUDIO_FILE = "/home/pujal/Documents/NodeEditor/Bird_audio/bird.ogg"
audio_file = ""
audio_folder = None

# Load metadata
metadata = pd.read_csv(METADATA_PATH)
label_to_name = dict(zip(metadata["primary_label"], metadata["common_name"]))
num_classes = len(label_to_name)  # Dynamic class count

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

def convert_audio_to_spectrogram(audio_path, save_path):
    """Converts an audio file to a spectrogram and saves it."""
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(2, 2), dpi=100)
    librosa.display.specshow(mel_spec_db, sr=sr, cmap="magma")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    del y, mel_spec, mel_spec_db
    gc.collect()  # Avoid excessive GPU memory clearing


class EfficientNetBirdClassifier(nn.Module):
    """EfficientNet Model for Bird Classification."""
    def __init__(self, num_classes):
        super(EfficientNetBirdClassifier, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=False)
        self.model.classifier = nn.Linear(1280, num_classes)  # Dynamic classifier

    def forward(self, x):
        return self.model(x)

def load_model(model_path, num_classes):
    """Loads the model while handling class mismatches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetBirdClassifier(num_classes)

    # Load checkpoint with strict=False to avoid size mismatch errors
    checkpoint = torch.load(model_path, map_location=device)

    # Ensure classifier matches checkpoint
    saved_class_count = checkpoint["model.classifier.bias"].shape[0]
    if saved_class_count != num_classes:
        print(f"⚠ Model class mismatch: Checkpoint has {saved_class_count}, but expected {num_classes}. Adjusting...")
        model.model.classifier = nn.Linear(1280, saved_class_count)

    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model, device


def predict_bird(audio_file, model, device):
    """Processes the audio and predicts the bird species."""
    convert_audio_to_spectrogram(audio_file, TEMP_SPECTROGRAM)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(TEMP_SPECTROGRAM).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class_id = torch.argmax(output, dim=1).item()

    del image, output
    torch.cuda.empty_cache()
    gc.collect()

    predicted_label = list(label_to_name.keys())[predicted_class_id]
    return label_to_name.get(predicted_label, "Unknown Bird")

def load_csv(sender, app_data, user_data):
    file_path = app_data['file_path_name']
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)  # Convert to list

    # Clear previous table data
    dpg.delete_item("table", children_only=True)

    # Add headers (first row)
    dpg.add_table_column(label=data[0][i], parent="table") 
    for i in range(label=(data[0]),parent="table"):
        for row in data[1:]:
            with dpg.table_row(parent="table"):
                for cell in row:
                    dpg.add_text(cell)




def RUN(sender, app_data):
    model, device = load_model(MODEL_PATH, num_classes)
    if audio_file:  # Check if an audio file is selected
        predicted_bird = predict_bird(audio_file, model, device)
        print(f"Detected Bird: {predicted_bird}")
        dpg.set_value("birdname_", f"Bird name: {predicted_bird}")

        # Load the spectrogram image
        width, height, channels, data = dpg.load_image(TEMP_SPECTROGRAM)

        # If the spectrogram window exists, delete it before recreating
        if dpg.does_item_exist("spectrogram_window"):
            dpg.delete_item("spectrogram_window")

        # If the texture exists, delete it before recreating
        if dpg.does_item_exist("texture_tag"):
            dpg.delete_item("texture_tag")

        # Recreate the texture
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag="texture_tag")

        # Create a new spectrogram window
        with dpg.window(label="Log-Mel Spectrogram", tag="spectrogram_window", pos=[640, 360], width=350, height=250):
            dpg.add_image("texture_tag")

        # Fetch Wikipedia facts

    else:
        print("⚠ No audio file selected!")
        dpg.set_value("birdname_", "Please select an audio file first.")




def select_audio_file():
    def file_dialog():
        global audio_file  # Ensure global access
        file_selected = easygui.fileopenbox(
            title="Select an Audio File",
            filetypes=["*.mp3", "*.wav", "*.ogg", "*.flac", "*.*"]
        )  
        if file_selected:
            audio_file = str(file_selected)  # Store the selected file
            print(f"Selected File: {audio_file}")  # Debugging output
            dpg.configure_item("audio_path", default_value=file_selected)  # Update UI

    threading.Thread(target=file_dialog, daemon=True).start()

# Initialize Dear PyGui
dpg.create_context()
dpg.create_viewport(title='Audio Selector', width=600, height=200)


dpg.create_context()
dpg.create_viewport(title='Bird Detection Model', width=1280, height=720)

with dpg.window(label="Editor window",width=1024,height=568):
    dpg.add_text("Bird name : ",tag="birdname_")
    dpg.add_button(label="RUN",callback=RUN)
    dpg.add_text("Selected Audio File:")
    dpg.add_input_text(tag="audio_path", readonly=True)  # Display selected file path
    dpg.add_button(label="Select Audio File", callback=select_audio_file) 


with dpg.window(label="CSV Reader",pos=[1280/2,600]):
    dpg.add_button(label="Load CSV", callback=lambda: dpg.show_item("file_dialog"))
    dpg.add_table(header_row=True, resizable=True, borders_innerH=True, borders_innerV=True, tag="table")

    # File dialog to select CSV
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_csv, tag="file_dialog", width=400, height=300):
        dpg.add_file_extension("*.csv", color=(150, 255, 150, 255))

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()