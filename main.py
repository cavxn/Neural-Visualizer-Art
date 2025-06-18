import arff
import pandas as pd
from eeg_utils import get_dominant_band
from art_generator import load_image, get_activation, visualize_activation, band_to_layer
from tensorflow.keras.applications import VGG16

# Load .arff EEG file
dataset = arff.load(open("EEG Eye State.arff", 'r'))
df = pd.DataFrame(dataset['data'])
eeg_data = df.iloc[:, :-1]  # remove eye state label column

# Load image and model
image_tensor = load_image("/Users/cavins/Desktop/project/Visual-NN-Art/image.jpg!w700wp")
model = VGG16(weights="imagenet", include_top=True)

# Process first few EEG rows
for i in range(min(5, len(eeg_data))):  # Generate 5 frames
    row = eeg_data.iloc[i]
    band = get_dominant_band(row)
    layer = band_to_layer[band]
    activation = get_activation(model, image_tensor, layer)
    visualize_activation(activation, layer, i)

print("✅ Brainwave-based abstract art generated.")
