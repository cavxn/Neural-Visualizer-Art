import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# EEG Band → CNN Layer mapping
band_to_layer = {
    'delta': 'block1_conv2',
    'theta': 'block2_conv2',
    'alpha': 'block3_conv3',
    'beta':  'block4_conv3',
    'gamma': 'block5_conv3',
}

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def get_activation(model, image_tensor, layer_name):
    sub_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return sub_model.predict(image_tensor)

def visualize_activation(activation, layer_name, index):
    size = activation.shape[1]
    n_features = activation.shape[-1]

    # More expressive grid layout
    n_cols = int(np.ceil(np.sqrt(min(n_features, 64))))  # up to 64 features
    n_rows = int(np.ceil(min(n_features, 64) / n_cols))

    display_grid = np.zeros((n_rows * size, n_cols * size))

    for i in range(min(n_features, 64)):
        row = i // n_cols
        col = i % n_cols
        feature_map = activation[0, :, :, i]
        feature_map -= feature_map.mean()
        feature_map /= (feature_map.std() + 1e-6)
        feature_map *= 64
        feature_map += 128
        feature_map = np.clip(feature_map, 0, 255).astype('uint8')
        display_grid[row * size:(row + 1) * size,
                     col * size:(col + 1) * size] = feature_map

    plt.figure(figsize=(12, 12))
    plt.title(f"{layer_name} - Frame {index}")
    plt.imshow(display_grid, cmap='inferno')  # try 'plasma', 'magma', 'viridis'
    plt.axis('off')
    plt.savefig(f"{layer_name}_abstract_art_{index}.jpg", bbox_inches='tight')
    plt.close()
