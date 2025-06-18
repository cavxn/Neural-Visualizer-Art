import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import imageio

# Layers to visualize (in order)
layers_to_visualize = [
    "block1_conv1", "block1_conv2",
    "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3"
]

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def get_activation(model, image_tensor, layer_name):
    sub_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return sub_model.predict(image_tensor)

def save_activation_frame(activation, layer_name, index):
    size = activation.shape[1]
    n_features = activation.shape[-1]
    n_cols = int(np.ceil(np.sqrt(min(n_features, 64))))
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
    plt.title(f"{layer_name}", fontsize=18)
    plt.imshow(display_grid, cmap='inferno')
    plt.axis('off')
    frame_path = f"layer_frame_{index:02d}_{layer_name}.jpg"
    plt.savefig(frame_path, bbox_inches='tight')
    plt.close()
    return frame_path

# Main
image_tensor = load_image("/Users/cavins/Desktop/project/Visual-NN-Art/image.jpg!w700wp")
model = VGG16(weights="imagenet", include_top=True)

frames = []
for idx, layer in enumerate(layers_to_visualize):
    activation = get_activation(model, image_tensor, layer)
    frame_path = save_activation_frame(activation, layer, idx)
    frames.append(imageio.imread(frame_path))

# Create video/GIF
imageio.mimsave("cnn_visual_explanation.gif", frames, fps=1)
print("✅ Generated cnn_visual_explanation.gif")
