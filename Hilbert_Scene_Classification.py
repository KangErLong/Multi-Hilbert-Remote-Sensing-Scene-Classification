import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.utils import to_categorical
from hilbertcurve.hilbertcurve import HilbertCurve
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

# --- Parameters ---
NUM_CLASSES = 21  # UCMerced has 21 classes
INPUT_SHAPE_CNN = (64, 64, 3)
INPUT_SHAPE_IMG = (256, 256, 3)  # UCMerced images are 256x256
HILBERT_DIMS = 2

# Multiple Hilbert curve parameters
HILBERT_CONFIGS = [
    {"p": 6, "size": 64},   # For capturing fine details (64x64)
    {"p": 5, "size": 32},   # For capturing medium-scale features (32x32)
    {"p": 4, "size": 16}    # For capturing coarse features (16x16)
]

# For backward compatibility with existing code
HILBERT_P = HILBERT_CONFIGS[0]["p"]  # Use the first config's p value as default

# Total number of features from all Hilbert curves
HILBERT_FEATURE_DIM = 8 * len(HILBERT_CONFIGS)  # 8 features per curve (5 stats + entropy + 2 FFT)

# --- Initialize multiple Hilbert curves ---
hilbert_curves = [HilbertCurve(config["p"], HILBERT_DIMS) for config in HILBERT_CONFIGS]
# Also initialize the original single curve for backward compatibility
hilbert_curve = HilbertCurve(HILBERT_P, HILBERT_DIMS)

def extract_multi_hilbert_features(images):
    """
    Extracts features using multiple Hilbert curves at different scales.
    """
    num_images = images.shape[0]
    features = np.zeros((num_images, HILBERT_FEATURE_DIM))
    
    for i in tqdm(range(num_images), desc="Extracting Multi-Scale Hilbert Features"):
        # Convert to uint8 before using cvtColor
        img_uint8 = (images[i] * 255).astype(np.uint8)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        feature_idx = 0
        # Process each Hilbert curve configuration
        for curve_idx, (curve, config) in enumerate(zip(hilbert_curves, HILBERT_CONFIGS)):
            # Resize image to match the current Hilbert curve size
            current_size = config["size"]
            img_resized = cv2.resize(img_gray, (current_size, current_size), interpolation=cv2.INTER_AREA)
            
            # Map pixels to 1D Hilbert sequence
            num_pixels = current_size * current_size
            pixel_sequence = np.zeros(num_pixels)
            coords = curve.points_from_distances(np.arange(num_pixels))
            
            for idx, coord in enumerate(coords):
                pixel_sequence[idx] = img_resized[coord[1], coord[0]]
            
            # Extract statistical features for this curve
            features[i, feature_idx] = np.mean(pixel_sequence)  # Mean
            features[i, feature_idx + 1] = np.std(pixel_sequence)  # Standard deviation
            features[i, feature_idx + 2] = np.median(pixel_sequence)  # Median
            features[i, feature_idx + 3] = np.percentile(pixel_sequence, 25)  # 1st quartile
            features[i, feature_idx + 4] = np.percentile(pixel_sequence, 75)  # 3rd quartile
            
            # Calculate entropy
            hist = np.histogram(pixel_sequence, bins=32, density=True)[0]
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            features[i, feature_idx + 5] = entropy
            
            # Calculate FFT coefficients
            fft = np.fft.fft(pixel_sequence)
            fft_magnitude = np.abs(fft)
            features[i, feature_idx + 6] = np.mean(fft_magnitude)
            features[i, feature_idx + 7] = np.std(fft_magnitude)
            
            # Move to the next set of features
            feature_idx += 8
    
    return features

def load_ucmerced_dataset(dataset_path):
    images = []
    labels = []
    class_names = []
    
    # Get class names from directory names
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    # Load images and labels
    for class_name in tqdm(class_names, desc="Loading classes"):
        class_dir = os.path.join(dataset_path, class_name)
        class_idx = class_to_idx[class_name]
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, (256, 256))  # Ensure consistent dimensions
                    images.append(img)
                    labels.append(class_idx)
    
    return np.array(images), np.array(labels), class_names

# --- 1. Load and Preprocess UCMerced Dataset ---
print("Loading and preprocessing UCMerced Land Use dataset...")

def resize_images(images, target_shape):
    resized_images = np.zeros((len(images), target_shape[0], target_shape[1], target_shape[2]))
    for i, img in enumerate(tqdm(images, desc="Resizing images")):
        resized_images[i] = cv2.resize(img, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_AREA)
    return resized_images

# Load the dataset
dataset_path = "e:\\Trae_Python\\Hilbert_RemoteSensing_SceneClassification\\UCMerced_LandUse\\Images"  # Path to your local dataset
x_data, y_data, class_names = load_ucmerced_dataset(dataset_path)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

print(f"Loaded {len(x_train)} training and {len(x_test)} testing images")
print(f"Classes: {class_names}")

# Resize images for CNN input and Hilbert curve
x_train_resized = resize_images(x_train, INPUT_SHAPE_CNN)
x_test_resized = resize_images(x_test, INPUT_SHAPE_CNN)
print(f"Data resized to {INPUT_SHAPE_CNN}")

# --- Update the code to use the new multi-scale Hilbert features ---
print("Pre-calculating multi-scale Hilbert features for training set...")
x_train_hilbert = extract_multi_hilbert_features(x_train_resized)
print("Pre-calculating multi-scale Hilbert features for test set...")
x_test_hilbert = extract_multi_hilbert_features(x_test_resized)
print(f"Multi-scale Hilbert features shape: {x_train_hilbert.shape}")


BATCH_SIZE = 32  # Smaller batch size due to larger images
EPOCHS = 50
FINE_TUNE_EPOCHS = 10

# --- 1. Load and Preprocess UCMerced Dataset ---
print("Loading and preprocessing UCMerced Land Use dataset...")

def load_ucmerced_dataset(dataset_path):
    images = []
    labels = []
    class_names = []
    
    # Get class names from directory names
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    # Load images and labels
    for class_name in tqdm(class_names, desc="Loading classes"):
        class_dir = os.path.join(dataset_path, class_name)
        class_idx = class_to_idx[class_name]
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, (256, 256))  # Ensure consistent dimensions
                    images.append(img)
                    labels.append(class_idx)
    
    return np.array(images), np.array(labels), class_names

# Load the dataset
dataset_path = "e:\\Trae_Python\\Hilbert_RemoteSensing_SceneClassification\\UCMerced_LandUse\\Images"  # Path to your local dataset
x_data, y_data, class_names = load_ucmerced_dataset(dataset_path)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

print(f"Loaded {len(x_train)} training and {len(x_test)} testing images")
print(f"Classes: {class_names}")

# Resize images for CNN input and Hilbert curve
def resize_images(images, target_shape):
    resized_images = np.zeros((len(images), target_shape[0], target_shape[1], target_shape[2]))
    for i, img in enumerate(tqdm(images, desc="Resizing images")):
        resized_images[i] = cv2.resize(img, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_AREA)
    return resized_images

x_train_resized = resize_images(x_train, INPUT_SHAPE_CNN)
x_test_resized = resize_images(x_test, INPUT_SHAPE_CNN)
print(f"Data resized to {INPUT_SHAPE_CNN}")

# --- 2. Enhanced Hilbert Curve Feature Extraction ---
print("Defining Hilbert feature extraction...")
hilbert_curve = HilbertCurve(HILBERT_P, HILBERT_DIMS)

def extract_hilbert_features(images):
    """
    Extracts enhanced statistical features from the Hilbert curve mapping of images.
    """
    num_images = images.shape[0]
    features = np.zeros((num_images, HILBERT_FEATURE_DIM))
    num_pixels = images.shape[1] * images.shape[2]
    if num_pixels != hilbert_curve.max_h + 1:
         raise ValueError(f"Image dimensions ({images.shape[1]}x{images.shape[2]}) do not match Hilbert curve size (P={HILBERT_P}, size={2**HILBERT_P}x{2**HILBERT_P})")

    for i in tqdm(range(num_images), desc="Extracting Hilbert Features"):
        # Convert to grayscale
        img_uint8 = (images[i] * 255).astype('uint8')  # Convert to uint8 first
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Map pixels to 1D Hilbert sequence
        pixel_sequence = np.zeros(num_pixels)
        coords = hilbert_curve.points_from_distances(np.arange(num_pixels))

        for idx, coord in enumerate(coords):
            pixel_sequence[idx] = img_gray[coord[1], coord[0]]

        # Extract enhanced statistical features
        features[i, 0] = np.mean(pixel_sequence)  # Mean
        features[i, 1] = np.std(pixel_sequence)   # Standard deviation
        features[i, 2] = np.median(pixel_sequence)  # Median
        features[i, 3] = np.percentile(pixel_sequence, 25)  # 1st quartile
        features[i, 4] = np.percentile(pixel_sequence, 75)  # 3rd quartile
        
        # Calculate entropy
        hist = np.histogram(pixel_sequence, bins=32, density=True)[0]
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        features[i, 5] = entropy
        
        # Calculate FFT coefficients
        fft = np.fft.fft(pixel_sequence)
        fft_magnitude = np.abs(fft)
        features[i, 6] = np.mean(fft_magnitude)
        features[i, 7] = np.std(fft_magnitude)

    return features

# --- 3. Pre-calculate Hilbert Features ---
print("Pre-calculating Hilbert features for training set...")
x_train_hilbert = extract_hilbert_features(x_train_resized)
print("Pre-calculating Hilbert features for test set...")
x_test_hilbert = extract_hilbert_features(x_test_resized)
print(f"Hilbert features shape: {x_train_hilbert.shape}")

# --- 4. Build Multiple Baseline Models ---
print("Building Baseline Models...")

def build_baseline_model(model_name, input_shape, num_classes):
    # Select base model based on name
    if model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    # Freeze the base model layers initially
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name=f"baseline_{model_name}")
    return model, base_model

# Choose one model for this run
baseline_model, base_model_backbone = build_baseline_model('mobilenet', INPUT_SHAPE_CNN, NUM_CLASSES)
baseline_model.summary()

# --- 5. Build Enhanced Fusion Model ---
print("Building Fusion Model...")

def build_fusion_model(cnn_input_shape, hilbert_input_dim, num_classes):
    # CNN Branch
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=cnn_input_shape)
    base_model.trainable = False

    input_cnn = keras.Input(shape=cnn_input_shape, name='cnn_input')
    cnn_features = base_model(input_cnn, training=False)
    cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
    cnn_features = layers.Dense(512, activation='relu')(cnn_features)
    cnn_features = layers.Dropout(0.3)(cnn_features)

    # Hilbert Branch with separate processing for each scale
    input_hilbert = keras.Input(shape=(hilbert_input_dim,), name='hilbert_input')
    
    # Split the input into chunks for each Hilbert curve configuration
    features_per_curve = 5
    num_curves = hilbert_input_dim // features_per_curve
    
    hilbert_outputs = []
    for i in range(num_curves):
        # Extract features for this curve
        start_idx = i * features_per_curve
        end_idx = start_idx + features_per_curve
        curve_features = layers.Lambda(lambda x: x[:, start_idx:end_idx])(input_hilbert)
        
        # Process features for this curve
        x = layers.Dense(32, activation='relu')(curve_features)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Add attention for this scale
        attention = layers.Dense(1, activation='sigmoid')(x)
        weighted_features = layers.Multiply()([x, attention])
        
        hilbert_outputs.append(weighted_features)
    
    # Concatenate features from all curves
    if len(hilbert_outputs) > 1:
        hilbert_features = layers.Concatenate()(hilbert_outputs)
    else:
        hilbert_features = hilbert_outputs[0]
    
    # Additional processing of combined Hilbert features
    hilbert_features = layers.Dense(128, activation='relu')(hilbert_features)
    hilbert_features = layers.Dropout(0.3)(hilbert_features)
    
    # Global attention mechanism
    global_attention = layers.Dense(1, activation='sigmoid')(hilbert_features)
    weighted_hilbert = layers.Multiply()([hilbert_features, global_attention])
    
    # Concatenate CNN and Hilbert features
    merged_features = layers.Concatenate()([cnn_features, weighted_hilbert])

    # Classification Head
    x = layers.Dense(256, activation='relu')(merged_features)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=[input_cnn, input_hilbert], outputs=outputs, name="multi_scale_fusion_model")
    return model, base_model


fusion_model, fusion_base_model = build_fusion_model(INPUT_SHAPE_CNN, HILBERT_FEATURE_DIM, NUM_CLASSES)
fusion_model.summary()

# Plot model architecture
from tensorflow.keras.utils import plot_model
plot_model(baseline_model, to_file='baseline_model.png', show_shapes=True, show_layer_names=True)
plot_model(fusion_model, to_file='fusion_model.png', show_shapes=True, show_layer_names=True)

# --- 6. Train Models with Callbacks ---
# Define callbacks
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'fusion_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
)

# Compile
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
baseline_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
fusion_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train Baseline
print("\n--- Training Baseline Model ---")
history_baseline = baseline_model.fit(
    x_train_resized, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Train Fusion Model
print("\n--- Training Fusion Model ---")
history_fusion = fusion_model.fit(
    [x_train_resized, x_train_hilbert], y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# --- Fine-tuning phase ---
print("\n--- Fine-tuning Baseline Model ---")
# Unfreeze some layers in the base model
for layer in base_model_backbone.layers[-20:]:
    layer.trainable = True
    
baseline_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_baseline_ft = baseline_model.fit(
    x_train_resized, y_train_cat,
    batch_size=BATCH_SIZE // 2,  # Smaller batch size for fine-tuning
    epochs=FINE_TUNE_EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr]
)

print("\n--- Fine-tuning Fusion Model ---")
# Unfreeze some layers in the fusion base model
for layer in fusion_base_model.layers[-20:]:
    layer.trainable = True
    
fusion_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fusion_ft = fusion_model.fit(
    [x_train_resized, x_train_hilbert], y_train_cat,
    batch_size=BATCH_SIZE // 2,
    epochs=FINE_TUNE_EPOCHS,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr]
)

# --- 7. Evaluate Models ---
print("\n--- Evaluating Models ---")

# Evaluate Baseline
loss_base, acc_base = baseline_model.evaluate(x_test_resized, y_test_cat, verbose=1)
print(f"Baseline Model Test Accuracy: {acc_base:.4f}")

# Evaluate Fusion Model
loss_fusion, acc_fusion = fusion_model.evaluate([x_test_resized, x_test_hilbert], y_test_cat, verbose=1)
print(f"Fusion Model Test Accuracy:   {acc_fusion:.4f}")

# --- Comparison ---
print("\n--- Comparison ---")
if acc_fusion > acc_base:
    print(f"Fusion model improved accuracy by {acc_fusion - acc_base:.4f}")
elif acc_fusion < acc_base:
     print(f"Fusion model decreased accuracy by {acc_base - acc_fusion:.4f}")
else:
     print("Fusion model accuracy is the same as the baseline model.")

# --- 8. Visualizations and Analysis ---
# Plot training history
def plot_history(history1, history2, metric='accuracy'):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy values for baseline
    plt.subplot(1, 2, 1)
    plt.plot(history1.history[metric])
    plt.plot(history1.history[f'val_{metric}'])
    plt.title(f'Baseline Model {metric}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation accuracy values for fusion
    plt.subplot(1, 2, 2)
    plt.plot(history2.history[metric])
    plt.plot(history2.history[f'val_{metric}'])
    plt.title(f'Fusion Model {metric}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{metric}_comparison.png')
    plt.show()

# Plot accuracy and loss
plot_history(history_baseline, history_fusion, 'accuracy')
plot_history(history_baseline, history_fusion, 'loss')

# Confusion Matrix
def plot_confusion_matrix(model, x_test, y_test, model_name):
    if isinstance(x_test, list):
        y_pred = model.predict(x_test)
    else:
        y_pred = model.predict(x_test)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.squeeze(y_test)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()
    
    # Print classification report
    print(f"\nClassification Report - {model_name}")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Generate confusion matrices
plot_confusion_matrix(baseline_model, x_test_resized, y_test, "Baseline")
plot_confusion_matrix(fusion_model, [x_test_resized, x_test_hilbert], y_test, "Fusion")

# ---