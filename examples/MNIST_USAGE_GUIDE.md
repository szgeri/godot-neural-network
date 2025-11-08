# MNIST Digit Classification - Usage Guide

This guide shows you how to train and use a digit classifier on MNIST data using this Godot Neural Network Framework.

## Quick Start (Using Pre-trained Model)

The easiest way to get started is to use the pre-trained model included in the project:

### Option 1: Using the Scene

1. Open the scene: `examples/mnist_simple_classifier.tscn`
2. In the Inspector, configure:
   - **Use Pretrained Model**: ✓ (checked)
   - **Pretrained Model Path**: `res://assets/models/digit_recognition.tres`
   - **Test Image Path**: Path to your digit image (e.g., `res://my_digit.png`)
3. Run the scene (F6)
4. Check the Output panel to see the prediction results!

### Option 2: Using the Script Directly

```gdscript
extends Node

func _ready():
    var classifier = MNISTSimpleClassifier.new()
    add_child(classifier)
    
    # Load pre-trained model
    classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
    classifier.use_pretrained_model = true
    
    # Wait for it to initialize
    await get_tree().process_frame
    
    # Classify an image
    var result = classifier.classify_from_file("res://path/to/digit.png")
    
    print("Predicted digit: ", result["predicted_digit"])
    print("Confidence: ", result["confidence"] * 100, "%")
```

### Option 3: Using the Existing Classifier UI

The project includes a full-featured classifier UI:

1. Open `examples/classifier/digit_classifier.tscn`
2. Run the scene
3. Click "Choose Network" and select `res://assets/models/digit_recognition.tres`
4. Click "Choose Input Folder" and select a folder containing digit images
5. Click on any image to see the classification result!

## Training Your Own Model

If you want to train your own model on MNIST data:

### Step 1: Prepare Your Dataset

Organize your digit images in this structure:

```
training_data/
├── 0/
│   ├── digit_0_sample_1.png
│   ├── digit_0_sample_2.png
│   └── ...
├── 1/
│   ├── digit_1_sample_1.png
│   └── ...
├── 2/
│   └── ...
...
└── 9/
    └── ...
```

**Image Requirements:**
- Any common format (PNG, JPG, etc.)
- Images will be automatically scaled to 32×32 pixels
- Grayscale conversion is automatic
- Recommended: Use images that are roughly square
- More images = better accuracy (aim for 500+ per digit)

### Step 2: Configure Training

1. Open `examples/mnist_simple_classifier.tscn`
2. In the Inspector, set:
   - **Use Pretrained Model**: ✗ (unchecked)
   - **Training Data Dir**: Path to your training_data folder
   - **Epochs**: 100-400 (more = better accuracy, longer training)
   - **Batch Size**: 512 (adjust based on your GPU)
   - **Learning Rate**: 0.1 (start here, adjust if needed)

### Step 3: Train

1. Run the scene (F6)
2. Watch the Output panel for training progress
3. Training will take a few minutes depending on:
   - Dataset size
   - Number of epochs
   - GPU performance
4. The trained model will be saved to `res://trained_mnist_model.tres`

### Step 4: Test Your Model

After training, test it:

```gdscript
classifier.pretrained_model_path = "res://trained_mnist_model.tres"
classifier.use_pretrained_model = true
classifier.test_image_path = "res://my_test_digit.png"
```

## Understanding the Output

When you classify a digit, you'll get:

```
==============================================================
[Classifying Image]
Path: res://my_digit.png
Image processed: 1024 features

--------------------------------------------------------------
PREDICTION RESULT
Predicted Digit: 7
Confidence: 98.45%
--------------------------------------------------------------

Class Probabilities:
  0: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.12%
  1: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.08%
  2: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.23%
  3: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.15%
  4: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.18%
  5: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.10%
  6: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.05%
  7: ██████████████████████████████████████████████████ 98.45%
  8: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.34%
  9: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.30%
==============================================================
```

**Key Metrics:**
- **Predicted Digit**: The network's best guess (0-9)
- **Confidence**: How sure the network is (0-100%)
- **Class Probabilities**: Probability for each digit (should sum to ~100%)

## Advanced Usage: From Code

### Example 1: Batch Classification

```gdscript
extends Node

var classifier: MNISTSimpleClassifier

func _ready():
    # Initialize classifier
    classifier = MNISTSimpleClassifier.new()
    add_child(classifier)
    classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
    classifier.use_pretrained_model = true
    
    await get_tree().process_frame
    
    # Classify multiple images
    var image_paths = [
        "res://digit1.png",
        "res://digit2.png",
        "res://digit3.png"
    ]
    
    for path in image_paths:
        var result = classifier.classify_from_file(path)
        print("%s -> %d (%.1f%%)" % [
            path,
            result["predicted_digit"],
            result["confidence"] * 100
        ])
```

### Example 2: Real-time Drawing Classification

```gdscript
extends Control

var classifier: MNISTSimpleClassifier
var drawing_surface: Image

func _ready():
    classifier = MNISTSimpleClassifier.new()
    add_child(classifier)
    classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
    classifier.use_pretrained_model = true
    
    drawing_surface = Image.create(128, 128, false, Image.FORMAT_RGBA8)

func _on_drawing_completed():
    # Convert drawing to neural network input
    var image_data = ImageUtils.read_image_from_data(drawing_surface, 0.25)
    
    # Classify
    var result = classifier.classify_from_data(image_data)
    
    # Show result
    $ResultLabel.text = "You drew: %s (%.0f%% confident)" % [
        result["predicted_label"],
        result["confidence"] * 100
    ]
```

### Example 3: Getting Top 3 Predictions

```gdscript
func get_top_k_predictions(classifier: MNISTSimpleClassifier, image_path: String, k: int = 3) -> Array:
    var result = classifier.classify_from_file(image_path)
    var probs = result["probabilities"]
    
    # Create array of [digit, probability] pairs
    var predictions = []
    for i in range(probs.size()):
        predictions.append({"digit": i, "prob": probs[i]})
    
    # Sort by probability (descending)
    predictions.sort_custom(func(a, b): return a["prob"] > b["prob"])
    
    # Return top k
    return predictions.slice(0, k)

# Usage:
var top3 = get_top_k_predictions(classifier, "res://my_digit.png", 3)
for pred in top3:
    print("Digit %d: %.2f%%" % [pred["digit"], pred["prob"] * 100])
```

## Improving Accuracy

If your model isn't performing well:

1. **More Training Data**: Aim for 1000+ images per digit
2. **Data Augmentation**: Rotate, scale, and shift your training images
3. **Increase Epochs**: Train for 200-400 epochs
4. **Adjust Learning Rate**: Try 0.05, 0.1, or 0.2
5. **Deeper Network**: Try `[1024, 256, 128, 64, 10]`
6. **Better Image Preprocessing**: Ensure images are centered and properly cropped

## Performance Tips

- **GPU Acceleration**: This framework uses GPU compute shaders - ensure Vulkan is enabled
- **Batch Size**: Larger batches = faster training but more memory
- **Inference Speed**: The pre-trained model can classify 1000s of images per second
- **Model Size**: The `.tres` file is lightweight and loads instantly

## Troubleshooting

### "Failed to load image"
- Check that the image path is correct and uses `res://` prefix
- Ensure the image file exists and is a valid format (PNG, JPG)

### "No model loaded"
- Make sure `use_pretrained_model` is checked (or training completed)
- Verify the model path points to a valid `.tres` file

### Low Accuracy
- Check that your test images match the training data style
- MNIST is designed for clean, centered digits on dark background
- Try inverting colors if needed (set `invert_image` to true)

### Training is Slow
- Reduce batch_size if running out of memory
- Reduce dataset size for faster iteration
- Ensure GPU compute shaders are enabled in Godot

## Next Steps

- Check out the visual training demo: `examples/visual_training/classification_demo.tscn`
- Explore the full classifier UI: `examples/classifier/digit_classifier.tscn`
- Read the main README.md for advanced features
- Try the doodle classifier for multi-category classification

## API Reference

### MNISTSimpleClassifier Methods

```gdscript
# Load and classify an image file
func classify_from_file(path: String) -> Dictionary

# Classify preprocessed image data
func classify_from_data(image_data: PackedFloat32Array) -> Dictionary

# Get the underlying neural network (for advanced use)
func get_network() -> NeuralNetwork
```

### Result Dictionary Structure

```gdscript
{
    "predicted_digit": int,        # 0-9
    "predicted_label": String,     # e.g., "Seven (7)"
    "confidence": float,           # 0.0-1.0
    "probabilities": PackedFloat32Array  # All 10 class probabilities
}
```

---

**Happy Digit Classification! 🔢🤖**

