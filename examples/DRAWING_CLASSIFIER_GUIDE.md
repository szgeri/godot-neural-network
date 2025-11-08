# MNIST Drawing Classifier Guide

## Overview

The **MNIST Drawing Classifier** is an interactive example that lets you draw digits (0-9) on a canvas and get real-time classification results from the neural network.

## Features

✨ **Interactive Drawing Canvas**
- Draw digits with your mouse or touch input
- Soft brush with anti-aliasing for natural-looking strokes
- Adjustable brush size (16px by default)

🎯 **Smart Image Processing**
- Automatic bounding box detection
- Intelligent cropping with padding
- Center-alignment with square padding
- High-quality Lanczos resizing to 32x32
- Preview of processed image

🧠 **Real-time Classification**
- Uses pre-trained MNIST model
- Shows predicted digit with confidence score
- Displays probability distribution for all 10 digits
- Color-coded results (green = high confidence, yellow = medium, red = low)

## How to Use

### 1. Run the Scene

Open `examples/mnist_drawing_classifier.tscn` in Godot and run it (F5).

### 2. Draw a Digit

- Click and drag on the left canvas to draw a digit (0-9)
- Try to draw clearly and centered
- Use most of the canvas space for best results

### 3. Classify

Click the **"Classify Digit"** button to:
- Automatically crop and resize your drawing
- Show a preview of the processed 32x32 image
- Get the predicted digit with confidence score
- See probability distribution across all digits

### 4. Clear and Try Again

Click **"Clear Canvas"** to start fresh and try another digit.

## How It Works

### Drawing Phase
1. Mouse/touch input is captured on the canvas
2. A soft circular brush draws white pixels on a black background
3. Drawing is stored in a 280x280 image buffer

### Processing Phase
1. **Bounding Box Detection**: Scans the image to find the smallest rectangle containing all drawn pixels
2. **Cropping with Padding**: Crops to the bounding box with 20px padding on all sides
3. **Square Padding**: Adds black padding to make the image square (preserves aspect ratio)
4. **Resizing**: Uses Lanczos interpolation to resize to exactly 32x32 pixels
5. **Normalization**: Converts grayscale values to [-1, 1] range expected by the network

### Classification Phase
1. Processed image is fed to the neural network
2. Network outputs 10 probabilities (one per digit)
3. The highest probability is the predicted digit
4. Results are displayed with visual feedback

## Technical Details

### Image Format
- Input to network: `PackedFloat32Array` of 1024 floats (32×32)
- Value range: -1.0 (black) to 1.0 (white)
- Formula: `normalized = (grayscale - 0.5) * 2.0`

### Cropping Algorithm
- Scans all pixels to find content boundaries
- Adds 20px padding to prevent cutting off strokes
- Centers content in a square canvas
- Preserves aspect ratio during resize

### Brush Settings
- Size: 16 pixels (diameter)
- Anti-aliasing: Smooth falloff based on distance
- Additive blending: Multiple strokes increase brightness

## Tips for Best Results

✅ **DO:**
- Draw digits large and centered
- Use clear, simple strokes
- Try to match MNIST training data style (single continuous stroke)
- Leave some margin around your digit

❌ **DON'T:**
- Draw too small (won't crop well)
- Draw in the corners (won't center properly)
- Use multiple disconnected strokes
- Draw decorative elements (stick to simple digits)

## Customization

You can modify these settings in the script:

```gdscript
# Canvas size (drawing area)
var canvas_size: Vector2i = Vector2i(280, 280)

# Brush size (diameter)
var brush_size: float = 16.0

# Padding around cropped digit
var padding: int = 20
```

## Model Configuration

The classifier uses the pre-trained model at:
```
res://assets/models/digit_recognition.tres
```

To use a different model, modify the `_initialize_classifier()` function:

```gdscript
classifier.pretrained_model_path = "res://path/to/your/model.tres"
```

## Troubleshooting

**Problem**: "Draw something first!" message
- **Solution**: Make sure you've drawn something visible on the canvas

**Problem**: Low confidence scores
- **Solution**: Try drawing clearer, larger digits that match the MNIST style

**Problem**: Wrong predictions
- **Solution**: The model was trained on MNIST data which has a specific style. Try to match that style.

**Problem**: "Failed to classify" error
- **Solution**: Ensure the pre-trained model exists at `res://assets/models/digit_recognition.tres`

## Code Structure

### Main Components

1. **UI Setup** (`_create_ui`): Creates the drawing canvas and results display
2. **Canvas Initialization** (`_initialize_canvas`): Sets up the drawing buffer
3. **Input Handling** (`_on_canvas_input`): Captures mouse/touch events
4. **Drawing** (`_draw_at_position`): Renders brush strokes
5. **Image Processing**:
   - `_find_bounding_box()`: Detects content boundaries
   - `_crop_image()`: Crops to bounding box
   - `_make_square_with_padding()`: Centers and squares the image
   - `_image_to_input_array()`: Converts to network input format
6. **Classification** (`_classify_drawing`): Runs the full pipeline
7. **Results Display** (`_display_results`): Shows predictions

## Related Examples

- `mnist_simple_classifier.gd`: Classify images from files
- `mnist_demo_ui.gd`: UI for loading and classifying image files
- `examples/training/digit_training/`: Train your own digit recognition model

## Next Steps

- Try training your own model with custom data
- Experiment with different brush sizes and canvas sizes
- Add color themes or additional drawing tools
- Implement auto-classification (classify as you draw)
- Add gesture recognition for multi-touch devices

Enjoy experimenting with handwritten digit recognition! 🎨🔢

