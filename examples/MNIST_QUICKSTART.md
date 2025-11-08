# 🎯 MNIST Digit Classification - Quick Start

This guide gets you classifying handwritten digits in **under 5 minutes**!

---

## 🚀 Method 1: Instant Classification (Fastest)

Use the pre-trained model that's already included:

### Step 1: Create a test scene

1. Create a new scene in Godot (Scene → New Scene)
2. Add a `Node` as the root
3. Attach the script: `examples/quick_test_classifier.gd`

### Step 2: Configure

In the Inspector panel:
- **Test Image Path**: Select any digit image (PNG/JPG)
  - Example: Point to any 0-9 digit image you have

### Step 3: Run!

- Press **F6** (Run Current Scene)
- Check the **Output** panel at the bottom
- You'll see the prediction instantly!

**Example Output:**
```
==================================================
QUICK CLASSIFICATION RESULT
==================================================
Image: res://my_digit_7.png
Predicted Digit: 7
Confidence: 96.8%
==================================================

Top 3 Predictions:
  1. Digit 7: 96.80%
  2. Digit 1: 2.10%
  3. Digit 9: 0.80%
```

---

## 🎨 Method 2: Interactive UI (Most Visual)

Use the full-featured classifier with visual feedback:

### Step 1: Open the classifier scene

File → Open Scene → `examples/classifier/digit_classifier.tscn`

### Step 2: Run and configure

1. Press **F5** (Run Project)
2. Click **"Choose Network"** → Select `res://assets/models/digit_recognition.tres`
3. Click **"Choose Input Folder"** → Select a folder containing digit images

### Step 3: Classify!

- Click on any digit image from the list
- See the prediction appear instantly with confidence scores
- Try different images!

---

## 🧪 Method 3: Code-Based Classification (Most Flexible)

Perfect for integrating into your own project:

### Simple Example

Create a script and attach it to any Node:

```gdscript
extends Node

func _ready():
    # Create classifier
    var classifier = MNISTSimpleClassifier.new()
    add_child(classifier)
    
    # Load pre-trained model
    classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
    classifier.use_pretrained_model = true
    
    # Wait for initialization
    await get_tree().process_frame
    
    # Classify an image
    var result = classifier.classify_from_file("res://your_digit.png")
    
    # Use the result
    print("Predicted: ", result["predicted_digit"])
    print("Confidence: ", result["confidence"] * 100, "%")
    
    # Access all probabilities
    for i in range(10):
        print("Digit %d: %.2f%%" % [i, result["probabilities"][i] * 100])
```

---

## 📦 What You Get

Every classification returns a **Dictionary** with:

```gdscript
{
    "predicted_digit": 7,              # The predicted digit (0-9)
    "predicted_label": "Seven (7)",    # Human-readable label
    "confidence": 0.968,               # Confidence (0.0 - 1.0)
    "probabilities": [0.001, 0.021, ...] # All 10 class probabilities
}
```

---

## 🎓 Training Your Own Model

Want to train on your own digit data?

### Prepare Your Dataset

Organize images like this:

```
my_digits/
├── 0/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── 1/
│   └── ...
...
└── 9/
    └── ...
```

### Train

1. Open: `examples/mnist_simple_classifier.tscn`
2. In Inspector:
   - **Use Pretrained Model**: ✗ (uncheck)
   - **Training Data Dir**: `/path/to/my_digits`
   - **Epochs**: 200
   - **Learning Rate**: 0.1
3. Run the scene (F6)
4. Wait for training to complete
5. Your model is saved to: `res://trained_mnist_model.tres`

**Training Time:** 2-10 minutes depending on dataset size and GPU

---

## 🔍 Testing with Your Own Images

### Image Requirements

✅ **Good:**
- Any size (auto-resized to 32×32)
- PNG, JPG, or other common formats
- Digit centered in the image
- Clear, readable digit

❌ **Avoid:**
- Multiple digits in one image
- Extremely low resolution
- Heavily rotated digits (>45°)

### Tips for Best Results

1. **Center the digit** - The neural network expects the digit to be roughly centered
2. **Good contrast** - Clear separation between digit and background
3. **Single digit** - One digit per image
4. **Clean background** - Minimal noise or clutter

---

## 💡 Common Use Cases

### 1. Validate User Input

```gdscript
func verify_handwritten_code(image_path: String) -> bool:
    var result = classifier.classify_from_file(image_path)
    return result["confidence"] > 0.9  # 90% confidence threshold
```

### 2. Batch Processing

```gdscript
func classify_folder(folder_path: String) -> Array:
    var results = []
    var files = DirAccess.open(folder_path).get_files()
    
    for file in files:
        var full_path = folder_path.path_join(file)
        var result = classifier.classify_from_file(full_path)
        results.append(result)
    
    return results
```

### 3. Real-time Drawing Recognition

```gdscript
func on_canvas_updated(canvas: Image):
    var image_data = ImageUtils.read_image_from_canvas(canvas, 0.25)
    var result = classifier.classify_from_data(image_data)
    
    $Label.text = "I think you drew: %d" % result["predicted_digit"]
    $ProgressBar.value = result["confidence"] * 100
```

---

## 🐛 Troubleshooting

### "Failed to load image"

**Cause:** Invalid path or unsupported format  
**Solution:** 
- Use `res://` prefix for resources
- Ensure file exists
- Check format is PNG, JPG, etc.

### "No model loaded"

**Cause:** Model file not found  
**Solution:**
- Verify path: `res://assets/models/digit_recognition.tres`
- Ensure `use_pretrained_model` is checked
- Check file exists in project

### Low Accuracy / Wrong Predictions

**Cause:** Input image doesn't match training data  
**Solutions:**
- Ensure digit is centered
- Check contrast (white digit on dark background, or vice versa)
- Try setting `invert_image = true` if colors are reversed
- Verify image isn't too blurry or distorted

### "GPU context failed"

**Cause:** Vulkan not enabled  
**Solution:**
- Project → Project Settings → Rendering
- Enable Vulkan compute shaders
- Restart Godot

---

## 📊 Performance

The framework is **GPU-accelerated** and extremely fast:

- **Single image inference:** < 1ms
- **Batch (100 images):** ~ 5-10ms
- **Training (1000 images, 100 epochs):** ~ 2-5 minutes
- **Model loading:** Instant

---

## 🎯 Next Steps

Now that you have digit classification working:

1. **Explore other demos:**
   - `examples/visual_training/classification_demo.tscn` - Live decision boundary visualization
   - `examples/training/doodle_training/` - Multi-category classification

2. **Read the full guide:**
   - `examples/MNIST_USAGE_GUIDE.md` - Complete API reference and advanced usage

3. **Customize the network:**
   - Try different layer sizes
   - Experiment with activation functions
   - Adjust learning rates and optimizers

4. **Build something cool:**
   - Digit recognition game
   - Handwriting tutor
   - Equation solver
   - Digit-based authentication

---

## 📚 Resources

- **Main README:** `/README.md` - Framework overview
- **Full Usage Guide:** `/examples/MNIST_USAGE_GUIDE.md` - Detailed documentation
- **Core Scripts:**
  - Simple Classifier: `/examples/mnist_simple_classifier.gd`
  - Quick Test: `/examples/quick_test_classifier.gd`
  - Full Training: `/examples/training/digit_training/digit_recognition.gd`

---

## 🤝 Need Help?

If you run into issues:

1. Check the **Output panel** in Godot for error messages
2. Review the **MNIST_USAGE_GUIDE.md** for detailed troubleshooting
3. Look at the existing working demos for reference
4. Ensure GPU/Vulkan is properly enabled

---

**Ready to classify some digits? Start with Method 1 above!** 🚀

