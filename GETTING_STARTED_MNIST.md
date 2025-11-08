# 🎯 Getting Started with MNIST Digit Classification

## ⚡ TL;DR - 30 Second Quick Start

Want to classify a handwritten digit **RIGHT NOW**?

1. Open: `examples/mnist_demo_ui.tscn`
2. Press **F5**
3. Click "Load Digit Image"
4. See your digit classified instantly! ✨

---

## 📖 What This Solution Does

**Goal:** Identify which handwritten digit (0-9) is in an image

**What you get:**
- ✅ Pre-trained model (97% accuracy, ready to use)
- ✅ Interactive UI demo
- ✅ Simple code examples
- ✅ Training pipeline for custom datasets
- ✅ Complete documentation

---

## 🎬 Three Ways to Use It

### 1️⃣ Interactive UI (Most Visual)

**Best for:** Testing images visually with instant feedback

**File:** `examples/mnist_demo_ui.tscn`

**Steps:**
1. Open the scene in Godot
2. Run (F5)
3. Click "Load Digit Image"
4. Select your image
5. See results with probability bars!

---

### 2️⃣ Simple Script (Fastest Code)

**Best for:** Quick integration into your project

**File:** `examples/simple_digit_matcher.gd`

**Steps:**
1. Open the script
2. Set `YOUR_DIGIT_IMAGE_PATH` at the top
3. Attach to any Node and run
4. See which digit is in your image!

**Example Output:**
```
🔍 SIMPLE DIGIT MATCHER
Analyzing image: res://my_digit_7.png
----------------------------------------

✅ MATCH FOUND!
======================================
The digit in your image is: 7
Confidence: 96.8%
→ Very high confidence - definitely this digit!
======================================
```

---

### 3️⃣ Code Integration (Most Flexible)

**Best for:** Building your own features

**Copy-paste this code:**

```gdscript
extends Node

func _ready():
    # Load the classifier
    var classifier = MNISTSimpleClassifier.new()
    add_child(classifier)
    classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
    classifier.use_pretrained_model = true
    
    # Wait for initialization
    await get_tree().process_frame
    
    # Classify your image
    var result = classifier.classify_from_file("res://your_digit.png")
    
    # Get the answer
    print("This is digit: %d" % result["predicted_digit"])
    print("Confidence: %.1f%%" % (result["confidence"] * 100))
```

---

## 📁 Files You Need to Know About

### Quick Start Files
- 🎮 **`mnist_demo_ui.tscn`** - Visual demo (run this first!)
- 📝 **`simple_digit_matcher.gd`** - Simplest code example
- 🚀 **`quick_test_classifier.gd`** - Minimal test script

### Core Files
- 🧠 **`mnist_simple_classifier.gd`** - Main classifier (use in your code)
- 🎓 **`digit_recognition.gd`** - Training script (if you want to train)

### Pre-trained Model
- 💾 **`assets/models/digit_recognition.tres`** - Ready-to-use trained model

### Documentation
- 📖 **`MNIST_QUICKSTART.md`** - Detailed quick start (5 min read)
- 📚 **`MNIST_USAGE_GUIDE.md`** - Complete reference
- 📋 **`MNIST_SOLUTION_SUMMARY.md`** - Overview of everything

---

## 🎯 What You Get from Classification

When you classify an image, you get a Dictionary with:

```gdscript
{
    "predicted_digit": 7,        # The digit found (0-9)
    "predicted_label": "Seven (7)",  # Human-readable
    "confidence": 0.968,         # How sure (0.0-1.0)
    "probabilities": [           # All 10 probabilities
        0.001,  # 0.1% chance it's 0
        0.002,  # 0.2% chance it's 1
        0.001,  # 0.1% chance it's 2
        ...
        0.968,  # 96.8% chance it's 7 ← Winner!
        ...
    ]
}
```

---

## 🎨 Real-World Examples

### Example 1: Basic Digit Detection

```gdscript
var result = classifier.classify_from_file("res://digit.png")
print("You wrote: %d" % result["predicted_digit"])
```

### Example 2: High-Confidence Only

```gdscript
var result = classifier.classify_from_file("res://digit.png")
if result["confidence"] > 0.9:
    print("Definitely a %d!" % result["predicted_digit"])
else:
    print("Image is unclear, please redraw")
```

### Example 3: Process Multiple Images

```gdscript
var images = ["img1.png", "img2.png", "img3.png"]
for img in images:
    var result = classifier.classify_from_file("res://" + img)
    print("%s → %d" % [img, result["predicted_digit"]])
```

### Example 4: Get Top 3 Guesses

```gdscript
var result = classifier.classify_from_file("res://digit.png")
var probs = result["probabilities"]

# Find top 3
var top = []
for i in range(10):
    top.append({"digit": i, "prob": probs[i]})
top.sort_custom(func(a, b): return a["prob"] > b["prob"])

print("Top 3 guesses:")
print("1. Digit %d: %.1f%%" % [top[0]["digit"], top[0]["prob"] * 100])
print("2. Digit %d: %.1f%%" % [top[1]["digit"], top[1]["prob"] * 100])
print("3. Digit %d: %.1f%%" % [top[2]["digit"], top[2]["prob"] * 100])
```

---

## 🏋️ Training Your Own Model (Optional)

Already have a pre-trained model, but want to train on your own data?

### Step 1: Organize Your Data

```
training_data/
├── 0/  (images of digit 0)
├── 1/  (images of digit 1)
...
└── 9/  (images of digit 9)
```

### Step 2: Configure Training

Open `examples/mnist_simple_classifier.tscn`:
- Uncheck "Use Pretrained Model"
- Set "Training Data Dir" to your folder
- Set epochs to 200
- Run the scene

### Step 3: Wait for Training

Training takes 2-10 minutes. You'll see:
```
Training...
Epoch 1/200 - Loss: 2.3
Epoch 2/200 - Loss: 1.8
...
✓ Training complete!
Test Accuracy: 97.2%
✓ Model saved to: res://trained_mnist_model.tres
```

---

## ⚡ Performance

Your framework is **GPU-accelerated** - it's FAST! ⚡

| Operation | Time |
|-----------|------|
| Load model | < 10ms |
| Classify 1 image | < 1ms |
| Classify 100 images | ~ 10ms |
| Training (9000 images, 100 epochs) | 2-5 min |

---

## ❓ Common Questions

### Q: What image format should I use?
**A:** PNG or JPG. Any size - it's auto-resized to 32×32.

### Q: Does the background color matter?
**A:** The model was trained on dark backgrounds with light digits. If your images are inverted, set `invert_image = true`.

### Q: How accurate is it?
**A:** ~97% accuracy on standard MNIST test set. Real-world accuracy depends on image quality.

### Q: Can I use this for other things?
**A:** Yes! The framework supports any image classification task. See the doodle recognition demo.

### Q: Do I need internet?
**A:** No! Everything runs locally in Godot with GPU acceleration.

### Q: What if the prediction is wrong?
**A:** Check confidence score. Low confidence means unclear image. Also check if colors need inverting.

---

## 🐛 Troubleshooting

### "Failed to load model"
→ Check path is `res://assets/models/digit_recognition.tres`

### "No model loaded"
→ Add `await get_tree().process_frame` after creating classifier

### Wrong predictions
→ Ensure digit is centered and clear
→ Try `invert_image = true` if needed

### "GPU context failed"
→ Enable Vulkan in Project Settings → Rendering

---

## 🎓 Learning Path

**Just starting?**
1. Run `mnist_demo_ui.tscn` to see it work
2. Try `simple_digit_matcher.gd` for code
3. Read `MNIST_QUICKSTART.md` for details

**Want to integrate?**
1. Copy code from "Code Integration" section above
2. Adjust paths to your images
3. Use `result["predicted_digit"]` in your logic

**Want to train?**
1. Prepare dataset as shown above
2. Configure training in scene
3. Run and wait for results

**Want to understand?**
1. Read `MNIST_USAGE_GUIDE.md`
2. Explore `/scripts/neural_network/`
3. Study the visual training demo

---

## 🚀 Next Steps

**Now that you're set up:**

✅ Test with your digit images  
✅ Integrate into your project  
✅ Experiment with different images  
✅ Try training on custom data  
✅ Build something awesome! 🎮

---

## 📞 Need More Help?

- 📖 **Quick Start:** `examples/MNIST_QUICKSTART.md`
- 📚 **Full Guide:** `examples/MNIST_USAGE_GUIDE.md`
- 📋 **Overview:** `MNIST_SOLUTION_SUMMARY.md`
- 🔧 **Framework Docs:** `README.md`

---

## 🎉 You're Ready!

You now have **everything you need** to classify handwritten digits!

**Start here:** Open `examples/mnist_demo_ui.tscn` and press F5!

---

*Godot Neural Network Framework*  
*GPU-Accelerated ML - No External Dependencies*  
*Made with ❤️ in pure Godot*

