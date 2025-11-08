extends Node

## Quick Test Script for MNIST Digit Classification
## 
## HOW TO USE:
## 1. Attach this script to any Node in your scene
## 2. Set test_image_path in the Inspector to your digit image
## 3. Run the scene - results appear in Output panel
##
## This demonstrates the simplest possible way to classify a digit image.

@export_file("*.png", "*.jpg") var test_image_path: String = ""

func _ready() -> void:
	if test_image_path == "":
		print("No test image specified. Set test_image_path in the Inspector.")
		return
	
	# Create and initialize classifier
	var classifier = MNISTSimpleClassifier.new()
	add_child(classifier)
	
	# Configure to use pre-trained model
	classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
	classifier.use_pretrained_model = true
	classifier.test_image_path = ""  # We'll classify manually
	
	# Wait for initialization
	await get_tree().process_frame
	
	# Classify the test image
	var result = classifier.classify_from_file(test_image_path)
	
	if result.is_empty():
		print("Classification failed!")
		return
	
	# Print simple result
	var separator: String = ""
	for i in range(50):
		separator += "="
	print("\n" + separator)
	print("QUICK CLASSIFICATION RESULT")
	print(separator)
	print("Image: %s" % test_image_path)
	print("Predicted Digit: %d" % result["predicted_digit"])
	print("Confidence: %.1f%%" % (result["confidence"] * 100))
	print(separator)
	
	# Get top 3 predictions
	print("\nTop 3 Predictions:")
	var predictions = []
	for i in range(result["probabilities"].size()):
		predictions.append({"digit": i, "prob": result["probabilities"][i]})
	predictions.sort_custom(func(a, b): return a["prob"] > b["prob"])
	
	for i in range(min(3, predictions.size())):
		var pred = predictions[i]
		print("  %d. Digit %d: %.2f%%" % [i+1, pred["digit"], pred["prob"] * 100])

