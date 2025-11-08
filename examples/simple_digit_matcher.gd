extends Node

## Simple Digit Matcher - Direct Example
## 
## This script answers the question: "I want to match an image and get the correct class"
## 
## USAGE:
## 1. Set your_digit_image_path below
## 2. Run this script
## 3. See which digit is in your image!

# ====================================================================
# CONFIGURE THIS: Set the path to your digit image
# ====================================================================
const YOUR_DIGIT_IMAGE_PATH = "res://path/to/your/digit_image.png"

# Or use export to set it in the Inspector
@export_file("*.png", "*.jpg") var digit_image_path: String = ""

# ====================================================================
# The classifier will tell you which digit (0-9) is in your image
# ====================================================================

func _ready() -> void:
	# Determine which image to use
	var image_to_classify = digit_image_path if digit_image_path != "" else YOUR_DIGIT_IMAGE_PATH
	
	if image_to_classify == "" or image_to_classify == "res://path/to/your/digit_image.png":
		var sep: String = ""
		for i in range(70):
			sep += "="
		print(sep)
		print("⚠️  PLEASE SET YOUR IMAGE PATH")
		print(sep)
		print("Edit this script and set YOUR_DIGIT_IMAGE_PATH")
		print("OR")
		print("Set 'digit_image_path' in the Inspector panel")
		print(sep)
		return
	
	var sep_eq: String = ""
	var sep_dash: String = ""
	for i in range(70):
		sep_eq += "="
		sep_dash += "-"
	print("\n" + sep_eq)
	print("🔍 SIMPLE DIGIT MATCHER")
	print(sep_eq)
	print("Analyzing image: %s" % image_to_classify)
	print(sep_dash)
	
	# Initialize the classifier with pre-trained model
	var classifier = MNISTSimpleClassifier.new()
	add_child(classifier)
	classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
	classifier.use_pretrained_model = true
	
	# Wait for initialization
	await get_tree().process_frame
	
	# Match the image to find which digit it is
	var result = match_digit_image(classifier, image_to_classify)
	
	# Display the answer
	print_result(result)

# ====================================================================
# Match an image to find which digit (0-9) it contains
# ====================================================================
func match_digit_image(classifier: MNISTSimpleClassifier, image_path: String) -> Dictionary:
	"""
	Takes an image and tells you which digit is in it.
	
	Returns:
		Dictionary with:
		- "digit": The digit found (0-9)
		- "confidence": How sure we are (0-100%)
		- "is_confident": True if confidence > 90%
	"""
	var result = classifier.classify_from_file(image_path)
	
	if result.is_empty():
		return {"digit": -1, "confidence": 0.0, "is_confident": false, "error": true}
	
	return {
		"digit": result["predicted_digit"],
		"confidence": result["confidence"] * 100.0,
		"is_confident": result["confidence"] > 0.9,
		"all_probabilities": result["probabilities"],
		"error": false
	}

# ====================================================================
# Display the result in a clear, readable format
# ====================================================================
func print_result(result: Dictionary) -> void:
	if result.get("error", false):
		print_rich("[color=red]❌ ERROR: Could not classify image[/color]")
		return
	
	var separator: String = ""
	for i in range(70):
		separator += "="
	print("\n" + separator)
	print_rich("[color=cyan]✅ MATCH FOUND![/color]")
	print(separator)
	
	# Main answer
	print_rich("[color=yellow][b]The digit in your image is: %d[/b][/color]" % result["digit"])
	
	# Confidence
	var confidence: float = result["confidence"]
	var confidence_color: String = "green" if result["is_confident"] else "orange"
	print_rich("[color=%s]Confidence: %.1f%%[/color]" % [confidence_color, confidence])
	
	# Confidence level description
	if confidence > 95:
		print_rich("[color=green]→ Very high confidence - definitely this digit![/color]")
	elif confidence > 80:
		print_rich("[color=green]→ High confidence - very likely this digit[/color]")
	elif confidence > 60:
		print_rich("[color=orange]→ Moderate confidence - probably this digit[/color]")
	else:
		print_rich("[color=orange]→ Low confidence - image might be unclear[/color]")
	
	print(separator)
	
	# Show alternative possibilities (top 3)
	print("\nAlternative possibilities:")
	var probs = result["all_probabilities"]
	var alternatives = []
	for i in range(10):
		if i != result["digit"]:  # Skip the main prediction
			alternatives.append({"digit": i, "prob": probs[i]})
	alternatives.sort_custom(func(a, b): return a["prob"] > b["prob"])
	
	for i in range(min(3, alternatives.size())):
		var alt = alternatives[i]
		print("  %d. Digit %d: %.1f%%" % [i+1, alt["digit"], alt["prob"] * 100])
	
	# Reuse separator from above
	print(separator + "\n")

# ====================================================================
# EXAMPLE: How to use this programmatically
# ====================================================================
func example_usage():
	"""
	This function shows how you can use digit matching in your own code
	"""
	var classifier = MNISTSimpleClassifier.new()
	add_child(classifier)
	classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
	classifier.use_pretrained_model = true
	await get_tree().process_frame
	
	# Example 1: Match a single image
	var result = match_digit_image(classifier, "res://my_digit.png")
	if not result["error"]:
		print("Found digit: %d" % result["digit"])
	
	# Example 2: Match multiple images
	var images = ["res://img1.png", "res://img2.png", "res://img3.png"]
	for img_path in images:
		var r = match_digit_image(classifier, img_path)
		if not r["error"]:
			print("%s contains digit: %d (%.0f%% confident)" % [
				img_path, r["digit"], r["confidence"]
			])
	
	# Example 3: Only accept high-confidence matches
	var r = match_digit_image(classifier, "res://digit.png")
	if not r["error"] and r["is_confident"]:
		print("High-confidence match: %d" % r["digit"])
	else:
		print("Image unclear or low confidence")
	
	# Example 4: Get the actual digit class
	r = match_digit_image(classifier, "res://digit.png")
	var detected_digit = r["digit"]  # This is the class (0-9)
	
	# Use it in your logic
	match detected_digit:
		0:
			print("You wrote zero!")
		1:
			print("You wrote one!")
		7:
			print("You wrote seven!")
		_:
			print("You wrote: %d" % detected_digit)

