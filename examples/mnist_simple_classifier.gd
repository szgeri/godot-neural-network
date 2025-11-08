extends Node
class_name MNISTSimpleClassifier

## Simple MNIST Digit Classifier
## This script demonstrates how to:
## 1. Train a neural network on MNIST-like digit data
## 2. Load a pre-trained model
## 3. Classify handwritten digit images
## 4. Get predictions with confidence scores

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
const IMAGE_WIDTH: int = 28
const IMAGE_HEIGHT: int = 28
const INPUT_VECTOR_SIZE: int = IMAGE_WIDTH * IMAGE_HEIGHT

@export_category("Model Settings")
@export_file("*.tres") var pretrained_model_path: String = "res://assets/models/mnist_digit_classifier.tres"
@export var use_pretrained_model: bool = true

@export_category("Training Settings (if training from scratch)")
@export_global_dir var training_data_dir: String = "/Users/szgeri/mnist_png/training"  # Should contain subdirectories: 0/, 1/, 2/, ..., 9/
@export var layer_sizes: Array[int] = [INPUT_VECTOR_SIZE, 256, 128, 10]  # 28x28 images = 784 input neurons
@export_range(0.0, 1.0) var image_scale: float = 1.0  # Use 1.0 for 28x28 images
@export_range(1, 1000) var epochs: int = 100
@export_range(1, 10000) var batch_size: int = 512
@export_range(0.001, 1.0) var learning_rate: float = 0.1

@export_category("Test Image")
@export_file("*.png", "*.jpg") var test_image_path: String  # Path to a single digit image to classify

# -------------------------------------------------------------------
# Internal State
# -------------------------------------------------------------------
var network: NeuralNetwork
var forward_runner: ForwardPassRunner
var expected_input_size: int = INPUT_VECTOR_SIZE

# Digit labels (0-9)
var digit_labels: Array[String] = [
	"Zero (0)", "One (1)", "Two (2)", "Three (3)", "Four (4)",
	"Five (5)", "Six (6)", "Seven (7)", "Eight (8)", "Nine (9)"
]

# -------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------
func _ready() -> void:
	print("=======================================")
	print("MNIST DIGIT CLASSIFIER - Simple Example")
	print("=======================================")
	
	# Initialize GPU runner
	forward_runner = ForwardPassRunner.new()
	
	if use_pretrained_model:
		load_pretrained_model()
	else:
		train_new_model()

	_validate_network_input_size()
	
	# Test the model with a single image
	if test_image_path != "":
		classify_digit_image(test_image_path)

# -------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------
func load_pretrained_model() -> void:
	print("\n[Loading Pre-trained Model]")
	print("Path: %s" % pretrained_model_path)
	
	network = NeuralNetworkSerializer.import(forward_runner, pretrained_model_path)
	
	if network:
		print_rich("[color=green]✓ Model loaded successfully![/color]")
		print("Layers: %d" % network.layers.size())
		for i in range(network.layers.size()):
			var layer = network.layers[i]
			print("  Layer %d: %d → %d neurons" % [i, layer.input_size, layer.output_size])
	else:
		push_error("Failed to load model!")


func _validate_network_input_size() -> void:
	if network == null:
		return
	if network.layers.is_empty():
		return

	var network_input_size: int = network.layers[0].input_size
	if network_input_size <= 0:
		return

	if network_input_size != expected_input_size:
		expected_input_size = network_input_size
		_ensure_layer_sizes_match_input()
		print_rich(
			"[color=yellow]Adjusted expected input size to %d based on loaded model.[/color]"
			% expected_input_size
		)


func _ensure_layer_sizes_match_input() -> void:
	if layer_sizes.is_empty():
		layer_sizes = [expected_input_size, 256, 128, 10]
		return

	if layer_sizes[0] != expected_input_size:
		layer_sizes[0] = expected_input_size

# -------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------
func train_new_model() -> void:
	print("\n[Training New Model]")
	
	if training_data_dir == "":
		push_error("Training data directory not set! Please set training_data_dir in the inspector.")
		return
	
	print("Loading training data from: %s" % training_data_dir)
	
	# Load dataset
	var training_inputs: Array[PackedFloat32Array] = []
	var training_targets: Array[PackedFloat32Array] = []
	
	# Load images from each digit directory (0-9)
	for digit in range(10):
		var digit_dir = training_data_dir.path_join(str(digit))
		print("Loading digit %d from: %s" % [digit, digit_dir])
		
		var images = ImageUtils.read_images(digit_dir, image_scale)
		print("  Found %d images" % images.size())
		
		for img in images:
			training_inputs.append(img)
			# Create one-hot encoded target
			var target = PackedFloat32Array()
			target.resize(10)
			target.fill(0.0)
			target[digit] = 1.0
			training_targets.append(target)
	
	print("Total training samples: %d" % training_inputs.size())

	if training_inputs.is_empty():
		push_error("No training data found in %s" % training_data_dir)
		return

	expected_input_size = training_inputs[0].size()
	_ensure_layer_sizes_match_input()
	
	# Create network
	print("\nCreating neural network...")
	network = NeuralNetwork.new({
		ConfigKeys.NETWORK.LAYER_SIZES: layer_sizes,
		ConfigKeys.NETWORK.RUNNER: forward_runner,
		ConfigKeys.NETWORK.HIDDEN_ACT: Activations.Type.TANH,
		ConfigKeys.NETWORK.OUTPUT_ACT: Activations.Type.SOFTMAX,
		ConfigKeys.NETWORK.WEIGHT_INIT: NetworkLayer.WeightInitialization.XAVIER
	})
	
	# Split into train/test sets
	var split: DataSplit = DataSetUtils.train_test_split(
		training_inputs,
		training_targets,
		0.2  # 20% test set
	)
	
	print("Training set: %d samples" % split.train_inputs.size())
	print("Test set: %d samples" % split.test_inputs.size())
	
	# Create trainer
	var backward_runner: BackwardPassRunner = BackwardPassRunner.new()
	var trainer: Trainer = Trainer.new({
		ConfigKeys.TRAINER.NETWORK: network,
		ConfigKeys.TRAINER.RUNNER: backward_runner,
		ConfigKeys.TRAINER.LOSS: Loss.Type.CCE,  # Categorical Cross-Entropy for multi-class
		ConfigKeys.TRAINER.LEARNING_RATE: learning_rate,
		ConfigKeys.TRAINER.LAMBDA_L2: 0.0001,
		ConfigKeys.TRAINER.EPOCHS: epochs,
		ConfigKeys.TRAINER.BATCH_SIZE: batch_size,
	})
	
	# Train
	print("\nStarting training...")
	var start_time = Time.get_ticks_msec()
	trainer.train(split.train_inputs, split.train_targets)
	var training_time = Time.get_ticks_msec() - start_time
	
	print_rich("[color=green]✓ Training completed in %d ms[/color]" % training_time)
	
	# Evaluate
	print("\nEvaluating model...")
	var test_acc = ModelEvaluator.evaluate_model_soft_max(
		network,
		split.test_inputs,
		split.test_targets
	)
	print_rich("[color=cyan]Test Accuracy: %.2f%%[/color]" % (test_acc * 100.0))
	
	# Save model
	NeuralNetworkSerializer.export(network, pretrained_model_path)
	print_rich("[color=green]✓ Model saved to: %s[/color]" % pretrained_model_path)

# -------------------------------------------------------------------
# Image Classification
# -------------------------------------------------------------------
func classify_digit_image(image_path: String) -> Dictionary:
	"""
	Classifies a single digit image and returns prediction results.
	
	Returns a Dictionary with:
	  - "predicted_digit": int (0-9)
	  - "predicted_label": String (e.g., "Seven (7)")
	  - "confidence": float (0.0 - 1.0)
	  - "probabilities": PackedFloat32Array (all 10 class probabilities)
	"""
	print("=======================================")
	print("[Classifying Image]")
	print("Path: %s" % image_path)
	
	if not network:
		push_error("No model loaded!")
		return {}
	
	# Load and preprocess the image
	var image_data = ImageUtils.read_image(image_path, image_scale)
	
	if image_data.size() == 0:
		push_error("Failed to load image!")
		return {}

	if image_data.size() != expected_input_size:
		push_error(
			"Image has %d features but expected %d. Check preprocessing."
			% [image_data.size(), expected_input_size]
		)
		return {}
	
	print("Image processed: %d features" % image_data.size())
	
	# Run forward pass
	var predictions: PackedFloat32Array = network.forward_pass([image_data])
	
	# Get the predicted class (argmax)
	var predicted_digit = ModelEvaluator.find_max_value_index(predictions)
	var confidence = predictions[predicted_digit]
	
	# Print results
	print("\n=======================================")
	print_rich("[color=yellow][b]PREDICTION RESULT[/b][/color]")
	print_rich("[color=green]Predicted Digit: %d[/color]" % predicted_digit)
	print_rich("[color=green]Confidence: %.2f%%[/color]" % (confidence * 100.0))
	print("=======================================")
	
	# Print all probabilities
	print("\nClass Probabilities:")
	for i in range(predictions.size()):
		var prob = predictions[i]
		var bar_length = int(prob * 50)
		var filled = ""
		var empty = ""
		for j in range(bar_length):
			filled += "█"
		for j in range(50 - bar_length):
			empty += "░"
		var bar = filled + empty
		var color = "green" if i == predicted_digit else "white"
		print_rich("  [color=%s]%d: %s %.2f%%[/color]" % [color, i, bar, prob * 100.0])
	
	print("=======================================")
	
	# Return results as dictionary
	return {
		"predicted_digit": predicted_digit,
		"predicted_label": digit_labels[predicted_digit],
		"confidence": confidence,
		"probabilities": predictions
	}

# -------------------------------------------------------------------
# Public API Functions
# -------------------------------------------------------------------

## Classify a digit from a file path
func classify_from_file(path: String) -> Dictionary:
	return classify_digit_image(path)

## Classify a digit from raw image data (PackedFloat32Array)
func classify_from_data(image_data: PackedFloat32Array) -> Dictionary:
	if not network:
		push_error("No model loaded!")
		return {}

	if image_data.size() != expected_input_size:
		push_error(
			"Input has %d features but expected %d. Check preprocessing."
			% [image_data.size(), expected_input_size]
		)
		return {}
	
	var predictions: PackedFloat32Array = network.forward_pass([image_data])
	var predicted_digit = ModelEvaluator.find_max_value_index(predictions)
	
	return {
		"predicted_digit": predicted_digit,
		"predicted_label": digit_labels[predicted_digit],
		"confidence": predictions[predicted_digit],
		"probabilities": predictions
	}

## Get the loaded network (for advanced usage)
func get_network() -> NeuralNetwork:
	return network
