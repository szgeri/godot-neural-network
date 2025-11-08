extends Control

## Interactive MNIST Classifier Demo UI
## This creates a simple visual interface for digit classification
## with image preview and confidence display

# UI Components
var image_preview: TextureRect
var result_label: Label
var confidence_label: Label
var load_button: Button
var probabilities_container: VBoxContainer

# Classifier
var classifier: MNISTSimpleClassifier
var current_image_path: String = ""

func _ready() -> void:
	_create_ui()
	_initialize_classifier()

# -------------------------------------------------------------------
# UI Creation
# -------------------------------------------------------------------
func _create_ui() -> void:
	# Main container
	var main_vbox = VBoxContainer.new()
	main_vbox.set_anchors_preset(Control.PRESET_FULL_RECT)
	main_vbox.add_theme_constant_override("separation", 20)
	add_child(main_vbox)
	
	# Title
	var title = Label.new()
	title.text = "MNIST Digit Classifier Demo"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 32)
	main_vbox.add_child(title)
	
	# Horizontal split
	var hbox = HBoxContainer.new()
	hbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	hbox.add_theme_constant_override("separation", 40)
	main_vbox.add_child(hbox)
	
	# Left side - Image preview
	var left_panel = VBoxContainer.new()
	left_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hbox.add_child(left_panel)
	
	var preview_label = Label.new()
	preview_label.text = "Image Preview"
	preview_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	left_panel.add_child(preview_label)
	
	image_preview = TextureRect.new()
	image_preview.custom_minimum_size = Vector2(300, 300)
	image_preview.expand_mode = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
	image_preview.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	left_panel.add_child(image_preview)
	
	load_button = Button.new()
	load_button.text = "Load Digit Image"
	load_button.pressed.connect(_on_load_button_pressed)
	left_panel.add_child(load_button)
	
	# Right side - Results
	var right_panel = VBoxContainer.new()
	right_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hbox.add_child(right_panel)
	
	var results_title = Label.new()
	results_title.text = "Classification Results"
	results_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	right_panel.add_child(results_title)
	
	result_label = Label.new()
	result_label.text = "No image loaded"
	result_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	result_label.add_theme_font_size_override("font_size", 48)
	result_label.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	right_panel.add_child(result_label)
	
	confidence_label = Label.new()
	confidence_label.text = ""
	confidence_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	confidence_label.add_theme_font_size_override("font_size", 24)
	right_panel.add_child(confidence_label)
	
	var separator = HSeparator.new()
	right_panel.add_child(separator)
	
	var prob_title = Label.new()
	prob_title.text = "All Probabilities:"
	prob_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	right_panel.add_child(prob_title)
	
	# Scrollable probabilities
	var scroll = ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right_panel.add_child(scroll)
	
	probabilities_container = VBoxContainer.new()
	probabilities_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(probabilities_container)

# -------------------------------------------------------------------
# Classifier Initialization
# -------------------------------------------------------------------
func _initialize_classifier() -> void:
	classifier = MNISTSimpleClassifier.new()
	add_child(classifier)
	
	# classifier.pretrained_model_path = "res://assets/models/digit_recognition.tres"
	classifier.pretrained_model_path = "res://assets/models/mnist_digit_classifier.tres"
	classifier.use_pretrained_model = true
	classifier.test_image_path = ""
	
	print("Classifier initialized!")

# -------------------------------------------------------------------
# Event Handlers
# -------------------------------------------------------------------
func _on_load_button_pressed() -> void:
	# Create file dialog
	var file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	file_dialog.filters = PackedStringArray([
		"*.png ; PNG Images",
		"*.jpg,*.jpeg ; JPEG Images"
	])
	file_dialog.file_selected.connect(_on_file_selected)
	add_child(file_dialog)
	file_dialog.popup_centered(Vector2(800, 600))

func _on_file_selected(path: String) -> void:
	current_image_path = path
	_classify_current_image()

# -------------------------------------------------------------------
# Classification
# -------------------------------------------------------------------
func _classify_current_image() -> void:
	if current_image_path == "":
		return
	
	# Load and display image
	var image = Image.load_from_file(current_image_path)
	if image:
		var texture = ImageTexture.create_from_image(image)
		image_preview.texture = texture
	
	# Wait a frame for classifier to be ready
	await get_tree().process_frame
	
	# Classify
	var result = classifier.classify_from_file(current_image_path)
	
	if result.is_empty():
		result_label.text = "Error"
		confidence_label.text = "Failed to classify"
		return
	
	# Update UI
	_display_results(result)

# -------------------------------------------------------------------
# UI Update
# -------------------------------------------------------------------
func _display_results(result: Dictionary) -> void:
	# Main prediction
	result_label.text = "Digit: %d" % result["predicted_digit"]
	confidence_label.text = "Confidence: %.1f%%" % (result["confidence"] * 100)
	
	# Update confidence label color based on confidence
	var conf = result["confidence"]
	if conf > 0.9:
		confidence_label.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	elif conf > 0.7:
		confidence_label.add_theme_color_override("font_color", Color(0.8, 0.8, 0.2))
	else:
		confidence_label.add_theme_color_override("font_color", Color(0.8, 0.2, 0.2))
	
	# Clear previous probabilities
	for child in probabilities_container.get_children():
		child.queue_free()
	
	# Display all probabilities
	var probs = result["probabilities"]
	for i in range(probs.size()):
		var prob = probs[i]
		var prob_item = _create_probability_item(i, prob, i == result["predicted_digit"])
		probabilities_container.add_child(prob_item)

func _create_probability_item(digit: int, probability: float, is_predicted: bool) -> HBoxContainer:
	var hbox = HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 10)
	
	# Digit label
	var label = Label.new()
	label.text = "Digit %d:" % digit
	label.custom_minimum_size.x = 80
	if is_predicted:
		label.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	hbox.add_child(label)
	
	# Progress bar
	var progress = ProgressBar.new()
	progress.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	progress.custom_minimum_size = Vector2(200, 24)
	progress.max_value = 1.0
	progress.value = probability
	progress.show_percentage = false
	hbox.add_child(progress)
	
	# Percentage label
	var percent = Label.new()
	percent.text = "%.2f%%" % (probability * 100)
	percent.custom_minimum_size.x = 70
	percent.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	if is_predicted:
		percent.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	hbox.add_child(percent)
	
	return hbox
