extends Control

## Interactive MNIST Drawing Classifier
## Draw a digit on the canvas, and the classifier will recognize it in real-time
## Features:
## - Drawing canvas with mouse/touch input
## - Automatic bounding box detection and cropping
## - Smart resizing to 28x28 for the neural network
## - Live classification results with confidence scores

# -------------------------------------------------------------------
# UI Components
# -------------------------------------------------------------------
var canvas: Control
var canvas_texture_rect: TextureRect
var canvas_texture: ImageTexture
var hint_label: Label
var clear_button: Button
var classify_button: Button
var result_label: Label
var confidence_label: Label
var probabilities_container: VBoxContainer
var preview_texture: TextureRect

# -------------------------------------------------------------------
# Drawing State
# -------------------------------------------------------------------
var is_drawing: bool = false
var drawing_data: Array[Vector2] = []
var drawn_image: Image
var canvas_size: Vector2i = Vector2i(280, 280)
var brush_size: float = 25.0  # Increased for bolder strokes
var last_draw_position: Vector2
var has_last_draw_position: bool = false

# -------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------
const MODEL_IMAGE_SIZE: int = 28

# -------------------------------------------------------------------
# Classifier
# -------------------------------------------------------------------
var classifier: MNISTSimpleClassifier

# -------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------
func _ready() -> void:
	_create_ui()
	_initialize_canvas()
	_initialize_classifier()

# -------------------------------------------------------------------
# UI Creation
# -------------------------------------------------------------------
func _create_ui() -> void:
	# Main container
	var main_vbox: VBoxContainer = VBoxContainer.new()
	main_vbox.set_anchors_preset(Control.PRESET_FULL_RECT)
	main_vbox.add_theme_constant_override("separation", 20)
	add_child(main_vbox)
	
	# Title
	var title: Label = Label.new()
	title.text = "Draw a Digit (0-9)"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 32)
	main_vbox.add_child(title)
	
	# Main content area
	var content_hbox: HBoxContainer = HBoxContainer.new()
	content_hbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	content_hbox.add_theme_constant_override("separation", 30)
	main_vbox.add_child(content_hbox)
	
	# Left panel - Drawing canvas
	var left_panel: VBoxContainer = VBoxContainer.new()
	left_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	content_hbox.add_child(left_panel)
	
	var canvas_label: Label = Label.new()
	canvas_label.text = "Drawing Canvas"
	canvas_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	left_panel.add_child(canvas_label)
	
	# Drawing canvas container (captures input)
	canvas = Control.new()
	canvas.custom_minimum_size = Vector2(canvas_size.x, canvas_size.y)
	canvas.size = canvas.custom_minimum_size
	canvas.mouse_filter = Control.MOUSE_FILTER_STOP
	canvas.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	canvas.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	canvas.set_anchors_preset(Control.PRESET_TOP_LEFT)
	left_panel.add_child(canvas)
	
	# Texture that displays the drawn image
	canvas_texture_rect = TextureRect.new()
	canvas_texture_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	canvas_texture_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	canvas_texture_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	canvas_texture_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	canvas_texture_rect.texture_filter = TextureRect.TEXTURE_FILTER_NEAREST
	canvas.add_child(canvas_texture_rect)
	
	# Hint label shown until the user starts drawing
	hint_label = Label.new()
	hint_label.set_anchors_preset(Control.PRESET_FULL_RECT)
	hint_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	hint_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hint_label.text = "Click and drag to draw"
	hint_label.add_theme_font_size_override("font_size", 18)
	hint_label.modulate = Color(0.4, 0.4, 0.4, 0.9)
	canvas.add_child(hint_label)
	
	# Buttons
	var button_hbox: HBoxContainer = HBoxContainer.new()
	button_hbox.alignment = BoxContainer.ALIGNMENT_CENTER
	button_hbox.add_theme_constant_override("separation", 10)
	left_panel.add_child(button_hbox)
	
	clear_button = Button.new()
	clear_button.text = "Clear Canvas"
	clear_button.pressed.connect(_on_clear_pressed)
	button_hbox.add_child(clear_button)
	
	classify_button = Button.new()
	classify_button.text = "Classify Digit"
	classify_button.pressed.connect(_on_classify_pressed)
	button_hbox.add_child(classify_button)
	
	# Right panel - Results
	var right_panel: VBoxContainer = VBoxContainer.new()
	right_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	content_hbox.add_child(right_panel)
	
	var results_title: Label = Label.new()
	results_title.text = "Classification Results"
	results_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	right_panel.add_child(results_title)
	
	# Preview of processed image
	var preview_label: Label = Label.new()
	preview_label.text = "Processed Image (28x28)"
	preview_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	right_panel.add_child(preview_label)
	
	preview_texture = TextureRect.new()
	preview_texture.custom_minimum_size = Vector2(128, 128)
	preview_texture.expand_mode = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
	preview_texture.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	right_panel.add_child(preview_texture)
	
	result_label = Label.new()
	result_label.text = "Draw something!"
	result_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	result_label.add_theme_font_size_override("font_size", 48)
	result_label.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	right_panel.add_child(result_label)
	
	confidence_label = Label.new()
	confidence_label.text = ""
	confidence_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	confidence_label.add_theme_font_size_override("font_size", 24)
	right_panel.add_child(confidence_label)
	
	var separator: HSeparator = HSeparator.new()
	right_panel.add_child(separator)
	
	var prob_title: Label = Label.new()
	prob_title.text = "All Probabilities:"
	prob_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	right_panel.add_child(prob_title)
	
	# Scrollable probabilities
	var scroll: ScrollContainer = ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right_panel.add_child(scroll)
	
	probabilities_container = VBoxContainer.new()
	probabilities_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(probabilities_container)

# -------------------------------------------------------------------
# Canvas Initialization
# -------------------------------------------------------------------
func _initialize_canvas() -> void:
	# Create the drawing image (white background, black strokes)
	drawn_image = Image.create(canvas_size.x, canvas_size.y, false, Image.FORMAT_RGBA8)
	drawn_image.fill(Color(1, 1, 1, 1))
	
	# Create texture and assign to texture rect
	canvas_texture = ImageTexture.create_from_image(drawn_image)
	canvas_texture_rect.texture = canvas_texture
	
	# Connect input handler
	canvas.gui_input.connect(_on_canvas_input)
	
	hint_label.visible = true
	is_drawing = false
	
	# Force canvas to be the right size
	canvas.size = Vector2(canvas_size.x, canvas_size.y)

# -------------------------------------------------------------------
# Classifier Initialization
# -------------------------------------------------------------------
func _initialize_classifier() -> void:
	classifier = MNISTSimpleClassifier.new()
	classifier.pretrained_model_path = "res://assets/models/mnist_digit_classifier.tres"
	classifier.use_pretrained_model = true
	classifier.test_image_path = ""
	classifier.image_scale = 1.0  # We'll provide 28x28 images
	
	add_child(classifier)
	
	print("Drawing classifier initialized!")

# -------------------------------------------------------------------
# Input Handling
# -------------------------------------------------------------------
func _on_canvas_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mouse_event: InputEventMouseButton = event as InputEventMouseButton
		if mouse_event.button_index == MOUSE_BUTTON_LEFT:
			is_drawing = mouse_event.pressed
			if is_drawing:
				has_last_draw_position = true
				last_draw_position = mouse_event.position
				_draw_line_segment(last_draw_position, mouse_event.position)
			else:
				has_last_draw_position = false
	
	elif event is InputEventMouseMotion:
		if is_drawing and has_last_draw_position:
			_draw_line_segment(last_draw_position, event.position)
			last_draw_position = event.position

func _draw_line_segment(from_pos: Vector2, to_pos: Vector2) -> void:
	var distance: float = from_pos.distance_to(to_pos)
	var step: float = max(1.0, brush_size * 0.35)
	var steps: int = int(ceil(distance / step))
	if steps <= 0:
		steps = 1
	
	for i in range(steps + 1):
		var t: float = float(i) / float(steps)
		var pos: Vector2 = from_pos.lerp(to_pos, t)
		_stamp_brush(pos)
	
	_update_canvas_texture()
	hint_label.visible = false

func _stamp_brush(pos: Vector2) -> void:
	var x_start: int = maxi(0, int(pos.x - brush_size / 2))
	var y_start: int = maxi(0, int(pos.y - brush_size / 2))
	var x_end: int = mini(canvas_size.x, int(pos.x + brush_size / 2))
	var y_end: int = mini(canvas_size.y, int(pos.y + brush_size / 2))
	
	for y in range(y_start, y_end):
		for x in range(x_start, x_end):
			var dx: float = x - pos.x
			var dy: float = y - pos.y
			var dist: float = sqrt(dx * dx + dy * dy)
			
			if dist < brush_size / 2:
				var alpha: float = 1.0 - (dist / (brush_size / 2))
				alpha = alpha * alpha
				alpha = maxf(alpha, 0.15)
				var current_value: float = drawn_image.get_pixel(x, y).r
				var new_value: float = maxf(0.0, current_value - alpha * 0.9)
				drawn_image.set_pixel(x, y, Color(new_value, new_value, new_value, 1.0))

func _update_canvas_texture() -> void:
	canvas_texture = ImageTexture.create_from_image(drawn_image)
	canvas_texture_rect.texture = canvas_texture

# -------------------------------------------------------------------
# Button Handlers
# -------------------------------------------------------------------
func _on_clear_pressed() -> void:
	drawn_image.fill(Color(1, 1, 1, 1))
	_update_canvas_texture()
	hint_label.visible = true
	is_drawing = false
	has_last_draw_position = false
	
	result_label.text = "Draw something!"
	confidence_label.text = ""
	preview_texture.texture = null
	
	# Clear probabilities
	for child in probabilities_container.get_children():
		child.queue_free()

func _on_classify_pressed() -> void:
	_classify_drawing()

# -------------------------------------------------------------------
# Image Processing
# -------------------------------------------------------------------
func _classify_drawing() -> void:
	# Find bounding box of drawn content
	var bounds: Rect2i = _find_bounding_box()
	
	if bounds.size.x == 0 or bounds.size.y == 0:
		result_label.text = "Draw something first!"
		confidence_label.text = ""
		return
	
	# MNIST preprocessing: match the original MNIST preparation method
	var model_image: Image = _preprocess_like_mnist(drawn_image, bounds)
	
	# Show preview (what the network sees)
	var preview: Image = model_image.duplicate()
	preview.resize(128, 128, Image.INTERPOLATE_NEAREST)
	preview_texture.texture = ImageTexture.create_from_image(preview)
	
	# Convert to neural network input format
	var input_data: PackedFloat32Array = _image_to_input_array(model_image)
	
	# Classify
	var result: Dictionary = classifier.classify_from_data(input_data)
	
	if result.is_empty():
		result_label.text = "Error"
		confidence_label.text = "Failed to classify"
		return
	
	# Display results
	_display_results(result)

func _find_bounding_box() -> Rect2i:
	var min_x: int = canvas_size.x
	var min_y: int = canvas_size.y
	var max_x: int = 0
	var max_y: int = 0
	var found_pixel: bool = false
	
	for y in range(canvas_size.y):
		for x in range(canvas_size.x):
			var color: Color = drawn_image.get_pixel(x, y)
			# Detect pixels that are significantly darker than the white background
			if color.r < 0.95:
				found_pixel = true
				min_x = mini(min_x, x)
				min_y = mini(min_y, y)
				max_x = maxi(max_x, x)
				max_y = maxi(max_y, y)
	
	if not found_pixel:
		return Rect2i(0, 0, 0, 0)
	
	return Rect2i(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

func _crop_image(img: Image, rect: Rect2i) -> Image:
	var cropped: Image = Image.create(rect.size.x, rect.size.y, false, Image.FORMAT_RGBA8)
	
	for y in range(rect.size.y):
		for x in range(rect.size.x):
			var source_x: int = rect.position.x + x
			var source_y: int = rect.position.y + y
			cropped.set_pixel(x, y, img.get_pixel(source_x, source_y))
	
	return cropped

func _preprocess_like_mnist(img: Image, bounds: Rect2i) -> Image:
	# Step 1: Crop to bounding box
	var cropped: Image = _crop_image(img, bounds)
	
	# Step 2: Resize to fit in 20x20 while preserving aspect ratio
	var width: int = cropped.get_width()
	var height: int = cropped.get_height()
	var max_dim: int = maxi(width, height)
	var scale: float = 20.0 / float(max_dim)
	var new_width: int = int(width * scale)
	var new_height: int = int(height * scale)
	
	cropped.resize(new_width, new_height, Image.INTERPOLATE_LANCZOS)
	
	# Step 3: Invert colors (drawing is dark on white, MNIST is white on black)
	var inverted: Image = Image.create(new_width, new_height, false, Image.FORMAT_RGBA8)
	for y in range(new_height):
		for x in range(new_width):
			var color: Color = cropped.get_pixel(x, y)
			var grayscale: float = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
			var inv: float = 1.0 - grayscale
			inverted.set_pixel(x, y, Color(inv, inv, inv, 1.0))
	
	# Step 4: Calculate center of mass
	var com: Vector2 = _calculate_center_of_mass(inverted)
	
	# Step 5: Place in 28x28 canvas, centered by center of mass
	var final_image: Image = Image.create(28, 28, false, Image.FORMAT_RGBA8)
	final_image.fill(Color(0, 0, 0, 1))  # Black background (MNIST style)
	
	# Calculate offset to center the center of mass at (14, 14)
	var offset_x: int = int(14.0 - com.x)
	var offset_y: int = int(14.0 - com.y)
	
	# Copy the resized image into the final canvas
	for y in range(new_height):
		for x in range(new_width):
			var dest_x: int = x + offset_x
			var dest_y: int = y + offset_y
			if dest_x >= 0 and dest_x < 28 and dest_y >= 0 and dest_y < 28:
				final_image.set_pixel(dest_x, dest_y, inverted.get_pixel(x, y))
	
	return final_image

func _calculate_center_of_mass(img: Image) -> Vector2:
	var total_mass: float = 0.0
	var sum_x: float = 0.0
	var sum_y: float = 0.0
	
	for y in range(img.get_height()):
		for x in range(img.get_width()):
			var color: Color = img.get_pixel(x, y)
			var intensity: float = color.r  # Already grayscale
			total_mass += intensity
			sum_x += float(x) * intensity
			sum_y += float(y) * intensity
	
	if total_mass == 0.0:
		return Vector2(img.get_width() / 2.0, img.get_height() / 2.0)
	
	return Vector2(sum_x / total_mass, sum_y / total_mass)


func _image_to_input_array(img: Image) -> PackedFloat32Array:
	var result: PackedFloat32Array = PackedFloat32Array()
	result.resize(MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE)
	
	for y in range(MODEL_IMAGE_SIZE):
		for x in range(MODEL_IMAGE_SIZE):
			var color: Color = img.get_pixel(x, y)
			# Convert to grayscale [0, 1]
			var grayscale: float = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
			# Normalize to [-1, 1] range (matches training normalization)
			var normalized: float = (grayscale - 0.5) * 2.0
			result[y * MODEL_IMAGE_SIZE + x] = normalized
	
	return result

# -------------------------------------------------------------------
# Results Display
# -------------------------------------------------------------------
func _display_results(result: Dictionary) -> void:
	# Main prediction
	result_label.text = "Digit: %d" % result["predicted_digit"]
	confidence_label.text = "Confidence: %.1f%%" % (result["confidence"] * 100)
	
	# Update confidence label color based on confidence
	var conf: float = result["confidence"]
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
	var probs: PackedFloat32Array = result["probabilities"]
	for i in range(probs.size()):
		var prob: float = probs[i]
		var prob_item: HBoxContainer = _create_probability_item(i, prob, i == result["predicted_digit"])
		probabilities_container.add_child(prob_item)

func _create_probability_item(digit: int, probability: float, is_predicted: bool) -> HBoxContainer:
	var hbox: HBoxContainer = HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 10)
	
	# Digit label
	var label: Label = Label.new()
	label.text = "Digit %d:" % digit
	label.custom_minimum_size.x = 80
	if is_predicted:
		label.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	hbox.add_child(label)
	
	# Progress bar
	var progress: ProgressBar = ProgressBar.new()
	progress.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	progress.custom_minimum_size = Vector2(200, 24)
	progress.max_value = 1.0
	progress.value = probability
	progress.show_percentage = false
	hbox.add_child(progress)
	
	# Percentage label
	var percent: Label = Label.new()
	percent.text = "%.2f%%" % (probability * 100)
	percent.custom_minimum_size.x = 70
	percent.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	if is_predicted:
		percent.add_theme_color_override("font_color", Color(0.2, 0.8, 0.2))
	hbox.add_child(percent)
	
	return hbox
