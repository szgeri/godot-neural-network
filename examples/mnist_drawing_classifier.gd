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
	
	# Crop to bounding box with padding
	var padding: int = 20
	var crop_start_x: int = maxi(0, bounds.position.x - padding)
	var crop_start_y: int = maxi(0, bounds.position.y - padding)
	var crop_end_x: int = mini(canvas_size.x - 1, bounds.position.x + bounds.size.x + padding - 1)
	var crop_end_y: int = mini(canvas_size.y - 1, bounds.position.y + bounds.size.y + padding - 1)
	
	var crop_width: int = crop_end_x - crop_start_x + 1
	var crop_height: int = crop_end_y - crop_start_y + 1
	var crop_rect: Rect2i = Rect2i(crop_start_x, crop_start_y, crop_width, crop_height)
	
	print("=== Image Processing Debug ===")
	print("Bounding box: ", bounds)
	print("Crop rect: ", crop_rect)
	
	var cropped: Image = _crop_image(drawn_image, crop_rect)
	print("Cropped size: %d x %d" % [cropped.get_width(), cropped.get_height()])
	
	# Make it square by adding padding to shorter dimension
	var square_image: Image = _make_square_with_padding(cropped)
	print("Square size: %d x %d" % [square_image.get_width(), square_image.get_height()])
	
	# Resize to model input size
	square_image.resize(
		MODEL_IMAGE_SIZE,
		MODEL_IMAGE_SIZE,
		Image.INTERPOLATE_LANCZOS
	)
	print("Final size (canvas space): %d x %d" % [
		square_image.get_width(),
		square_image.get_height()
	])
	
	# Prepare image for the neural network (invert so digits are light on dark)
	var model_image: Image = _prepare_image_for_model(square_image)
	
	# Show preview (what the network sees)
	var preview: Image = model_image.duplicate()
	preview.resize(128, 128, Image.INTERPOLATE_NEAREST)
	preview_texture.texture = ImageTexture.create_from_image(preview)
	
	# Convert to neural network input format
	var input_data: PackedFloat32Array = _image_to_input_array(model_image)
	print("Input data size: %d" % input_data.size())
	print("Sample values: [%.3f, %.3f, %.3f, %.3f, %.3f]" % [
		input_data[0], input_data[1], input_data[2], input_data[3], input_data[4]
	])
	print("=======================\n")
	
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

func _make_square_with_padding(img: Image) -> Image:
	var width: int = img.get_width()
	var height: int = img.get_height()
	var square_size: int = maxi(width, height)
	
	var square: Image = Image.create(square_size, square_size, false, Image.FORMAT_RGBA8)
	square.fill(Color(1, 1, 1, 1))  # White padding to match canvas background
	
	var offset_x: int = (square_size - width) / 2
	var offset_y: int = (square_size - height) / 2
	
	for y in range(height):
		for x in range(width):
			square.set_pixel(offset_x + x, offset_y + y, img.get_pixel(x, y))
	
	return square

func _prepare_image_for_model(img: Image) -> Image:
	var prepared: Image = Image.create(
		img.get_width(),
		img.get_height(),
		false,
		Image.FORMAT_RGBA8
	)

	for y: int in range(img.get_height()):
		for x: int in range(img.get_width()):
			var color: Color = img.get_pixel(x, y)
			var grayscale: float = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
			var inverted: float = 1.0 - grayscale
			prepared.set_pixel(x, y, Color(inverted, inverted, inverted, 1.0))

	return prepared

func _image_to_input_array(img: Image) -> PackedFloat32Array:
	var result: PackedFloat32Array = PackedFloat32Array()
	result.resize(MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE)
	
	for y in range(MODEL_IMAGE_SIZE):
		for x in range(MODEL_IMAGE_SIZE):
			var color: Color = img.get_pixel(x, y)
			# Convert to grayscale [0, 1]
			var grayscale: float = color.r * 0.299 + color.g * 0.587 + color.b * 0.114
			# Normalize to [-1, 1] range expected by the network
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
