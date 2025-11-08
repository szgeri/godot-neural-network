extends Control
class_name DigitEvaluationDashboard

@export_file("*.tres") var model_path: String = "res://assets/models/mnist_digit_classifier.tres"
@export_global_dir var training_data_dir: String = "/Users/szgeri/mnist_png/training"
@export_global_dir var test_data_dir: String = "/Users/szgeri/mnist_png/testing"
@export_range(0.0, 1.0) var image_scale: float = 1.0
@export var limit_per_class: int = 0
@export var show_training_metrics: bool = true
@export var max_misclassifications_to_display: int = 8
@export var invert_images: bool = true

@onready var status_label: RichTextLabel = $MarginContainer/VBox/StatusLabel
@onready var accuracy_label: Label = $MarginContainer/VBox/SummaryPanel/SummaryMargin/SummaryVBox/AccuracyRow/AccuracyValue
@onready var dataset_summary_label: RichTextLabel = $MarginContainer/VBox/SummaryPanel/SummaryMargin/SummaryVBox/DatasetSummary
@onready var per_class_label: RichTextLabel = $MarginContainer/VBox/PerClassPanel/PerClassMargin/PerClassVBox/PerClassMetrics
@onready var confusion_matrix_label: RichTextLabel = $MarginContainer/VBox/ConfusionPanel/ConfusionMargin/ConfusionVBox/ConfusionMatrix
@onready var error_summary_label: RichTextLabel = $MarginContainer/VBox/ErrorPanel/ErrorMargin/ErrorVBox/ErrorSummary
@onready var refresh_button: Button = $MarginContainer/VBox/RefreshButton


func _ready() -> void:
	refresh_button.pressed.connect(_on_refresh_pressed)
	_run_evaluation()


func _on_refresh_pressed() -> void:
	_run_evaluation()


func _run_evaluation() -> void:
	status_label.text = "[color=yellow]Evaluating model...[/color]"
	_clear_outputs()

	var test_dataset: Dictionary = _load_dataset(test_data_dir)
	if test_dataset.is_empty():
		status_label.text = "[color=red]No test dataset found at %s[/color]" % test_data_dir
		return

	var forward_runner: ForwardPassRunner = ForwardPassRunner.new()
	var network: NeuralNetwork = NeuralNetworkSerializer.import(forward_runner, model_path)
	if network == null:
		status_label.text = "[color=red]Failed to load model: %s[/color]" % model_path
		return

	var test_accuracy: float = ModelEvaluator.evaluate_model_soft_max(
		network,
		test_dataset["inputs"],
		test_dataset["targets"]
	)
	var test_details: Dictionary = ModelEvaluator.compute_confusion_details(
		network,
		test_dataset["inputs"],
		test_dataset["targets"]
	)

	var training_summary: Dictionary = {}
	if show_training_metrics and training_data_dir != "":
		var train_dataset: Dictionary = _load_dataset(training_data_dir)
		if not train_dataset.is_empty():
			var train_accuracy: float = ModelEvaluator.evaluate_model_soft_max(
				network,
				train_dataset["inputs"],
				train_dataset["targets"]
			)
			training_summary = {
				"samples": train_dataset["inputs"].size(),
				"accuracy": train_accuracy
			}

	_update_dashboard(test_dataset, test_accuracy, test_details, training_summary)
	status_label.text = "[color=green]Evaluation complete[/color]"


func _update_dashboard(
	test_dataset: Dictionary,
	test_accuracy: float,
	test_details: Dictionary,
	training_summary: Dictionary
) -> void:
	accuracy_label.text = "%5.2f%%" % (test_accuracy * 100.0)

	dataset_summary_label.text = _format_dataset_summary(
		test_dataset,
		test_accuracy,
		training_summary
	)

	per_class_label.text = _format_per_class_metrics(
		test_details.get("matrix", []),
		test_dataset.get("class_labels", [])
	)

	confusion_matrix_label.text = _format_confusion_matrix(
		test_details.get("matrix", []),
		test_dataset.get("class_labels", [])
	)

	error_summary_label.text = _format_error_summary(
		test_details.get("misclassified", []),
		test_dataset.get("file_paths", []),
		test_dataset.get("class_labels", [])
	)


func _clear_outputs() -> void:
	accuracy_label.text = "—"
	dataset_summary_label.text = ""
	per_class_label.text = ""
	confusion_matrix_label.text = ""
	error_summary_label.text = ""


func _load_dataset(root_path: String) -> Dictionary:
	if root_path == "":
		return {}

	var directories: PackedStringArray = FileUtils.list_dirs(root_path)
	directories.sort()
	if directories.is_empty():
		return {}

	var class_labels: Array[String] = []
	var label_to_index: Dictionary = {}
	for dir_path: String in directories:
		var label_name: String = dir_path.get_file()
		if not label_to_index.has(label_name):
			label_to_index[label_name] = class_labels.size()
			class_labels.append(label_name)

	var class_count: int = class_labels.size()
	if class_count == 0:
		return {}

	var class_counts: PackedInt32Array = PackedInt32Array()
	class_counts.resize(class_count)
	class_counts.fill(0)

	var inputs: Array[PackedFloat32Array] = []
	var targets: Array[PackedFloat32Array] = []
	var sample_labels: Array[int] = []
	var file_paths: Array[String] = []

	for dir_path: String in directories:
		var label_name: String = dir_path.get_file()
		var class_idx: int = int(label_to_index.get(label_name, -1))
		if class_idx < 0:
			continue

		var files: PackedStringArray = FileUtils.list_files(dir_path)
		files.sort()

		var max_files: int = files.size()
		if limit_per_class > 0:
			max_files = mini(limit_per_class, files.size())

		for file_idx: int in range(max_files):
			var file_path: String = files[file_idx]
			var image_data: PackedFloat32Array = ImageUtils.read_image(
				file_path,
				image_scale,
				invert_images
			)
			if image_data.is_empty():
				continue

			inputs.append(image_data)
			file_paths.append(file_path)
			sample_labels.append(class_idx)
			class_counts[class_idx] += 1

	if inputs.is_empty():
		return {}

	for label_idx: int in sample_labels:
		targets.append(_create_one_hot(label_idx, class_count))

	return {
		"inputs": inputs,
		"targets": targets,
		"class_labels": class_labels,
		"labels": sample_labels,
		"file_paths": file_paths,
		"class_counts": class_counts
	}


func _create_one_hot(index: int, length: int) -> PackedFloat32Array:
	var target: PackedFloat32Array = PackedFloat32Array()
	target.resize(length)
	target.fill(0.0)
	if index >= 0 and index < length:
		target[index] = 1.0
	return target


func _format_dataset_summary(
	dataset: Dictionary,
	test_accuracy: float,
	training_summary: Dictionary
) -> String:
	if dataset.is_empty():
		return "[color=red]Dataset failed to load.[/color]"

	var inputs: Array = dataset.get("inputs", [])
	var class_labels: Array[String] = dataset.get("class_labels", [])
	var class_counts: PackedInt32Array = dataset.get("class_counts", PackedInt32Array())

	var total_samples: int = inputs.size()
	var feature_size: int = 0
	if total_samples > 0:
		var first_sample: PackedFloat32Array = inputs[0]
		feature_size = first_sample.size()

	var summary: String = ""
	summary += "[b]Model:[/b] %s\n" % model_path
	summary += "[b]Test dataset:[/b] %s\n" % test_data_dir
	summary += "[b]Samples:[/b] %d across %d classes\n" % [
		total_samples,
		class_labels.size()
	]
	if feature_size > 0:
		summary += "[b]Feature size:[/b] %d\n" % feature_size
	summary += "[b]Accuracy:[/b] %.2f%%\n" % (test_accuracy * 100.0)
	if limit_per_class > 0:
		summary += "[b]Per-class limit:[/b] %d\n" % limit_per_class
	if not class_labels.is_empty():
		summary += "[b]Class counts:[/b] %s\n" % _format_class_counts(
			class_labels,
			class_counts
		)
	if not training_summary.is_empty():
		var train_samples: int = int(training_summary.get("samples", 0))
		var train_accuracy: float = float(training_summary.get("accuracy", 0.0))
		summary += "[b]Training accuracy:[/b] %.2f%% (%d samples)\n" % [
			train_accuracy * 100.0,
			train_samples
		]

	return summary.strip_edges()


func _format_class_counts(
	class_labels: Array[String],
	class_counts: PackedInt32Array
) -> String:
	var parts: Array[String] = []
	for i: int in range(class_labels.size()):
		var count: int = 0
		if i < class_counts.size():
			count = class_counts[i]
		parts.append("%s: %d" % [class_labels[i], count])
	return ", ".join(parts)


func _format_per_class_metrics(
	matrix: Array,
	class_labels: Array[String]
) -> String:
	if matrix.is_empty():
		return "[color=gray]No confusion matrix available.[/color]"

	var num_classes: int = matrix.size()
	var table_lines: Array[String] = []
	table_lines.append(
		_pad_right("Class", 8)
		+ _pad_left("Prec%", 8)
		+ _pad_left("Rec%", 8)
		+ _pad_left("F1%", 8)
		+ _pad_left("Support", 9)
	)

	for i: int in range(num_classes):
		var row: PackedInt32Array = matrix[i]
		var true_positive: int = 0
		if i < row.size():
			true_positive = row[i]

		var support: int = 0
		for value: int in row:
			support += value

		var false_positive: int = 0
		for r_idx: int in range(num_classes):
			var other_row: PackedInt32Array = matrix[r_idx]
			if r_idx == i:
				continue
			if i < other_row.size():
				false_positive += other_row[i]
		var precision: float = 0.0
		var precision_denom: int = true_positive + false_positive
		if precision_denom > 0:
			precision = float(true_positive) / float(precision_denom)

		var recall: float = 0.0
		if support > 0:
			recall = float(true_positive) / float(support)

		var f1: float = 0.0
		if precision + recall > 0.0:
			f1 = 2.0 * precision * recall / (precision + recall)

		var label: String = _label_for_index(i, class_labels)
		table_lines.append(
			_pad_right(label, 8)
			+ _pad_left("%.2f" % (precision * 100.0), 8)
			+ _pad_left("%.2f" % (recall * 100.0), 8)
			+ _pad_left("%.2f" % (f1 * 100.0), 8)
			+ _pad_left(str(support), 9)
		)

	var table_text: String = ""
	for line: String in table_lines:
		table_text += line + "\n"

	return "[b]Per-class Metrics[/b]\n[code]%s[/code]" % table_text.strip_edges()


func _format_confusion_matrix(
	matrix: Array,
	class_labels: Array[String]
) -> String:
	if matrix.is_empty():
		return "[color=gray]No predictions available to build confusion matrix.[/color]"

	var num_classes: int = matrix.size()
	var max_value: int = 0
	for row: PackedInt32Array in matrix:
		for value: int in row:
			if value > max_value:
				max_value = value

	var value_width: int = max(4, str(max_value).length())
	var label_width: int = 5
	for label: String in class_labels:
		label_width = max(label_width, label.length())

	var header: String = _pad_right("Actual\\Pred", label_width)
	for label: String in class_labels:
		header += _pad_left(label, value_width + 1)

	var lines: Array[String] = [header]
	for i: int in range(num_classes):
		var row_label: String = _label_for_index(i, class_labels)
		var row_text: String = _pad_right(row_label, label_width)
		var row: PackedInt32Array = matrix[i]
		for j: int in range(num_classes):
			var cell_value: int = 0
			if j < row.size():
				cell_value = row[j]
			row_text += _pad_left(str(cell_value), value_width + 1)
		lines.append(row_text)

	var matrix_text: String = ""
	for line: String in lines:
		matrix_text += line + "\n"

	return "[b]Confusion Matrix[/b]\n[code]%s[/code]" % matrix_text.strip_edges()


func _format_error_summary(
	misclassified: Array,
	file_paths: Array[String],
	class_labels: Array[String]
) -> String:
	var total_errors: int = misclassified.size()
	if total_errors == 0:
		return "[color=green]No misclassifications detected on this dataset.[/color]"

	var summary: String = "[b]Total misclassifications:[/b] %d\n" % total_errors

	var pair_counts: Dictionary = {}
	for entry: Dictionary in misclassified:
		var target_idx: int = int(entry.get("target", -1))
		var predicted_idx: int = int(entry.get("prediction", -1))
		if target_idx < 0 or predicted_idx < 0:
			continue
		var key: String = "%d.%d" % [target_idx, predicted_idx]
		pair_counts[key] = pair_counts.get(key, 0) + 1

	if pair_counts.is_empty():
		summary += "\n[b]Error patterns:[/b] None\n"
	else:
		var pair_list: Array[Dictionary] = []
		for key: String in pair_counts.keys():
			var parts: PackedStringArray = key.split(".")
			if parts.size() != 2:
				continue
			pair_list.append({
				"target": int(parts[0]),
				"prediction": int(parts[1]),
				"count": int(pair_counts[key])
			})
		pair_list.sort_custom(Callable(self, "_pair_count_desc"))

		summary += "\n[b]Top error patterns[/b]\n"
		var max_pairs: int = min(5, pair_list.size())
		for i: int in range(max_pairs):
			var pair_entry: Dictionary = pair_list[i]
			var actual_label: String = _label_for_index(int(pair_entry.get("target", -1)), class_labels)
			var predicted_label: String = _label_for_index(int(pair_entry.get("prediction", -1)), class_labels)
			var count: int = int(pair_entry.get("count", 0))
			summary += "- %s → %s : %d\n" % [actual_label, predicted_label, count]

	var sorted_misclassified: Array[Dictionary] = misclassified.duplicate()
	sorted_misclassified.sort_custom(Callable(self, "_confidence_desc"))

	summary += "\n[b]Most confident mistakes[/b]\n"
	var limit: int = min(max_misclassifications_to_display, sorted_misclassified.size())
	if limit == 0:
		summary += "None.\n"
	else:
		for i: int in range(limit):
			var entry: Dictionary = sorted_misclassified[i]
			var sample_index: int = int(entry.get("index", -1))
			var actual_label: String = _label_for_index(int(entry.get("target", -1)), class_labels)
			var predicted_label: String = _label_for_index(int(entry.get("prediction", -1)), class_labels)
			var confidence: float = float(entry.get("confidence", 0.0))
			var sample_path: String = ""
			if sample_index >= 0 and sample_index < file_paths.size():
				sample_path = file_paths[sample_index]

			summary += "- %s → %s (conf %.2f%%) %s\n" % [
				actual_label,
				predicted_label,
				confidence * 100.0,
				sample_path
			]

	return summary.strip_edges()


func _pair_count_desc(a: Dictionary, b: Dictionary) -> bool:
	var count_a: int = int(a.get("count", 0))
	var count_b: int = int(b.get("count", 0))
	if count_a == count_b:
		return int(a.get("target", 0)) < int(b.get("target", 0))
	return count_a > count_b


func _confidence_desc(a: Dictionary, b: Dictionary) -> bool:
	var conf_a: float = float(a.get("confidence", 0.0))
	var conf_b: float = float(b.get("confidence", 0.0))
	if is_equal_approx(conf_a, conf_b):
		return int(a.get("index", 0)) < int(b.get("index", 0))
	return conf_a > conf_b


func _label_for_index(index: int, class_labels: Array[String]) -> String:
	if index >= 0 and index < class_labels.size():
		return class_labels[index]
	return str(index)


func _pad_left(value: String, width: int) -> String:
	if value.length() >= width:
		return value
	return " ".repeat(width - value.length()) + value


func _pad_right(value: String, width: int) -> String:
	if value.length() >= width:
		return value
	return value + " ".repeat(width - value.length())
