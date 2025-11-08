extends RefCounted
class_name ModelEvaluator

## Provides model evaluation utilities for classification accuracy.
## WHY: Keeps testing logic modular and separate from training routines.

# -------------------------------------------------------------------
# Multi-class classification evaluation
# -------------------------------------------------------------------

## Evaluates classification accuracy for multi-class models using softmax.
## WHY: Uses argmax comparison against one-hot encoded targets to measure
##      prediction correctness.
##
## Params:
##   network - model under evaluation
##   inputs - batch of input vectors
##   targets - one-hot encoded expected outputs
##   debug - optional flag to print misclassified samples
##
## Returns:
##   Accuracy score between 0.0 and 1.0
static func evaluate_model_soft_max(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array],
	debug: bool = false
) -> float:
	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		predictions_flat,
		targets[0].size()
	)
	var correct: int = 0

	for i: int in range(predictions.size()):
		var pred: PackedFloat32Array = predictions[i]
		var target: PackedFloat32Array = targets[i]
		var idx: int = find_max_value_index(pred)

		if target[idx] == 1:
			correct += 1
		elif debug:
			print_rich(
				"Test case %3d: prediction [color=red]%d[/color] target [color=green]%d[/color]" % [i, idx, find_max_value_index(target)]
			)

	return float(correct) / float(predictions.size())


# -------------------------------------------------------------------
# Confusion matrix and detailed metrics
# -------------------------------------------------------------------

## Computes confusion matrix and detailed classification statistics.
## WHY: Provides granular insight into class-level performance for debugging.
##
## Returns Dictionary with:
##   - "matrix": Array[PackedInt32Array] confusion matrix (rows = actual, cols = predicted)
##   - "per_class_totals": PackedInt32Array with total samples per class
##   - "per_class_correct": PackedInt32Array with correctly classified samples per class
##   - "predicted_indices": PackedInt32Array with model predictions per sample
##   - "target_indices": PackedInt32Array with ground-truth indices per sample
##   - "misclassified": Array[Dictionary] with entries:
##         { "index", "target", "prediction", "confidence", "probabilities" }
static func compute_confusion_details(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> Dictionary:
	var results: Dictionary = {
		"matrix": [],
		"per_class_totals": PackedInt32Array(),
		"per_class_correct": PackedInt32Array(),
		"predicted_indices": PackedInt32Array(),
		"target_indices": PackedInt32Array(),
		"misclassified": []
	}

	if inputs.is_empty() or targets.is_empty():
		return results

	var num_classes: int = targets[0].size()
	if num_classes <= 0:
		return results

	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		predictions_flat,
		num_classes
	)

	var confusion_matrix: Array[PackedInt32Array] = []
	var per_class_totals: PackedInt32Array = PackedInt32Array()
	var per_class_correct: PackedInt32Array = PackedInt32Array()
	var predicted_indices: PackedInt32Array = PackedInt32Array()
	var target_indices: PackedInt32Array = PackedInt32Array()
	var misclassified: Array[Dictionary] = []

	per_class_totals.resize(num_classes)
	per_class_totals.fill(0)
	per_class_correct.resize(num_classes)
	per_class_correct.fill(0)

	predicted_indices.resize(predictions.size())
	target_indices.resize(predictions.size())

	for _i: int in range(num_classes):
		var row: PackedInt32Array = PackedInt32Array()
		row.resize(num_classes)
		row.fill(0)
		confusion_matrix.append(row)

	for sample_idx: int in range(predictions.size()):
		var pred_vec: PackedFloat32Array = predictions[sample_idx]
		var target_vec: PackedFloat32Array = targets[sample_idx]

		var predicted_idx: int = find_max_value_index(pred_vec)
		var target_idx: int = find_max_value_index(target_vec)

		if predicted_idx == -1 or target_idx == -1:
			continue

		predicted_indices[sample_idx] = predicted_idx
		target_indices[sample_idx] = target_idx

		confusion_matrix[target_idx][predicted_idx] += 1
		per_class_totals[target_idx] += 1

		if predicted_idx == target_idx:
			per_class_correct[target_idx] += 1
		else:
			misclassified.append({
				"index": sample_idx,
				"target": target_idx,
				"prediction": predicted_idx,
				"confidence": pred_vec[predicted_idx],
				"probabilities": pred_vec
			})

	results["matrix"] = confusion_matrix
	results["per_class_totals"] = per_class_totals
	results["per_class_correct"] = per_class_correct
	results["predicted_indices"] = predicted_indices
	results["target_indices"] = target_indices
	results["misclassified"] = misclassified

	return results

# -------------------------------------------------------------------
# Binary classification evaluation
# -------------------------------------------------------------------

## Evaluates binary classification accuracy using a fixed threshold of 0.5.
## WHY: Common for sigmoid output, translates continuous predictions to
##      discrete class labels.
##
## Params:
##   network - model under evaluation
##   inputs - batch of input vectors
##   targets - expected binary outputs
##
## Returns:
##   Accuracy score between 0.0 and 1.0
static func evaluate_model(
	network: NeuralNetwork,
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> float:
	var predictions_flat: PackedFloat32Array = network.forward_pass(inputs)
	var predictions: Array[PackedFloat32Array] = TensorUtils.unflatten_batch(
		predictions_flat,
		1
	)
	var correct: int = 0

	for i: int in range(predictions.size()):
		var pred: float = predictions[i][0]
		var target: float = targets[i][0]
		if int(pred >= 0.5) == int(target):
			correct += 1

	return float(correct) / float(predictions.size())

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

## Returns the index of the largest value in an array.
## WHY: Argmax is the standard for selecting predicted class in classification.
##      Returns -1 if input array is empty (avoids undefined index).
static func find_max_value_index(arr: PackedFloat32Array) -> int:
	var max_value: float = -INF
	var max_idx: int = -1
	for i: int in range(arr.size()):
		if arr[i] > max_value:
			max_value = arr[i]
			max_idx = i
	return max_idx
