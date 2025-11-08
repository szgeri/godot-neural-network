extends RefCounted
class_name TensorUtils

## Utility methods for tensor operations and data conversions.
## WHY: Centralizes low-level data reshaping, encoding/decoding, and validation
##      to keep training/evaluation code lean and maintainable.

# -------------------------------------------------------------------
# Batch flattening / unflattening
# -------------------------------------------------------------------

## Flattens a batch of vectors into a single packed float array.
## WHY: Shaders and GPU calls often require contiguous memory layouts.
static func flatten_batch(batch: Array[PackedFloat32Array]) -> PackedFloat32Array:
	var flat: PackedFloat32Array = PackedFloat32Array()
	for vector: PackedFloat32Array in batch:
		flat.append_array(vector)
	return flat

## Reshapes a flat array back into a batch of vectors.
## WHY: Restores logical per-sample grouping after GPU computation.
static func unflatten_batch(
	flat: PackedFloat32Array,
	vector_size: int
) -> Array[PackedFloat32Array]:
	var batch: Array[PackedFloat32Array] = []

	if vector_size <= 0:
		push_error("unflatten_batch: vector_size must be positive")
		return batch

	if flat.size() % vector_size != 0:
		push_error(
			"unflatten_batch: flat size %d is not divisible by vector_size %d"
			% [flat.size(), vector_size]
		)
		return batch

	for start_index: int in range(0, flat.size(), vector_size):
		var vector: PackedFloat32Array = PackedFloat32Array()
		for j: int in range(vector_size):
			vector.append(flat[start_index + j])
		batch.append(vector)

	return batch

# -------------------------------------------------------------------
# Bytes ↔ Floats / Ints conversions
# -------------------------------------------------------------------

## Decodes bytes to float32 array.
## WHY: Reads binary shader outputs back into GDScript-friendly floats.
static func bytes_to_floats(bytes: PackedByteArray) -> PackedFloat32Array:
	var floats: PackedFloat32Array = PackedFloat32Array()
	if bytes.size() % 4 != 0:
		push_error("bytes_to_floats: byte array size %d not divisible by 4" % bytes.size())
		return floats
	for i: int in range(0, bytes.size(), 4):
		floats.append(bytes.decode_float(i))
	return floats

## Decodes bytes to int32 array.
## WHY: For integer-coded tensor data from shaders.
static func bytes_to_ints(bytes: PackedByteArray) -> PackedInt32Array:
	var ints: PackedInt32Array = PackedInt32Array()
	if bytes.size() % 4 != 0:
		push_error("bytes_to_ints: byte array size %d not divisible by 4" % bytes.size())
		return ints
	for i: int in range(0, bytes.size(), 4):
		ints.append(bytes.decode_s32(i))
	return ints

## Decodes bytes holding uint bit patterns into floats.
## WHY: Needed when GPU atomics store float bits via uint encoding.
static func uint_bytes_to_floats(bytes: PackedByteArray) -> PackedFloat32Array:
	var floats: PackedFloat32Array = PackedFloat32Array()
	if bytes.size() % 4 != 0:
		push_error("uint_bytes_to_floats: byte array size %d not divisible by 4" % bytes.size())
		return floats
	for i: int in range(0, bytes.size(), 4):
		var bits: int = bytes.decode_u32(i)
		floats.append(float_from_bits(bits))

	return floats

## Interprets 32-bit integer bits as float value.
static func float_from_bits(bits: int) -> float:
	var ba: PackedByteArray = PackedByteArray()
	ba.resize(4)
	ba.encode_u32(0, bits)
	return ba.decode_float(0)

## Encodes float array into byte array.
static func floats_to_bytes(floats: PackedFloat32Array) -> PackedByteArray:
	return floats.to_byte_array()

# -------------------------------------------------------------------
# Weight / gradient reshaping
# -------------------------------------------------------------------

## Reshapes a flat array into a 2D weight matrix [output][input].
## WHY: Aligns gradients/weights to layer structure for CPU-side inspection.
static func reshape_weights(
	flat: PackedFloat32Array,
	input_size: int,
	output_size: int
) -> Array[Array]:
	var expected_size: int = input_size * output_size
	if flat.size() < expected_size:
		push_error("reshape_weights: Expected %d elements, got %d" % [expected_size, flat.size()])
		return []

	var matrix: Array[Array] = []
	for i: int in range(output_size):
		var row: Array[float] = []
		for j: int in range(input_size):
			row.append(flat[i * input_size + j])
		matrix.append(row)

	return matrix

# -------------------------------------------------------------------
# Dataset batching / shuffling
# -------------------------------------------------------------------

## Splits inputs and targets into mini-batches.
## WHY: Supports batch-based training without manual slicing.
static func create_batches(
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array],
	batch_size: int
) -> Array[Dictionary]:
	var batches: Array[Dictionary] = []
	var total_samples: int = inputs.size()

	for i: int in range(0, total_samples, batch_size):
		var end_idx: int = i + min(batch_size, total_samples - i)
		var input_batch: Array[PackedFloat32Array] = inputs.slice(i, end_idx)
		var target_batch: Array[PackedFloat32Array] = targets.slice(i, end_idx)
		batches.append({
			"inputs": input_batch,
			"targets": target_batch
		})

	return batches

## Shuffles inputs and targets in-place with shared permutation.
## WHY: Maintains input/target pairing while randomizing sample order.
static func shuffle_data(
	inputs: Array[PackedFloat32Array],
	targets: Array[PackedFloat32Array]
) -> void:
	var indices: Array[int] = []
	for i: int in range(inputs.size()):
		indices.append(i)
	indices.shuffle()

	var shuffled_inputs: Array[PackedFloat32Array] = []
	var shuffled_targets: Array[PackedFloat32Array] = []
	for i: int in indices:
		shuffled_inputs.append(inputs[i])
		shuffled_targets.append(targets[i])

	inputs.clear()
	targets.clear()
	inputs.append_array(shuffled_inputs)
	targets.append_array(shuffled_targets)

# -------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------

## Checks if value is NaN or above specified magnitude threshold.
## WHY: Detects numerical instability or exploding parameters.
static func is_nan_or_exploding(value: float, exploding_threshold: float) -> bool:
	return is_nan(value) or abs(value) > exploding_threshold
