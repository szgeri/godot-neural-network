extends Node

## Global Configuration
## This autoload singleton stores project-wide configuration settings.
## Access it from any script using: GlobalConfig.full_emnist_digits

# -------------------------------------------------------------------
# Dataset Configuration
# -------------------------------------------------------------------
@export_category("Dataset Selection")
@export var full_emnist_digits: bool = true  ## true = EMNIST dataset, false = MNIST dataset

# -------------------------------------------------------------------
# Path Helpers
# -------------------------------------------------------------------

## Get the appropriate model path based on dataset selection
func get_model_path() -> String:
	if full_emnist_digits:
		# return "res://assets/models/emnist_digit_classifier.tres"
		# return "res://assets/models/not_mnist_digit_classifier.tres"
		return "res://assets/models/hw1_classifier.tres"
	else:
		return "res://assets/models/mnist_digit_classifier.tres"

## Get the appropriate training data directory based on dataset selection
func get_training_data_dir() -> String:
	if full_emnist_digits:
		# return "/Users/szgeri/emnist_digits_flipped_vertically_png/training"
		# return "/Users/szgeri/not_mnist_png_inverted"
		return "/Users/szgeri/HWD-V1-processed/training"
	else:
		return "/Users/szgeri/mnist_png/training"

## Get the appropriate test data directory based on dataset selection
func get_test_data_dir() -> String:
	if full_emnist_digits:
		# return "/Users/szgeri/emnist_digits_flipped_vertically_png/testing"
		# return "/Users/szgeri/mnist_png/testing"
		return "/Users/szgeri/HWD-V1-processed/testing"
	else:
		return "/Users/szgeri/mnist_png/testing"

## Get a friendly name for the current dataset
func get_dataset_name() -> String:
	return "EMNIST" if full_emnist_digits else "MNIST"
