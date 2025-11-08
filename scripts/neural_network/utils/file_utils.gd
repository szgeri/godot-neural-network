class_name FileUtils
extends RefCounted

# Provides reusable helper functions for listing files and folders.
# Why: Keeps file I/O logic separate from image processing or evaluation.

# Lists all non-directory file paths in the given directory.
# Params:
#   path (String): Path to directory to scan.
# Returns:
#   PackedStringArray: List of file paths.
static func list_files(path: String) -> PackedStringArray:
	var files: PackedStringArray = []
	var resolved_path: String = resolve_path(path)
	if resolved_path == "":
		push_error("Failed to open directory: %s" % path)
		return files

	var dir: DirAccess = DirAccess.open(resolved_path)
	if dir == null:
		push_error("Failed to open directory: %s" % path)
		return files

	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if file_name == "." or file_name == "..":
			file_name = dir.get_next()
			continue
		if not dir.current_is_dir():
			if file_name.ends_with(".import"):
				file_name = dir.get_next()
				continue
			var base_path := resolved_path
			if base_path.ends_with("/"):
				base_path = base_path.substr(0, base_path.length() - 1)
			files.append(base_path + "/" + file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	return files


# Lists all subdirectory paths within the given directory.
# Params:
#   path (String): Path to directory to scan.
# Returns:
#   PackedStringArray: List of subdirectory paths.
static func list_dirs(path: String) -> PackedStringArray:
	var dirs: PackedStringArray = []
	var resolved_path: String = resolve_path(path)
	if resolved_path == "":
		push_error("Failed to open directory: %s" % path)
		return dirs

	var dir: DirAccess = DirAccess.open(resolved_path)
	if dir == null:
		push_error("Failed to open directory: %s" % path)
		return dirs

	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if file_name == "." or file_name == "..":
			file_name = dir.get_next()
			continue
		if dir.current_is_dir():
			var base_path := resolved_path
			if base_path.ends_with("/"):
				base_path = base_path.substr(0, base_path.length() - 1)
			dirs.append(base_path + "/" + file_name)
		file_name = dir.get_next()
	dir.list_dir_end()
	return dirs

static func resolve_path(path: String) -> String:
	if path == "":
		return path
	if path.begins_with("res://") or path.begins_with("user://"):
		return ProjectSettings.globalize_path(path)
	return path

static func get_home_directory() -> String:
	# Cross‑platform approach
	if OS.get_name() == "Windows":
		return OS.get_environment("USERPROFILE")  # Windows home path
	else:
		return OS.get_environment("HOME")         # Linux / macOS home path
