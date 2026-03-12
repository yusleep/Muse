"""File manipulation tools for ReAct agents."""

from __future__ import annotations

import fnmatch
import os
import re

from langchain_core.tools import tool


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file with optional line offset and limit."""

    if not os.path.isfile(file_path):
        return f"[read_file] Error: file not found: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except Exception as exc:  # noqa: BLE001
        return f"[read_file] Error reading {file_path}: {exc}"

    selected = lines[offset : offset + limit]
    numbered = []
    for index, line in enumerate(selected, start=offset + 1):
        numbered.append(f"{index:>6}\t{line.rstrip()[:2000]}")
    return "\n".join(numbered)


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""

    try:
        parent = os.path.dirname(file_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return f"OK: wrote {len(content)} bytes to {file_path}"
    except Exception as exc:  # noqa: BLE001
        return f"[write_file] Error writing {file_path}: {exc}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of an exact string inside a file."""

    if not os.path.isfile(file_path):
        return f"[edit_file] Error: file not found: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except Exception as exc:  # noqa: BLE001
        return f"[edit_file] Error reading {file_path}: {exc}"

    if old_string not in content:
        return f"[edit_file] Error: old_string not found in {file_path}"

    new_content = content.replace(old_string, new_string, 1)
    try:
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(new_content)
        return f"OK: replaced in {file_path}"
    except Exception as exc:  # noqa: BLE001
        return f"[edit_file] Error writing {file_path}: {exc}"


@tool
def glob_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a glob pattern under a directory."""

    matches: list[str] = []
    try:
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    matches.append(os.path.join(root, filename))
            if len(matches) > 200:
                break
    except Exception as exc:  # noqa: BLE001
        return f"[glob] Error: {exc}"

    if not matches:
        return f"[glob] No files matching '{pattern}' in {directory}"
    return "\n".join(sorted(matches)[:200])


@tool
def grep(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Search file contents for a regex pattern."""

    results: list[str] = []
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"[grep] Invalid regex: {exc}"

    def search_file(file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                for line_num, line in enumerate(handle, 1):
                    if compiled.search(line):
                        results.append(f"{file_path}:{line_num}: {line.rstrip()[:200]}")
                    if len(results) > 100:
                        return
        except Exception:
            return

    if os.path.isfile(path):
        search_file(path)
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for filename in files:
                if fnmatch.fnmatch(filename, file_glob):
                    search_file(os.path.join(root, filename))
                if len(results) > 100:
                    break
            if len(results) > 100:
                break
    else:
        return f"[grep] Error: path not found: {path}"

    if not results:
        return f"[grep] No matches for '{pattern}' in {path}"
    return "\n".join(results[:100])
