"""Tests for muse/tools/file.py"""

from __future__ import annotations

import os
import tempfile
import unittest


class FileToolTests(unittest.TestCase):
    def test_read_file_existing(self):
        from muse.tools.file import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
            handle.write("line1\nline2\nline3\n")
            path = handle.name
        try:
            result = read_file.invoke({"file_path": path})
            self.assertIn("line1", result)
            self.assertIn("line2", result)
        finally:
            os.unlink(path)

    def test_read_file_nonexistent(self):
        from muse.tools.file import read_file

        result = read_file.invoke({"file_path": "/tmp/_nonexistent_file_xyz.txt"})
        self.assertIn("error", result.lower())

    def test_read_file_with_offset_limit(self):
        from muse.tools.file import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
            for index in range(20):
                handle.write(f"line {index}\n")
            path = handle.name
        try:
            result = read_file.invoke({"file_path": path, "offset": 5, "limit": 3})
            self.assertIn("line 5", result)
            self.assertNotIn("line 0", result)
        finally:
            os.unlink(path)

    def test_write_file_creates_and_writes(self):
        from muse.tools.file import write_file

        path = os.path.join(tempfile.gettempdir(), "_muse_test_write.txt")
        try:
            result = write_file.invoke({"file_path": path, "content": "hello world"})
            self.assertIn("ok", result.lower())
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "hello world")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_edit_file_replaces_string(self):
        from muse.tools.file import edit_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
            handle.write("The quick brown fox.")
            path = handle.name
        try:
            result = edit_file.invoke(
                {
                    "file_path": path,
                    "old_string": "quick brown",
                    "new_string": "slow red",
                }
            )
            self.assertIn("ok", result.lower())
            with open(path, encoding="utf-8") as handle:
                self.assertIn("slow red", handle.read())
        finally:
            os.unlink(path)

    def test_edit_file_old_string_not_found(self):
        from muse.tools.file import edit_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
            handle.write("Hello world.")
            path = handle.name
        try:
            result = edit_file.invoke(
                {
                    "file_path": path,
                    "old_string": "xyz",
                    "new_string": "abc",
                }
            )
            self.assertIn("not found", result.lower())
        finally:
            os.unlink(path)

    def test_glob_finds_files(self):
        from muse.tools.file import glob_files

        result = glob_files.invoke(
            {"pattern": "*.py", "directory": os.path.dirname(__file__)}
        )
        self.assertIsInstance(result, str)

    def test_grep_searches_content(self):
        from muse.tools.file import grep

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
            handle.write("def hello_world():\n    pass\n")
            path = handle.name
        try:
            result = grep.invoke(
                {"pattern": "hello_world", "path": os.path.dirname(path)}
            )
            self.assertIn("hello_world", result)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
