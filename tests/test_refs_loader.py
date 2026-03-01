"""Tests for refs_loader — local reference file ingestion."""

import os
import tempfile
import unittest

from thesis_agent.refs_loader import (
    _extract_year_from_stem,
    _local_ref_id,
    _stem_to_title,
    load_local_refs,
)


class TestLocalRefId(unittest.TestCase):
    def test_simple_stem(self):
        self.assertEqual(_local_ref_id("smith2021security"), "@local_smith2021security")

    def test_spaces_become_underscores(self):
        ref_id = _local_ref_id("My Paper 2022")
        self.assertTrue(ref_id.startswith("@local_"))
        self.assertNotIn(" ", ref_id)

    def test_special_chars_removed(self):
        ref_id = _local_ref_id("paper!@#$%2023")
        # Only alphanumeric and underscores allowed inside
        inner = ref_id[len("@local_"):]
        self.assertTrue(all(c.isalnum() or c == "_" for c in inner))


class TestStemToTitle(unittest.TestCase):
    def test_underscores_to_spaces(self):
        self.assertEqual(_stem_to_title("smith_2021_security"), "Smith 2021 Security")

    def test_hyphens_to_spaces(self):
        self.assertEqual(_stem_to_title("my-paper-2022"), "My Paper 2022")


class TestExtractYear(unittest.TestCase):
    def test_year_in_middle(self):
        self.assertEqual(_extract_year_from_stem("smith_2021_security"), 2021)

    def test_no_year(self):
        self.assertIsNone(_extract_year_from_stem("my_paper"))

    def test_multiple_years_uses_last(self):
        self.assertEqual(_extract_year_from_stem("1998_review_2020"), 2020)

    def test_out_of_range_year_ignored(self):
        # 1800 and 2100 are not matched by the pattern (19xx or 20xx only)
        self.assertIsNone(_extract_year_from_stem("paper_1800"))
        self.assertIsNone(_extract_year_from_stem("paper_2100"))


class TestLoadLocalRefs(unittest.TestCase):
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs = load_local_refs(tmp)
            self.assertEqual(refs, [])

    def test_nonexistent_directory(self):
        refs = load_local_refs("/nonexistent_dir_xyz")
        self.assertEqual(refs, [])

    def test_md_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "smith_2021_security.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# Title\n\nSome content here.")

            refs = load_local_refs(tmp)
            self.assertEqual(len(refs), 1)
            ref = refs[0]
            self.assertEqual(ref["source"], "local")
            self.assertIn("@local_", ref["ref_id"])
            self.assertEqual(ref["year"], 2021)
            self.assertFalse(ref["verified_metadata"])
            self.assertIn("Some content", ref["full_text"])
            self.assertIsNone(ref["doi"])
            self.assertEqual(ref["authors"], [])

    def test_txt_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt_path = os.path.join(tmp, "note.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Just a plain text note.")

            refs = load_local_refs(tmp)
            self.assertEqual(len(refs), 1)
            self.assertIn("plain text", refs[0]["full_text"])

    def test_index_subdirectory_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create .index/ subdirectory (should be skipped)
            index_dir = os.path.join(tmp, ".index")
            os.makedirs(index_dir)
            idx_file = os.path.join(index_dir, "chunks.json")
            with open(idx_file, "w") as f:
                f.write("[]")
            # Create a valid md file
            with open(os.path.join(tmp, "paper.md"), "w") as f:
                f.write("Real paper content.")

            refs = load_local_refs(tmp)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0]["source"], "local")

    def test_unsupported_extension_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "data.csv"), "w") as f:
                f.write("a,b,c")
            refs = load_local_refs(tmp)
            self.assertEqual(refs, [])

    def test_abstract_is_first_1000_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = "A" * 2000
            with open(os.path.join(tmp, "longpaper.txt"), "w") as f:
                f.write(content)
            refs = load_local_refs(tmp)
            self.assertEqual(len(refs[0]["abstract"]), 1000)

    def test_empty_file_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "empty.md"), "w") as f:
                f.write("   \n  ")
            refs = load_local_refs(tmp)
            self.assertEqual(refs, [])
