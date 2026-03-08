import unittest

from muse.cli import build_parser


class CliSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_prog_name_is_muse(self):
        self.assertEqual(self.parser.prog, "muse")

    def test_output_format_rejects_docx(self):
        commands = [
            ["run", "--topic", "Test", "--output-format", "docx"],
            ["resume", "--run-id", "r1", "--output-format", "docx"],
            ["export", "--run-id", "r1", "--output-format", "docx"],
        ]

        for argv in commands:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    self.parser.parse_args(argv)

    def test_template_option_removed(self):
        commands = [
            ["run", "--topic", "Test", "--template", "template.docx"],
            ["resume", "--run-id", "r1", "--template", "template.docx"],
            ["export", "--run-id", "r1", "--template", "template.docx"],
        ]

        for argv in commands:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    self.parser.parse_args(argv)

    def test_output_format_still_accepts_markdown_latex_and_pdf(self):
        commands = [
            ["run", "--topic", "Test", "--output-format", "markdown"],
            ["resume", "--run-id", "r1"],
            ["export", "--run-id", "r1", "--output-format", "pdf"],
        ]

        for argv in commands:
            with self.subTest(argv=argv):
                args = self.parser.parse_args(argv)
                if hasattr(args, "output_format"):
                    self.assertIn(args.output_format, {"markdown", "latex", "pdf"})
                self.assertFalse(hasattr(args, "template"))


if __name__ == "__main__":
    unittest.main()
