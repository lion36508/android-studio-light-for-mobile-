import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEXER = ROOT / "src/main/java/com/bikash/mobileide/SourceSyntaxHighlighter.java"
EDITOR = ROOT / "src/main/java/com/bikash/mobileide/EditorActivity.java"


class SyntaxHighlightingContractTests(unittest.TestCase):
    def test_editor_uses_non_overlapping_tokens(self):
        lexer = LEXER.read_text(encoding="utf-8")
        editor = EDITOR.read_text(encoding="utf-8")
        self.assertIn("SourceSyntaxHighlighter.tokenize(s, currentLanguage)", editor)
        self.assertIn("scanQuoted(source, i, c)", lexer)
        self.assertIn("scanLine(source, i + 2)", lexer)
        self.assertIn("scanBlockComment(source, i + 2)", lexer)
        self.assertNotIn('Pattern.compile("//.*|#.*")', editor)

    def test_comment_markers_inside_strings_are_consumed_as_string_tokens(self):
        lexer = LEXER.read_text(encoding="utf-8")
        # Assert against the actual lexer call sites, not a stale copy of the
        # implementation's old if-condition formatting.
        quoted_branch = lexer.index("int end = scanQuoted(source, i, c);")
        line_comment_branch = lexer.index("int end = scanLine(source, i + 2);")
        hash_comment_branch = lexer.index("if (hashComments && c == '#')")
        self.assertLess(quoted_branch, line_comment_branch)
        self.assertLess(quoted_branch, hash_comment_branch)

    def test_hash_comments_are_language_aware(self):
        lexer = LEXER.read_text(encoding="utf-8")
        self.assertIn('"python".equalsIgnoreCase(language)', lexer)
        self.assertIn('"shell".equalsIgnoreCase(language)', lexer)
        self.assertIn("if (hashComments && c == '#')", lexer)


if __name__ == "__main__":
    unittest.main()
