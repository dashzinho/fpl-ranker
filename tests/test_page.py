"""The page is one big hand-written script inside app/board_template.html. A typo there only shows up in a browser,
so syntax-check it here (needs Node; skipped when Node is not installed)."""
import os, re, shutil, subprocess, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PageScript(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_template_script_parses(self):
        html = open(os.path.join(ROOT, "app", "board_template.html"), encoding="utf-8").read()
        scripts = [s for s in re.findall(r"<script[^>]*>(.*?)</script>", html, re.S) if "function viewModel" in s]
        self.assertEqual(len(scripts), 1, "expected exactly one page script")
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "page.js")
            open(f, "w", encoding="utf-8").write(scripts[0])
            r = subprocess.run(["node", "--check", f], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[:500])

    def test_template_has_the_data_placeholder_once(self):
        html = open(os.path.join(ROOT, "app", "board_template.html"), encoding="utf-8").read()
        self.assertEqual(html.count("__DATA__"), 1)


if __name__ == "__main__":
    unittest.main()
