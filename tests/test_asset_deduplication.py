"""Ensure APK assets are unique and only the required menu artwork remains."""
import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


class AssetDeduplicationTests(unittest.TestCase):
    def test_android_platform_jar_is_not_copied_into_apk_assets(self):
        self.assertTrue((ROOT / "android.jar").is_file())
        self.assertFalse((ASSETS / "android.jar").exists())

    def test_only_distinct_menu_images_remain(self):
        ui_dir = ASSETS / "ui"
        expected = {"blue_neon.png", "green_neon.png"}
        actual = {path.name for path in ui_dir.glob("*.png")}
        self.assertEqual(expected, actual)

    def test_required_menu_artwork_is_still_referenced_and_present(self):
        main = (
            ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"
        ).read_text(encoding="utf-8")
        self.assertIn('"ui/blue_neon.png"', main)
        self.assertIn('"ui/green_neon.png"', main)
        for name in ("blue_neon.png", "green_neon.png"):
            self.assertTrue((ASSETS / "ui" / name).is_file())

    def test_image_files_under_assets_are_not_duplicate_blobs(self):
        files = [
            path for path in ASSETS.rglob("*")
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ]
        seen = {}
        duplicates = []
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                duplicates.append((str(seen[digest].relative_to(ROOT)), str(path.relative_to(ROOT))))
            else:
                seen[digest] = path
        self.assertEqual([], duplicates, "Duplicate image assets waste APK size.")


if __name__ == "__main__":
    unittest.main()
