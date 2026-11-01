import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"


class ProjectCatalogPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_catalog_is_loaded_before_dashboard_is_built(self):
        self.assertIn("if (!loadProjectCatalog())", self.source)
        self.assertIn('prefs.getString("project_catalog", null)', self.source)

    def test_catalog_round_trips_all_project_fields(self):
        for field in ("name", "subtitle", "date", "icon", "badgeColor",
                      "lang", "fileName", "starterCode"):
            self.assertIn('saved.put("' + field + '"', self.source)
            self.assertIn('item.optString("' + field + '"', self.source) if field != "badgeColor" else self.assertIn('item.optInt("badgeColor"', self.source)

    def test_add_and_remove_update_the_saved_catalog(self):
        self.assertIn("projectList.add(0, newItem);\n                    saveProjectCatalog();", self.source)
        self.assertIn("projectList.remove(item);\n                        saveProjectCatalog();", self.source)

    def test_reset_defaults_also_persists(self):
        start = self.source.index("private void initDefaultProjects()")
        end = self.source.index("private boolean loadProjectCatalog()", start)
        self.assertIn("saveProjectCatalog();", self.source[start:end])


if __name__ == "__main__":
    unittest.main()
