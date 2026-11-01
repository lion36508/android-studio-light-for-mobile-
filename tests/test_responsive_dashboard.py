import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"


class ResponsiveDashboardTests(unittest.TestCase):
    def test_quick_actions_use_adaptive_grid_columns(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn("import android.widget.GridLayout;", source)
        self.assertIn("GridLayout cardsRow = new GridLayout(this);", source)
        self.assertIn("screenWidthDp >= 600 ? 4 : 2", source)
        self.assertIn("cardsRow.addView(buildActionCard(", source)

    def test_cards_use_weighted_grid_cells_and_accessible_labels(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn("GridLayout.spec(GridLayout.UNDEFINED, 1f)", source)
        self.assertIn("lp.width = 0;", source)
        self.assertIn("card.setContentDescription((title + \". \" + sub).replace(\"\\n\", \" \"));", source)
        self.assertIn("card.setFocusable(true);", source)

    def test_no_fixed_single_row_container_remains(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertNotIn("LinearLayout cardsRow = new LinearLayout(this);", source)


if __name__ == "__main__":
    unittest.main()
