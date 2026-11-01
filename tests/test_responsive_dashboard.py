import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "src/main/java/com/bikash/mobileide/MainActivity.java"


class ResponsiveDashboardTests(unittest.TestCase):
    def test_quick_actions_use_adaptive_two_column_rows(self):
        source = MAIN.read_text(encoding="utf-8")
        # PR #101 replaced the GridLayout dashboard with two-column
        # LinearLayout rows, which are responsive on narrow phones.
        self.assertIn("LinearLayout primaryActionsRow", source)
        self.assertIn("LinearLayout secondaryActionsRow", source)
        self.assertIn("actionsGrid.addView(primaryActionsRow)", source)
        self.assertIn("actionsGrid.addView(secondaryActionsRow)", source)

    def test_cards_use_weighted_cells_and_accessible_labels(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn("LinearLayout.LayoutParams(0, dp(94), 1f)", source)
        self.assertIn('card.setContentDescription((title + ". " + sub).replace("\\n", " "));', source)
        self.assertIn("card.setFocusable(true);", source)

    def test_no_fixed_single_row_container_remains(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertNotIn("LinearLayout cardsRow = new LinearLayout(this);", source)


if __name__ == "__main__":
    unittest.main()
