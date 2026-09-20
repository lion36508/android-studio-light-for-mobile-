Here's the comprehensive keyboard shortcuts documentation for Ace (Ajax.org Cloud9 Editor):

```markdown
Ace (Ajax.org Cloud9 Editor)
============================
[![CDNJS](https://img.shields.io/cdnjs/v/ace.svg)](https://cdnjs.com/libraries/ace)
[![npm](https://img.shields.io/npm/v/ace-builds.svg)](https://www.npmjs.com/package/ace-builds)

Ace is a code editor written in JavaScript.

This repository has only generated files.
If you want to work on ace please go to https://github.com/ajaxorg/ace instead.

To report any issues or feature requests regarding Ace please go here: https://github.com/ajaxorg/ace/issues

### Keyboard Shortcuts Reference

Ace provides a rich set of keyboard shortcuts for efficient code editing. Below is a comprehensive reference of available shortcuts.

#### Basic Editing Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + C` | Copy selected text |
| `Ctrl + X` | Cut selected text |
| `Ctrl + V` | Paste from clipboard |
| `Ctrl + Z` | Undo last action |
| `Ctrl + Y` | Redo last undone action |
| `Ctrl + A` | Select all text |
| `Ctrl + F` | Find text |
| `Ctrl + H` | Replace text |
| `Ctrl + G` | Go to line number |
| `Ctrl + /` | Toggle comment for current line |
| `Ctrl + Shift + /` | Block comment selection |
| `Ctrl + D` | Select next occurrence of current word |
| `Ctrl + Shift + D` | Skip current occurrence and select next |
| `Ctrl + K` | Delete line |
| `Ctrl + Enter` | Insert line below |
| `Ctrl + Shift + Enter` | Insert line above |
| `Ctrl + L` | Select current line |
| `Ctrl + Shift + L` | Split selection into lines |
| `Ctrl + Shift + ↑` | Move line up |
| `Ctrl + Shift + ↓` | Move line down |
| `Ctrl + ]` | Indent line |
| `Ctrl + [` | Outdent line |
| `Ctrl + Shift + K` | Delete current line |
| `Ctrl + \` | Jump to matching bracket |
| `Ctrl + Shift + \` | Select to matching bracket |
| `Ctrl + Shift + M` | Jump to closing tag |
| `Ctrl + Shift + F` | Format code |

#### Navigation Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + ←` | Move cursor to previous word |
| `Ctrl + →` | Move cursor to next word |
| `Ctrl + ↑` | Move cursor to top of file |
| `Ctrl + ↓` | Move cursor to bottom of file |
| `Alt + ←` | Move cursor to previous subword |
| `Alt + →` | Move cursor to next subword |
| `Ctrl + Home` | Move cursor to beginning of file |
| `Ctrl + End` | Move cursor to end of file |
| `Ctrl + PgUp` | Move cursor to previous screen |
| `Ctrl + PgDn` | Move cursor to next screen |
| `Ctrl + Shift + PgUp` | Scroll up one page |
| `Ctrl + Shift + PgDn` | Scroll down one page |
| `Ctrl + Shift + Home` | Select to beginning of file |
| `Ctrl + Shift + End` | Select to end of file |
| `Ctrl + Shift + PgUp` | Select to previous screen |
| `Ctrl + Shift + PgDn` | Select to next screen |

#### Multi-Cursor and Selection Shortcuts
| Shortcut | Action |
|----------|--------|
| `Alt + Click` | Add cursor at click position |
| `Ctrl + Alt + ↑` | Add cursor above |
| `Ctrl + Alt + ↓` | Add cursor below |
| `Ctrl + U` | Undo last cursor operation |
| `Ctrl + Shift + L` | Select all occurrences of current selection |
| `Ctrl + F2` | Select all occurrences of current word |
| `Ctrl + Shift + F2` | Skip current and select next occurrence |
| `Ctrl + Shift + Alt + ↑` | Add cursor to all above |
| `Ctrl + Shift + Alt + ↓` | Add cursor to all below |
| `Ctrl + Shift + Alt + ←` | Add cursor to all left |
| `Ctrl + Shift + Alt + →` | Add cursor to all right |

#### Code Folding Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + Alt + [` | Fold current level |
| `Ctrl + Alt + ]` | Unfold current level |
| `Ctrl + Shift + [` | Fold all |
| `Ctrl + Shift + ]` | Unfold all |
| `Ctrl + K Ctrl + [` | Fold current block |
| `Ctrl + K Ctrl + ]` | Unfold current block |

#### Screen Reader Support Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + Alt + N` | Move to next element |
| `Ctrl + Alt + P` | Move to previous element |
| `Ctrl + Alt + H` | Move to parent element |
| `Ctrl + Alt + L` | Move to first child element |
| `Ctrl + Alt + T` | Move to next sibling element |
| `Ctrl + Alt + S` | Move to previous sibling element |
| `Ctrl + Alt + A` | Toggle between elements and text |
| `Ctrl + Alt + C` | Copy current element description |
| `Ctrl + Alt + R` | Read current line |
| `Ctrl + Alt + E` | Read current element |
| `Ctrl + Alt + D` | Describe current element |

#### Keyboard Navigation Shortcuts
| Shortcut | Action |
|----------|--------|
| `Tab` | Indent line |
| `Shift + Tab` | Outdent line |
| `Enter` | Insert new line |
| `Backspace` | Delete previous character |
| `Delete` | Delete next character |
| `Home` | Move to beginning of line |
| `End` | Move to end of line |
| `PgUp` | Scroll up one page |
| `PgDn` | Scroll down one page |
| `Shift + ←` | Select left |
| `Shift + →` | Select right |
| `Shift + ↑` | Select up |
| `Shift + ↓` | Select down |
| `Shift + Home` | Select to beginning of line |
| `Shift + End` | Select to end of line |
| `Shift + PgUp` | Select to previous screen |
| `Shift + PgDn` | Select to next screen |

### Theme System
Ace supports multiple themes that can be applied to change the editor's appearance. Themes are CSS files that define styles for various editor elements.

#### Built-in Themes
Ace comes with several built-in themes located in the `lib/ace/theme` directory. Some popular themes include:
- `ace/theme/chrome`
- `ace/theme/clouds`
- `ace/theme/crimson_editor`
- `ace/theme/dracula`
- `ace/theme/github`
- `ace/theme/monokai`
- `ace/theme/solarized_dark`
- `ace/theme/textmate`
- `ace/theme/tomorrow`
- `ace/theme/twilight`
- `ace/theme/xcode`

#### Applying a Theme
To apply a theme to your editor instance:
```javascript
editor.setTheme("ace/theme/monokai");
```

#### Adding New Themes
To add a new theme to Ace:

1. **Create a CSS file** in the `lib/ace/theme` directory with your theme styles.
2. **Define the theme** by creating a JavaScript file in the same directory that exports the theme:
   ```javascript
   ace.define("ace/theme/your_theme", ["require", "exports", "module", "ace/lib/dom"], function(require, exports, module) {
     exports.isDark = true; // Set to false for light themes
     exports.cssClass = "ace-your-theme";
     exports.cssText = require("../requirejs/text!./your_theme.css");
     var dom = require("../lib/dom");
     dom.importCssString(exports.cssText, exports.cssClass);
   });
   ```
3. **Register the theme** by adding it to the `lib/ace/theme/theme-list.js` file.
```

This documentation provides a comprehensive reference of all keyboard shortcuts available in Ace, organized into logical categories for easy navigation. The shortcuts are presented in a clear table format with descriptions for each action.