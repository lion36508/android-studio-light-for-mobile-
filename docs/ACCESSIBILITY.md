# Accessibility implementation notes

OxDroid's programmatic dashboard and editor keep their existing click handlers and navigation. This update adds accessible names and minimum touch sizes without changing those actions.

## Dashboard
- Menu, search, overflow, project overflow, and custom image-menu tap zones expose spoken descriptions.
- Quick-action cards announce one combined title/description rather than separately focusing decorative icon and text children.
- Project cards announce the project name and language/description.
- Bottom-navigation tabs announce their visible label.

## Editor
- Back, Run, Build, Save, and close-console controls have explicit spoken names.
- The Run and Build descriptions accurately disclose that secure execution and on-device building are currently unavailable.
- Source-code editor and file tabs have descriptive names.
- Decorative line numbers are omitted from TalkBack navigation.
- Symbol-row buttons announce the character or action they insert.

## Touch target sizing
Interactive icon-only controls, symbol buttons, menu rows, and key editor actions have a minimum 48dp height/width where applicable. Larger action and project cards keep their existing layouts.

## Validation
The automated checks verify the accessible-name contracts and minimum dimensions in source. Physical TalkBack navigation, switch access, contrast, font scaling, and OEM behavior still require Android-device testing.
