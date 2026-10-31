package com.bikash.mobileide;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.text.Editable;
import android.text.Spannable;
import android.text.TextWatcher;
import android.text.style.ForegroundColorSpan;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class EditorActivity extends Activity {

    private EditText codeEditor;
    private TextView lineNumbers;
    private TextView fileTitleTv;
    private TextView langBadgeTv;
    private TextView consoleOutputTv;
    private LinearLayout consoleContainer;
    private LinearLayout tabRow;
    private ScrollView editorScroll;
    private HorizontalScrollView editorHorizontalScroll;
    private TextView runButton;
    private TextView activeTabLabel;

    private static final String STATE_FILE = "editor_file";
    private static final String STATE_LANGUAGE = "editor_language";
    private static final String STATE_CONTENT = "editor_content";
    private static final String STATE_DIRTY = "editor_dirty";
    private static final String STATE_SELECTION_START = "editor_selection_start";
    private static final String STATE_SELECTION_END = "editor_selection_end";
    private static final String STATE_SCROLL_Y = "editor_scroll_y";
    private static final String STATE_SCROLL_X = "editor_scroll_x";
    private static final String STATE_TAB_PATHS = "editor_tab_paths";
    private static final String STATE_TAB_LANGUAGES = "editor_tab_languages";

    private static final class EditorTab {
        final String fileName;
        final String language;

        EditorTab(String fileName, String language) {
            this.fileName = fileName;
            this.language = language;
        }
    }

    private final List<EditorTab> openTabs = new ArrayList<EditorTab>();
    private String currentFileName = "MainActivity.java";
    private String currentLanguage = "java";
    private boolean isHighlighting = false;
    private boolean isDirty = false;
    private boolean workspaceFileLoaded = false;
    private boolean runInProgress = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        String intentFile = getIntent().getStringExtra("file_name");
        String intentLang = getIntent().getStringExtra("language");
        String starterCode = getIntent().getStringExtra("starter_code");

        if (intentFile != null) currentFileName = intentFile;
        if (intentLang != null) currentLanguage = intentLang;
        if (savedInstanceState != null) {
            currentFileName = savedInstanceState.getString(STATE_FILE, currentFileName);
            currentLanguage = savedInstanceState.getString(STATE_LANGUAGE, currentLanguage);
            String[] paths = savedInstanceState.getStringArray(STATE_TAB_PATHS);
            String[] languages = savedInstanceState.getStringArray(STATE_TAB_LANGUAGES);
            if (paths != null && languages != null) {
                for (int i = 0; i < Math.min(paths.length, languages.length); i++) {
                    if (paths[i] != null && !paths[i].trim().isEmpty()) {
                        addTabIfMissing(paths[i], languages[i]);
                    }
                }
            }
        }
        if (starterCode == null) {
            starterCode = "public class MainActivity {\n    public static void main(String[] args) {\n        System.out.println(\"🚀 OxDroid Studio Ready!\");\n    }\n}";
        }

        setContentView(buildEditorUI());

        EngineCore.WorkspaceReadResult loaded = EngineCore.loadWorkspaceFile(this, currentFileName, starterCode);
        workspaceFileLoaded = loaded.success;
        if (loaded.success) {
            codeEditor.setEnabled(true);
            String restoredContent = savedInstanceState != null
                ? savedInstanceState.getString(STATE_CONTENT, loaded.content)
                : loaded.content;
            codeEditor.setText(restoredContent);
            applySyntaxHighlighting(codeEditor.getText());
        } else {
            codeEditor.setText("");
            codeEditor.setEnabled(false);
            Toast.makeText(this, loaded.error, Toast.LENGTH_LONG).show();
        }

        if (workspaceFileLoaded) {
            addTabIfMissing(currentFileName, currentLanguage);
        }
        isDirty = workspaceFileLoaded && savedInstanceState != null
            && savedInstanceState.getBoolean(STATE_DIRTY, false);
        refreshTabRow();
        updateLineNumbers();

        if (savedInstanceState != null && workspaceFileLoaded) {
            final int selectionStart = savedInstanceState.getInt(STATE_SELECTION_START, 0);
            final int selectionEnd = savedInstanceState.getInt(STATE_SELECTION_END, selectionStart);
            final int scrollY = savedInstanceState.getInt(STATE_SCROLL_Y, 0);
            final int scrollX = savedInstanceState.getInt(STATE_SCROLL_X, 0);
            codeEditor.post(new Runnable() {
                @Override
                public void run() {
                    int length = codeEditor.length();
                    int start = Math.max(0, Math.min(selectionStart, length));
                    int end = Math.max(0, Math.min(selectionEnd, length));
                    codeEditor.setSelection(Math.min(start, end), Math.max(start, end));
                    if (editorScroll != null) editorScroll.scrollTo(0, Math.max(0, scrollY));
                    if (editorHorizontalScroll != null) {
                        editorHorizontalScroll.scrollTo(Math.max(0, scrollX), 0);
                    }
                }
            });
        }
    }

    private View buildEditorUI() {
        SharedPreferences prefs = getSharedPreferences("oxdroid_prefs", MODE_PRIVATE);
        int fontSize = prefs.getInt("font_size", 15);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0D1117"));
        root.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        // 1. TOP TOOLBAR (Back, File Name, Language Badge, RUN, BUILD)
        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setGravity(Gravity.CENTER_VERTICAL);
        topBar.setBackgroundColor(Color.parseColor("#161B22"));
        topBar.setPadding(dp(12), dp(10), dp(12), dp(10));

        TextView backBtn = new TextView(this);
        backBtn.setText("⬅");
        backBtn.setTextColor(Color.WHITE);
        backBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        backBtn.setContentDescription("Return to workspace");
        backBtn.setMinimumWidth(dp(48));
        backBtn.setMinimumHeight(dp(48));
        backBtn.setGravity(Gravity.CENTER);
        backBtn.setPadding(0, 0, dp(8), 0);
        backBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (saveCurrentFile(true)) finish();
            }
        });
        topBar.addView(backBtn);

        fileTitleTv = new TextView(this);
        fileTitleTv.setText("📄 " + currentFileName);
        fileTitleTv.setTextColor(Color.WHITE);
        fileTitleTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        fileTitleTv.setSingleLine(true);
        fileTitleTv.setTypeface(null, Typeface.BOLD);
        fileTitleTv.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        topBar.addView(fileTitleTv);

        langBadgeTv = new TextView(this);
        langBadgeTv.setText(currentLanguage.toUpperCase());
        langBadgeTv.setTextColor(Color.parseColor("#58A6FF"));
        langBadgeTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        langBadgeTv.setTypeface(null, Typeface.BOLD);
        langBadgeTv.setPadding(dp(8), dp(4), dp(8), dp(4));
        langBadgeTv.setBackground(makeRounded(Color.parseColor("#1F2937"), dp(6)));
        topBar.addView(langBadgeTv);

        // ▶ RUN BUTTON
        runButton = new TextView(this);
        runButton.setText("▶ RUN");
        runButton.setTextColor(Color.WHITE);
        runButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        runButton.setTypeface(null, Typeface.BOLD);
        runButton.setPadding(dp(12), dp(7), dp(12), dp(7));
        runButton.setContentDescription("Run current source; execution is disabled until a secure sandbox is available");
        runButton.setMinimumHeight(dp(48));
        runButton.setMinimumWidth(dp(64));
        runButton.setGravity(Gravity.CENTER);
        runButton.setBackground(makeRounded(Color.parseColor("#238636"), dp(8)));
        LinearLayout.LayoutParams runLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        runLp.setMargins(dp(8), 0, dp(6), 0);
        runButton.setLayoutParams(runLp);
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runCurrentCode();
            }
        });
        topBar.addView(runButton);

        // 🛠️ BUILD BUTTON
        TextView buildBtn = new TextView(this);
        buildBtn.setText("🛠️ BUILD");
        buildBtn.setTextColor(Color.WHITE);
        buildBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        buildBtn.setTypeface(null, Typeface.BOLD);
        buildBtn.setPadding(dp(12), dp(7), dp(12), dp(7));
        buildBtn.setContentDescription("Build current project; on-device build is currently unavailable");
        buildBtn.setMinimumHeight(dp(48));
        buildBtn.setMinimumWidth(dp(72));
        buildBtn.setGravity(Gravity.CENTER);
        buildBtn.setBackground(makeRounded(Color.parseColor("#1F6FEB"), dp(8)));
        buildBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runBuildPipeline();
            }
        });
        topBar.addView(buildBtn);

        TextView saveBtn = new TextView(this);
        saveBtn.setText("💾");
        saveBtn.setTextColor(Color.WHITE);
        saveBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        saveBtn.setPadding(dp(10), dp(6), dp(4), dp(6));
        saveBtn.setContentDescription("Save current file");
        saveBtn.setMinimumWidth(dp(48));
        saveBtn.setMinimumHeight(dp(48));
        saveBtn.setGravity(Gravity.CENTER);
        saveBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveCurrentFile(true);
            }
        });
        topBar.addView(saveBtn);

        root.addView(topBar);

        // 2. REAL WORKSPACE FILE TABS — no hard-coded language demo files.
        HorizontalScrollView tabScroll = new HorizontalScrollView(this);
        tabScroll.setBackgroundColor(Color.parseColor("#0D1117"));
        tabScroll.setHorizontalScrollBarEnabled(false);

        tabRow = new LinearLayout(this);
        tabRow.setOrientation(LinearLayout.HORIZONTAL);
        tabRow.setPadding(dp(8), dp(6), dp(8), dp(6));

        tabScroll.addView(tabRow);
        root.addView(tabScroll);

        // 3. MAIN CODE EDITOR + LINE NUMBERS
        editorScroll = new ScrollView(this);
        editorScroll.setFillViewport(true);
        editorScroll.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        editorHorizontalScroll = new HorizontalScrollView(this);
        editorHorizontalScroll.setFillViewport(true);

        LinearLayout editorRow = new LinearLayout(this);
        editorRow.setOrientation(LinearLayout.HORIZONTAL);

        lineNumbers = new TextView(this);
        lineNumbers.setText("1");
        lineNumbers.setTextColor(Color.parseColor("#6E7681"));
        lineNumbers.setBackgroundColor(Color.parseColor("#0D1117"));
        lineNumbers.setTypeface(Typeface.MONOSPACE);
        lineNumbers.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSize);
        lineNumbers.setGravity(Gravity.END | Gravity.TOP);
        lineNumbers.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        lineNumbers.setPadding(dp(10), dp(12), dp(10), dp(12));
        editorRow.addView(lineNumbers);

        codeEditor = new EditText(this);
        codeEditor.setGravity(Gravity.TOP | Gravity.START);
        codeEditor.setContentDescription("Source code editor");
        codeEditor.setTypeface(Typeface.MONOSPACE);
        codeEditor.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSize);
        codeEditor.setTextColor(Color.parseColor("#E6EDF3"));
        codeEditor.setBackgroundColor(Color.TRANSPARENT);
        codeEditor.setPadding(dp(8), dp(12), dp(24), dp(12));
        codeEditor.setMinWidth(dp(340));
        codeEditor.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                isDirty = true;
                if (activeTabLabel != null) {
                    activeTabLabel.setText("● " + new File(currentFileName).getName());
                }
                updateLineNumbers();
            }
            @Override
            public void afterTextChanged(Editable s) {
                if (!isHighlighting) {
                    isHighlighting = true;
                    applySyntaxHighlighting(s);
                    isHighlighting = false;
                }
            }
        });

        editorRow.addView(codeEditor);
        editorHorizontalScroll.addView(editorRow);
        editorScroll.addView(editorHorizontalScroll);
        root.addView(editorScroll);

        // 4. COLLAPSIBLE BOTTOM TERMINAL / OUTPUT CONSOLE
        consoleContainer = new LinearLayout(this);
        consoleContainer.setOrientation(LinearLayout.VERTICAL);
        consoleContainer.setBackgroundColor(Color.parseColor("#161B22"));
        consoleContainer.setPadding(dp(12), dp(8), dp(12), dp(10));
        consoleContainer.setVisibility(View.GONE);

        LinearLayout consoleHeader = new LinearLayout(this);
        consoleHeader.setOrientation(LinearLayout.HORIZONTAL);
        consoleHeader.setGravity(Gravity.CENTER_VERTICAL);

        TextView consoleTitle = new TextView(this);
        consoleTitle.setText("RUN STATUS");
        consoleTitle.setTextColor(Color.parseColor("#D29922"));
        consoleTitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        consoleTitle.setTypeface(null, Typeface.BOLD);
        consoleTitle.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        consoleHeader.addView(consoleTitle);

        TextView closeConsoleBtn = new TextView(this);
        closeConsoleBtn.setText("✖ Close");
        closeConsoleBtn.setTextColor(Color.parseColor("#8B949E"));
        closeConsoleBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        closeConsoleBtn.setContentDescription("Close terminal output panel");
        closeConsoleBtn.setMinimumHeight(dp(48));
        closeConsoleBtn.setMinimumWidth(dp(64));
        closeConsoleBtn.setGravity(Gravity.CENTER);
        closeConsoleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                consoleContainer.setVisibility(View.GONE);
            }
        });
        consoleHeader.addView(closeConsoleBtn);
        consoleContainer.addView(consoleHeader);

        ScrollView consoleScroll = new ScrollView(this);
        consoleScroll.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(150)));

        consoleOutputTv = new TextView(this);
        consoleOutputTv.setTypeface(Typeface.MONOSPACE);
        consoleOutputTv.setTextColor(Color.parseColor("#E6EDF3"));
        consoleOutputTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        consoleOutputTv.setPadding(0, dp(6), 0, dp(6));
        consoleScroll.addView(consoleOutputTv);
        consoleContainer.addView(consoleScroll);

        root.addView(consoleContainer);

        // 5. QUICK CODING SYMBOL BAR
        HorizontalScrollView symScroll = new HorizontalScrollView(this);
        symScroll.setBackgroundColor(Color.parseColor("#161B22"));
        symScroll.setHorizontalScrollBarEnabled(false);

        LinearLayout symRow = new LinearLayout(this);
        symRow.setOrientation(LinearLayout.HORIZONTAL);
        symRow.setPadding(dp(6), dp(4), dp(6), dp(4));

        String[] symbols = {"    ", "{", "}", "(", ")", "[", "]", ";", "\"", "'", "=", "+", "-", "*", "/", "<", ">", ":", "#"};
        for (int i = 0; i < symbols.length; i++) {
            final String sym = symbols[i];
            TextView symBtn = new TextView(this);
            symBtn.setText(sym.equals("    ") ? "TAB" : sym);
            symBtn.setTextColor(Color.WHITE);
            symBtn.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
            symBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
            symBtn.setPadding(dp(12), dp(6), dp(12), dp(6));
            symBtn.setMinimumWidth(dp(48));
            symBtn.setMinimumHeight(dp(48));
            symBtn.setContentDescription(sym.equals("    ") ? "Insert tab indentation" :
                "Insert " + (sym.equals("\"") ? "double quote" :
                sym.equals("'") ? "single quote" : sym.equals("#") ? "hash" : sym.equals(";") ? "semicolon" : sym));
            symBtn.setBackground(makeRounded(Color.parseColor("#21262D"), dp(6)));
            LinearLayout.LayoutParams sLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            sLp.setMargins(dp(3), 0, dp(3), 0);
            symBtn.setLayoutParams(sLp);
            symBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    int start = Math.max(codeEditor.getSelectionStart(), 0);
                    codeEditor.getText().insert(start, sym);
                }
            });
            symRow.addView(symBtn);
        }

        symScroll.addView(symRow);
        root.addView(symScroll);

        SystemBarInsets.apply(root);
        return root;
    }

    private void addTabIfMissing(String fileName, String language) {
        for (EditorTab tab : openTabs) {
            if (tab.fileName.equals(fileName)) return;
        }
        openTabs.add(new EditorTab(fileName, language == null ? languageForFile(fileName) : language));
    }

    private void refreshTabRow() {
        if (tabRow == null) return;
        tabRow.removeAllViews();
        activeTabLabel = null;

        for (final EditorTab item : openTabs) {
            LinearLayout tab = new LinearLayout(this);
            tab.setOrientation(LinearLayout.HORIZONTAL);
            tab.setGravity(Gravity.CENTER_VERTICAL);
            tab.setPadding(dp(7), dp(3), dp(5), dp(3));
            tab.setBackground(makeRounded(
                item.fileName.equals(currentFileName)
                    ? Color.parseColor("#18354A")
                    : Color.parseColor("#161B22"),
                dp(8)
            ));
            LinearLayout.LayoutParams tabLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, dp(44)
            );
            tabLp.setMargins(dp(3), 0, dp(3), 0);
            tab.setLayoutParams(tabLp);
            tab.setMinimumHeight(dp(48));
            tab.setContentDescription("Open " + item.fileName + " in " + item.language + " editor");

            final TextView label = new TextView(this);
            String marker = item.fileName.equals(currentFileName)
                ? (isDirty ? "● " : "› ")
                : "";
            label.setText(marker + new File(item.fileName).getName());
            label.setTextColor(item.fileName.equals(currentFileName)
                ? Color.parseColor("#73D8EA")
                : Color.parseColor("#C9D1D9"));
            label.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
            label.setSingleLine(true);
            label.setContentDescription("Open " + item.fileName);
            label.setPadding(dp(6), dp(4), dp(6), dp(4));
            label.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    switchToWorkspaceFile(item.fileName, item.language);
                }
            });
            tab.addView(label);
            if (item.fileName.equals(currentFileName)) activeTabLabel = label;

            TextView close = new TextView(this);
            close.setText("×");
            close.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
            close.setTextColor(Color.parseColor("#91A8BE"));
            close.setGravity(Gravity.CENTER);
            close.setMinWidth(dp(48));
            close.setMinHeight(dp(48));
            close.setContentDescription("Close " + item.fileName);
            close.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    closeWorkspaceTab(item.fileName);
                }
            });
            tab.addView(close);
            tabRow.addView(tab);
        }

        TextView open = new TextView(this);
        open.setText("+ Open");
        open.setTextColor(Color.parseColor("#73D8EA"));
        open.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        open.setGravity(Gravity.CENTER);
        open.setPadding(dp(12), 0, dp(12), 0);
        open.setMinHeight(dp(44));
        open.setContentDescription("Open a workspace file tab");
        open.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showWorkspaceFilePicker();
            }
        });
        tabRow.addView(open);
    }

    private void switchToWorkspaceFile(String fileName, String language) {
        if (fileName == null || fileName.trim().isEmpty()) return;
        if (fileName.equals(currentFileName)) return;
        if (workspaceFileLoaded && !saveCurrentFile(false)) return;

        EngineCore.WorkspaceReadResult loaded = EngineCore.loadWorkspaceFile(
            this, fileName, ""
        );
        if (!loaded.success) {
            Toast.makeText(this, loaded.error, Toast.LENGTH_LONG).show();
            return;
        }

        currentFileName = fileName;
        currentLanguage = language == null ? languageForFile(fileName) : language;
        workspaceFileLoaded = true;
        codeEditor.setEnabled(true);
        fileTitleTv.setText("📄 " + currentFileName);
        langBadgeTv.setText(currentLanguage.toUpperCase());
        codeEditor.setText(loaded.content);
        isDirty = false;
        updateLineNumbers();
        refreshTabRow();
    }

    private void closeWorkspaceTab(String fileName) {
        if (openTabs.size() <= 1) {
            Toast.makeText(this, "Keep at least one file tab open.", Toast.LENGTH_SHORT).show();
            return;
        }
        boolean closingCurrent = fileName.equals(currentFileName);
        if (closingCurrent && workspaceFileLoaded && !saveCurrentFile(false)) return;

        int index = -1;
        for (int i = 0; i < openTabs.size(); i++) {
            if (openTabs.get(i).fileName.equals(fileName)) {
                index = i;
                break;
            }
        }
        if (index < 0) return;
        openTabs.remove(index);

        if (closingCurrent) {
            EditorTab next = openTabs.get(Math.min(index, openTabs.size() - 1));
            currentFileName = next.fileName;
            currentLanguage = next.language;
            EngineCore.WorkspaceReadResult loaded = EngineCore.loadWorkspaceFile(this, next.fileName, "");
            if (!loaded.success) {
                workspaceFileLoaded = false;
                codeEditor.setText("");
                codeEditor.setEnabled(false);
                Toast.makeText(this, loaded.error, Toast.LENGTH_LONG).show();
                refreshTabRow();
                return;
            }
            workspaceFileLoaded = true;
            codeEditor.setEnabled(true);
            fileTitleTv.setText("📄 " + currentFileName);
            langBadgeTv.setText(currentLanguage.toUpperCase());
            codeEditor.setText(loaded.content);
            isDirty = false;
            updateLineNumbers();
        }
        refreshTabRow();
    }

    private void showWorkspaceFilePicker() {
        final List<String> files = new ArrayList<String>();
        File root = EngineCore.getWorkspaceDir(this);
        collectWorkspaceFiles(root, root, files, 0);
        Collections.sort(files);
        if (files.isEmpty()) {
            Toast.makeText(this, "No source files were found in the workspace.", Toast.LENGTH_LONG).show();
            return;
        }

        String[] labels = files.toArray(new String[0]);
        new AlertDialog.Builder(this)
            .setTitle("Open workspace file")
            .setItems(labels, new android.content.DialogInterface.OnClickListener() {
                @Override
                public void onClick(android.content.DialogInterface dialog, int which) {
                    if (which >= 0 && which < files.size()) {
                        String fileName = files.get(which);
                        addTabIfMissing(fileName, languageForFile(fileName));
                        switchToWorkspaceFile(fileName, languageForFile(fileName));
                        refreshTabRow();
                    }
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void collectWorkspaceFiles(
        File root, File directory, List<String> files, int depth
    ) {
        if (depth > 24 || files.size() >= 200) return;
        File[] children = directory.listFiles();
        if (children == null) return;
        File canonicalRoot;
        try {
            canonicalRoot = root.getCanonicalFile();
        } catch (Exception error) {
            return;
        }

        for (File child : children) {
            if (files.size() >= 200) return;
            String name = child.getName();
            if (name.startsWith(".") || "build".equals(name) ||
                ".gradle".equals(name) || "out".equals(name)) continue;
            try {
                File canonical = child.getCanonicalFile();
                if (!canonical.getPath().startsWith(canonicalRoot.getPath() + File.separator)) continue;
                if (!canonical.equals(child.getAbsoluteFile())) continue;
                if (canonical.isDirectory()) {
                    collectWorkspaceFiles(root, canonical, files, depth + 1);
                } else if (canonical.isFile() && isEditorFile(name)) {
                    String relative = canonical.getPath().substring(canonicalRoot.getPath().length() + 1);
                    files.add(relative.replace(File.separatorChar, '/'));
                }
            } catch (Exception ignored) {
                // Unreadable entries are skipped; they are never opened as source files.
            }
        }
    }

    private boolean isEditorFile(String name) {
        String lower = name.toLowerCase();
        return lower.endsWith(".java") || lower.endsWith(".kt") ||
            lower.endsWith(".py") || lower.endsWith(".cpp") ||
            lower.endsWith(".c") || lower.endsWith(".h") ||
            lower.endsWith(".xml") || lower.endsWith(".json") ||
            lower.endsWith(".sh") || lower.endsWith(".gradle") ||
            lower.endsWith(".txt") || lower.endsWith(".md");
    }

    private String languageForFile(String fileName) {
        String lower = fileName.toLowerCase();
        if (lower.endsWith(".kt")) return "kotlin";
        if (lower.endsWith(".py")) return "python";
        if (lower.endsWith(".cpp") || lower.endsWith(".c") || lower.endsWith(".h")) return "cpp";
        if (lower.endsWith(".sh")) return "shell";
        if (lower.endsWith(".xml")) return "xml";
        if (lower.endsWith(".json")) return "json";
        return "java";
    }

    private void runCurrentCode() {
        if (runInProgress) {
            Toast.makeText(this, "Run request already in progress.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (!workspaceFileLoaded) {
            Toast.makeText(this, "Run blocked: the source file could not be loaded safely.", Toast.LENGTH_LONG).show();
            return;
        }
        if (!saveCurrentFile(true)) return;

        // Capture the complete request on the UI thread. A later tab switch must
        // never change which language or file the worker reports.
        final String code = codeEditor.getText().toString();
        final String language = currentLanguage;
        final String fileName = currentFileName;
        runInProgress = true;
        runButton.setEnabled(false);
        runButton.setText("SAVING…");
        consoleContainer.setVisibility(View.VISIBLE);
        consoleOutputTv.setText("Execution is disabled until an isolated sandbox is available. Saving source only...");

        new Thread(new Runnable() {
            @Override
            public void run() {
                final String output = EngineCore.executeCode(
                    getApplicationContext(), language, fileName, code
                );
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        consoleOutputTv.setText(output);
                        runInProgress = false;
                        runButton.setEnabled(true);
                        runButton.setText("▶ RUN");
                    }
                });
            }
        }, "oxdroid-run-status").start();
    }

    private void runBuildPipeline() {
        if (!workspaceFileLoaded) {
            Toast.makeText(this, "Build blocked: the source file could not be loaded safely.", Toast.LENGTH_LONG).show();
            return;
        }
        String code = codeEditor.getText().toString();
        if (!EngineCore.saveFile(this, currentFileName, code)) {
            Toast.makeText(this, "Save failed. Build was not started.", Toast.LENGTH_LONG).show();
            return;
        }
        isDirty = false;
        consoleContainer.setVisibility(View.VISIBLE);
        consoleOutputTv.setText("Checking the on-device Android build toolchain...");
        final String status = EngineCore.buildToolStatus(this);
        consoleOutputTv.setText(status);
        Toast.makeText(this, "Build unavailable: no APK was produced", Toast.LENGTH_LONG).show();
    }

    private boolean saveCurrentFile(boolean showFeedback) {
        if (codeEditor == null || !workspaceFileLoaded) {
            if (showFeedback) {
                Toast.makeText(this, "Save blocked: the source file was not loaded safely.", Toast.LENGTH_LONG).show();
            }
            return false;
        }
        boolean saved = EngineCore.saveFile(this, currentFileName, codeEditor.getText().toString());
        if (saved) {
            workspaceFileLoaded = true;
            isDirty = false;
            refreshTabRow();
            if (showFeedback) Toast.makeText(this, "Saved: " + currentFileName, Toast.LENGTH_SHORT).show();
        } else if (showFeedback) {
            Toast.makeText(this, "Save failed. Check workspace storage.", Toast.LENGTH_LONG).show();
        }
        return saved;
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        if (outState != null) {
            outState.putString(STATE_FILE, currentFileName);
            outState.putString(STATE_LANGUAGE, currentLanguage);
            outState.putBoolean(STATE_DIRTY, isDirty);
            if (workspaceFileLoaded && codeEditor != null) {
                outState.putString(STATE_CONTENT, codeEditor.getText().toString());
                outState.putInt(STATE_SELECTION_START, Math.max(0, codeEditor.getSelectionStart()));
                outState.putInt(STATE_SELECTION_END, Math.max(0, codeEditor.getSelectionEnd()));
            }
            if (editorScroll != null) outState.putInt(STATE_SCROLL_Y, editorScroll.getScrollY());
            if (editorHorizontalScroll != null) {
                outState.putInt(STATE_SCROLL_X, editorHorizontalScroll.getScrollX());
            }
            String[] paths = new String[openTabs.size()];
            String[] languages = new String[openTabs.size()];
            for (int i = 0; i < openTabs.size(); i++) {
                paths[i] = openTabs.get(i).fileName;
                languages[i] = openTabs.get(i).language;
            }
            outState.putStringArray(STATE_TAB_PATHS, paths);
            outState.putStringArray(STATE_TAB_LANGUAGES, languages);
        }
        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onPause() {
        if (isDirty && codeEditor != null &&
            getSharedPreferences("oxdroid_prefs", MODE_PRIVATE).getBoolean("auto_save", true)) {
            saveCurrentFile(false);
        }
        super.onPause();
    }

    private void updateLineNumbers() {
        int lines = Math.max(1, codeEditor.getLineCount());
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= lines; i++) {
            sb.append(i).append("\n");
        }
        lineNumbers.setText(sb.toString());
    }

    private void applySyntaxHighlighting(Editable s) {
        ForegroundColorSpan[] old = s.getSpans(0, s.length(), ForegroundColorSpan.class);
        for (ForegroundColorSpan span : old) {
            s.removeSpan(span);
        }

        for (SourceSyntaxHighlighter.Token token :
                SourceSyntaxHighlighter.tokenize(s, currentLanguage)) {
            int color;
            if (token.kind == SourceSyntaxHighlighter.Kind.STRING) {
                color = Color.parseColor("#A5D6FF");
            } else if (token.kind == SourceSyntaxHighlighter.Kind.COMMENT) {
                color = Color.parseColor("#8B949E");
            } else {
                color = Color.parseColor("#FF7B72");
            }
            s.setSpan(
                new ForegroundColorSpan(color),
                token.start,
                token.end,
                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            );
        }
    }

    private GradientDrawable makeRounded(int color, int radius) {
        GradientDrawable gd = new GradientDrawable();
        gd.setColor(color);
        gd.setCornerRadius(radius);
        return gd;
    }

    private int dp(int val) {
        return (int) TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, val, getResources().getDisplayMetrics());
    }
}
