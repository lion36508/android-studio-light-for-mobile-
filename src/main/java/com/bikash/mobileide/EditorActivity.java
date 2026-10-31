package com.bikash.mobileide;

import android.app.Activity;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class EditorActivity extends Activity {

    private EditText codeEditor;
    private TextView lineNumbers;
    private TextView fileTitleTv;
    private TextView langBadgeTv;
    private TextView consoleOutputTv;
    private LinearLayout consoleContainer;

    private String currentFileName = "MainActivity.java";
    private String currentLanguage = "java";
    private boolean isHighlighting = false;
    private boolean isDirty = false;
    private boolean workspaceFileLoaded = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        String intentFile = getIntent().getStringExtra("file_name");
        String intentLang = getIntent().getStringExtra("language");
        String starterCode = getIntent().getStringExtra("starter_code");

        if (intentFile != null) currentFileName = intentFile;
        if (intentLang != null) currentLanguage = intentLang;
        if (starterCode == null) {
            starterCode = "public class MainActivity {\n    public static void main(String[] args) {\n        System.out.println(\"🚀 OxDroid Studio Ready!\");\n    }\n}";
        }

        setContentView(buildEditorUI());

        EngineCore.WorkspaceReadResult loaded = EngineCore.loadWorkspaceFile(this, currentFileName, starterCode);
        workspaceFileLoaded = loaded.success;
        if (loaded.success) {
            codeEditor.setEnabled(true);
            codeEditor.setText(loaded.content);
            applySyntaxHighlighting(codeEditor.getText());
        } else {
            codeEditor.setText("");
            codeEditor.setEnabled(false);
            Toast.makeText(this, loaded.error, Toast.LENGTH_LONG).show();
        }
        isDirty = false;
        updateLineNumbers();
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
        TextView runBtn = new TextView(this);
        runBtn.setText("▶ RUN");
        runBtn.setTextColor(Color.WHITE);
        runBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        runBtn.setTypeface(null, Typeface.BOLD);
        runBtn.setPadding(dp(12), dp(7), dp(12), dp(7));
        runBtn.setContentDescription("Run current source; execution is disabled until a secure sandbox is available");
        runBtn.setMinimumHeight(dp(48));
        runBtn.setMinimumWidth(dp(64));
        runBtn.setGravity(Gravity.CENTER);
        runBtn.setBackground(makeRounded(Color.parseColor("#238636"), dp(8)));
        LinearLayout.LayoutParams runLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        runLp.setMargins(dp(8), 0, dp(6), 0);
        runBtn.setLayoutParams(runLp);
        runBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runCurrentCode();
            }
        });
        topBar.addView(runBtn);

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

        // 2. MULTI-LANGUAGE FILE SWITCHER TABS
        HorizontalScrollView tabScroll = new HorizontalScrollView(this);
        tabScroll.setBackgroundColor(Color.parseColor("#0D1117"));
        tabScroll.setHorizontalScrollBarEnabled(false);

        LinearLayout tabRow = new LinearLayout(this);
        tabRow.setOrientation(LinearLayout.HORIZONTAL);
        tabRow.setPadding(dp(8), dp(6), dp(8), dp(6));

        tabRow.addView(buildLangTab("☕ MainActivity.java", "java", "MainActivity.java",
            "public class MainActivity {\n    public static void main(String[] args) {\n        String ide = \"OxDroid Studio\";\n        System.out.println(\"☕ Java Engine Running inside \" + ide);\n    }\n}"));
        tabRow.addView(buildLangTab("🟣 Main.kt", "kotlin", "Main.kt",
            "fun main() {\n    val lang = \"Kotlin Native\"\n    println(\"🟣 Running \" + lang + \" on Android!\")\n}"));
        tabRow.addView(buildLangTab("🐍 script.py", "python", "script.py",
            "# Python 3 Native Script\nprint('🐍 Python 3 Offline Interpreter Ready!')\nfor i in range(3):\n    print('🔥 Python Loop Count:', i)"));
        tabRow.addView(buildLangTab("⚡ native.cpp", "cpp", "native.cpp",
            "#include <iostream>\nusing namespace std;\n\nint main() {\n    int a = 50;\n    int b = 70;\n    int sum = a + b;\n    cout << \"⚡ C++ Engine Sum: \" << sum << endl;\n    return 0;\n}"));
        tabRow.addView(buildLangTab("🖥️ shell.sh", "shell", "shell.sh",
            "echo '🖥️ OxDroid Linux Shell Connected'\nuname -a"));

        tabScroll.addView(tabRow);
        root.addView(tabScroll);

        // 3. MAIN CODE EDITOR + LINE NUMBERS
        ScrollView editorScroll = new ScrollView(this);
        editorScroll.setFillViewport(true);
        editorScroll.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        HorizontalScrollView hScroll = new HorizontalScrollView(this);
        hScroll.setFillViewport(true);

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
        hScroll.addView(editorRow);
        editorScroll.addView(hScroll);
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
        consoleTitle.setText("🖥️ LIVE TERMINAL OUTPUT");
        consoleTitle.setTextColor(Color.parseColor("#3FB950"));
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

    private View buildLangTab(String title, final String lang, final String file, final String defaultCode) {
        TextView tab = new TextView(this);
        tab.setText(title);
        tab.setTextColor(Color.parseColor("#C9D1D9"));
        tab.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        tab.setContentDescription("Open " + file + " in " + lang + " editor");
        tab.setMinimumHeight(dp(48));
        tab.setFocusable(true);
        tab.setPadding(dp(12), dp(6), dp(12), dp(6));
        tab.setBackground(makeRounded(Color.parseColor("#161B22"), dp(8)));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.setMargins(dp(4), 0, dp(4), 0);
        tab.setLayoutParams(lp);

        tab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (workspaceFileLoaded && !saveCurrentFile(false)) return;
                EngineCore.WorkspaceReadResult loaded = EngineCore.loadWorkspaceFile(
                    EditorActivity.this, file, defaultCode
                );
                if (!loaded.success) {
                    Toast.makeText(EditorActivity.this, loaded.error, Toast.LENGTH_LONG).show();
                    return;
                }
                currentFileName = file;
                currentLanguage = lang;
                workspaceFileLoaded = true;
                codeEditor.setEnabled(true);
                fileTitleTv.setText("📄 " + currentFileName);
                langBadgeTv.setText(currentLanguage.toUpperCase());
                codeEditor.setText(loaded.content);
                isDirty = false;
            }
        });
        return tab;
    }

    private void runCurrentCode() {
        if (!workspaceFileLoaded) {
            Toast.makeText(this, "Run blocked: the source file could not be loaded safely.", Toast.LENGTH_LONG).show();
            return;
        }
        if (!saveCurrentFile(true)) return;
        final String code = codeEditor.getText().toString();
        consoleContainer.setVisibility(View.VISIBLE);
        consoleOutputTv.setText("Execution is disabled until an isolated sandbox is available. Saving source...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                final String output = EngineCore.executeCode(EditorActivity.this, currentLanguage, currentFileName, code);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        consoleOutputTv.setText(output);
                    }
                });
            }
        }).start();
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
            if (showFeedback) Toast.makeText(this, "Saved: " + currentFileName, Toast.LENGTH_SHORT).show();
        } else if (showFeedback) {
            Toast.makeText(this, "Save failed. Check workspace storage.", Toast.LENGTH_LONG).show();
        }
        return saved;
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

        // Keywords (Orange/Red)
        Matcher kw = Pattern.compile("\\b(public|private|protected|class|static|void|int|long|double|float|String|boolean|fun|val|var|def|for|in|range|while|if|else|return|import|package|include|using|namespace|std|cout|endl|print|println)\\b").matcher(s);
        while (kw.find()) {
            s.setSpan(new ForegroundColorSpan(Color.parseColor("#FF7B72")), kw.start(), kw.end(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        }

        // Strings (Green)
        Matcher str = Pattern.compile("\"[^\"]*\"|'[^']*'").matcher(s);
        while (str.find()) {
            s.setSpan(new ForegroundColorSpan(Color.parseColor("#A5D6FF")), str.start(), str.end(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        }

        // Comments (Grey)
        Matcher cm = Pattern.compile("//.*|#.*").matcher(s);
        while (cm.find()) {
            s.setSpan(new ForegroundColorSpan(Color.parseColor("#8B949E")), cm.start(), cm.end(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
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
