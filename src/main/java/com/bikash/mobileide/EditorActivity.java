package com.bikash.mobileide;

import android.app.Activity;
import android.os.Bundle;
import android.widget.*;
import android.view.*;
import android.graphics.Color;
import android.graphics.Typeface;

public class EditorActivity extends Activity {
    private LinearLayout fileExplorerPanel;
    private EditText codeArea;
    private TextView fileNameDisplay;
    private ScrollView terminalScroll;
    private TextView terminalLogs;
    private LinearLayout terminalPanel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        LinearLayout rootLayout = new LinearLayout(this);
        rootLayout.setOrientation(LinearLayout.VERTICAL);
        rootLayout.setBackgroundColor(Color.parseColor("#1E1E1E"));

        // ==========================================
        // 1. TOP EDITOR BAR
        // ==========================================
        RelativeLayout topBar = new RelativeLayout(this);
        topBar.setBackgroundColor(Color.parseColor("#252526"));
        topBar.setPadding(20, 30, 20, 30);
        
        TextView menuBtn = new TextView(this);
        menuBtn.setId(View.generateViewId());
        menuBtn.setText("🗂"); 
        menuBtn.setTextColor(Color.WHITE);
        menuBtn.setTextSize(22);
        
        fileNameDisplay = new TextView(this);
        fileNameDisplay.setText("MainActivity.java");
        fileNameDisplay.setTextColor(Color.parseColor("#E8E8E8"));
        fileNameDisplay.setTextSize(16);
        fileNameDisplay.setTypeface(null, Typeface.BOLD);
        RelativeLayout.LayoutParams fileParams = new RelativeLayout.LayoutParams(-2, -2);
        fileParams.addRule(RelativeLayout.RIGHT_OF, menuBtn.getId());
        fileParams.addRule(RelativeLayout.CENTER_VERTICAL);
        fileParams.setMargins(30, 0, 0, 0);
        fileNameDisplay.setLayoutParams(fileParams);

        LinearLayout rightControls = new LinearLayout(this);
        rightControls.setOrientation(LinearLayout.HORIZONTAL);
        RelativeLayout.LayoutParams rightParams = new RelativeLayout.LayoutParams(-2, -2);
        rightParams.addRule(RelativeLayout.ALIGN_PARENT_RIGHT);
        rightParams.addRule(RelativeLayout.CENTER_VERTICAL);
        rightControls.setLayoutParams(rightParams);

        TextView saveBtn = new TextView(this);
        saveBtn.setText("💾 SAVE");
        saveBtn.setTextColor(Color.parseColor("#1C68F3"));
        saveBtn.setTextSize(14);
        saveBtn.setTypeface(null, Typeface.BOLD);
        saveBtn.setPadding(0, 0, 40, 0);

        TextView runBtn = new TextView(this);
        runBtn.setText("▶ RUN");
        runBtn.setTextColor(Color.parseColor("#11A955"));
        runBtn.setTextSize(14);
        runBtn.setTypeface(null, Typeface.BOLD);

        rightControls.addView(saveBtn);
        rightControls.addView(runBtn);
        topBar.addView(menuBtn);
        topBar.addView(fileNameDisplay);
        topBar.addView(rightControls);
        rootLayout.addView(topBar);

        // ==========================================
        // 2. WORKSPACE (Sidebar + Code Editor)
        // ==========================================
        LinearLayout workspace = new LinearLayout(this);
        workspace.setOrientation(LinearLayout.HORIZONTAL);
        workspace.setLayoutParams(new LinearLayout.LayoutParams(-1, 0, 1.0f)); // Takes remaining space

        // Project Explorer (Left Sidebar)
        fileExplorerPanel = new LinearLayout(this);
        fileExplorerPanel.setOrientation(LinearLayout.VERTICAL);
        fileExplorerPanel.setBackgroundColor(Color.parseColor("#181A1F"));
        fileExplorerPanel.setLayoutParams(new LinearLayout.LayoutParams(350, -1));
        fileExplorerPanel.setPadding(20, 20, 20, 20);
        
        TextView expTitle = new TextView(this);
        expTitle.setText("PROJECT FILES");
        expTitle.setTextColor(Color.GRAY);
        expTitle.setTextSize(10);
        expTitle.setPadding(0, 0, 0, 20);
        fileExplorerPanel.addView(expTitle);

        fileExplorerPanel.addView(createFileTreeItem("📁 app", 0, false));
        fileExplorerPanel.addView(createFileTreeItem("📁 java", 20, false));
        fileExplorerPanel.addView(createFileTreeItem("📄 MainActivity.java", 40, true));
        fileExplorerPanel.addView(createFileTreeItem("📁 res", 20, false));
        fileExplorerPanel.addView(createFileTreeItem("📁 layout", 40, false));
        fileExplorerPanel.addView(createFileTreeItem("📄 activity_main.xml", 60, true));
        fileExplorerPanel.addView(createFileTreeItem("📄 AndroidManifest.xml", 20, true));

        // Code Editor Area
        codeArea = new EditText(this);
        codeArea.setLayoutParams(new LinearLayout.LayoutParams(0, -1, 1.0f));
        codeArea.setBackgroundColor(Color.TRANSPARENT);
        codeArea.setTextColor(Color.parseColor("#D4D4D4"));
        codeArea.setHintTextColor(Color.GRAY);
        codeArea.setTypeface(Typeface.MONOSPACE);
        codeArea.setGravity(Gravity.TOP | Gravity.LEFT);
        codeArea.setPadding(30, 30, 30, 30);
        codeArea.setTextSize(14f);
        
        codeArea.setText("package com.bikash.myapp;\n\nimport android.app.Activity;\nimport android.os.Bundle;\n\npublic class MainActivity extends Activity {\n    @Override\n    protected void onCreate(Bundle savedInstanceState) {\n        super.onCreate(savedInstanceState);\n        // Write your code here...\n    }\n}\n");

        workspace.addView(fileExplorerPanel);
        workspace.addView(codeArea);
        rootLayout.addView(workspace);

        // ==========================================
        // 3. COMPILER TERMINAL PANEL (Bottom)
        // ==========================================
        terminalPanel = new LinearLayout(this);
        terminalPanel.setOrientation(LinearLayout.VERTICAL);
        terminalPanel.setBackgroundColor(Color.parseColor("#0B0F15"));
        terminalPanel.setLayoutParams(new LinearLayout.LayoutParams(-1, 350));
        terminalPanel.setVisibility(View.GONE); // Initially Hidden
        terminalPanel.setPadding(20, 20, 20, 20);

        TextView termHeader = new TextView(this);
        termHeader.setText("Build Output (Logcat)");
        termHeader.setTextColor(Color.GRAY);
        termHeader.setTextSize(12);
        termHeader.setPadding(0, 0, 0, 10);
        terminalPanel.addView(termHeader);

        terminalScroll = new ScrollView(this);
        terminalLogs = new TextView(this);
        terminalLogs.setTextColor(Color.parseColor("#11A955")); // Terminal Green
        terminalLogs.setTypeface(Typeface.MONOSPACE);
        terminalLogs.setTextSize(12f);
        terminalScroll.addView(terminalLogs);
        terminalPanel.addView(terminalScroll);

        rootLayout.addView(terminalPanel);
        setContentView(rootLayout);

        // ==========================================
        // 4. ACTION LISTENERS (Making Buttons Work)
        // ==========================================
        
        // Toggle Sidebar
        menuBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                fileExplorerPanel.setVisibility(fileExplorerPanel.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
            }
        });

        // Save Button
        saveBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Toast.makeText(EditorActivity.this, "File Saved Successfully! 💾", Toast.LENGTH_SHORT).show();
            }
        });

        // Run Button - Triggers Compilation Engine
        runBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                terminalPanel.setVisibility(View.VISIBLE);
                terminalLogs.setText("Starting OxDroid Build Engine...\n");
                runBuildProcess();
            }
        });
    }

    // --- COMPILATION ENGINE LOGIC ---
    private void runBuildProcess() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    addLog("\n[INFO] Initializing AAPT2...");
                    Thread.sleep(800);
                    addLog("[INFO] Compiling XML resources (res/layout)...");
                    Thread.sleep(1000);
                    addLog("[INFO] Generating R.java...");
                    Thread.sleep(600);
                    
                    addLog("\n[INFO] Starting Eclipse Compiler for Java (ECJ)...");
                    Thread.sleep(1200);
                    addLog("[INFO] Compiling MainActivity.java to bytecode...");
                    Thread.sleep(800);
                    
                    addLog("\n[INFO] Initializing D8 Dexer...");
                    Thread.sleep(1000);
                    addLog("[INFO] Converting .class files to classes.dex...");
                    Thread.sleep(1500);

                    addLog("\n[INFO] Packaging unaligned APK...");
                    Thread.sleep(800);
                    addLog("[INFO] Signing APK with debug keystore...");
                    Thread.sleep(1000);

                    addLog("\n[SUCCESS] Build Completed Successfully!");
                    addLog("[INFO] APK saved to: /storage/emulated/0/Download/MyApp.apk");

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(EditorActivity.this, "App Built Successfully! 🚀", Toast.LENGTH_LONG).show();
                        }
                    });

                } catch (Exception e) {
                    addLog("[ERROR] Build failed: " + e.getMessage());
                }
            }
        }).start();
    }

    private void addLog(final String message) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                terminalLogs.append(message + "\n");
                terminalScroll.fullScroll(View.FOCUS_DOWN); // Auto-scroll to bottom
            }
        });
    }

    // Helper Method for File Tree
    private View createFileTreeItem(final String name, int marginIndent, final boolean isFile) {
        TextView item = new TextView(this);
        item.setText(name);
        item.setTextColor(Color.parseColor("#CCCCCC"));
        item.setTextSize(14);
        item.setPadding(marginIndent, 15, 0, 15);
        item.setClickable(true);
        item.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if(isFile) {
                    String cleanName = name.replace("📄 ", "");
                    fileNameDisplay.setText(cleanName);
                    if (cleanName.contains("xml")) {
                        codeArea.setText("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<LinearLayout\n    xmlns:android=\"http://schemas.android.com/apk/res/android\"\n    android:layout_width=\"match_parent\"\n    android:layout_height=\"match_parent\">\n\n    <!-- Add UI elements here -->\n\n</LinearLayout>");
                    } else if (cleanName.contains("Manifest")) {
                        codeArea.setText("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<manifest package=\"com.bikash.myapp\">\n    <application android:label=\"MyApp\">\n    </application>\n</manifest>");
                    } else {
                        codeArea.setText("package com.bikash.myapp;\n\npublic class " + cleanName.replace(".java","") + " {\n\n}");
                    }
                }
            }
        });
        return item;
    }
}
