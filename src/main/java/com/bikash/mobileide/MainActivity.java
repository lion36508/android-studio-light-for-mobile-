package com.bikash.mobileide;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {

    private LinearLayout projectsContainer;
    private LinearLayout searchBarContainer;
    private EditText searchInput;
    private List<ProjectItem> projectList = new ArrayList<ProjectItem>();
    private SharedPreferences prefs;

    private static class ProjectItem {
        String name;
        String subtitle;
        String date;
        String icon;
        int badgeColor;
        String lang;
        String fileName;
        String starterCode;

        ProjectItem(String name, String subtitle, String date, String icon, int badgeColor, String lang, String fileName, String starterCode) {
            this.name = name;
            this.subtitle = subtitle;
            this.date = date;
            this.icon = icon;
            this.badgeColor = badgeColor;
            this.lang = lang;
            this.fileName = fileName;
            this.starterCode = starterCode;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("oxdroid_prefs", MODE_PRIVATE);
        initDefaultProjects();
        setContentView(buildOriginalUI());
    }

    private void initDefaultProjects() {
        projectList.clear();
        projectList.add(new ProjectItem(
            "Androidify",
            "Kotlin • Jetpack Compose",
            "Today",
            "🤖",
            Color.parseColor("#2EA043"),
            "kotlin",
            "Androidify.kt",
            "// Androidify - Kotlin Jetpack Compose\npackage com.bikash.androidify\n\nfun main() {\n    val appName = \"Androidify Compose\"\n    val version = 2\n    println(\"🚀 Launching \" + appName)\n    println(\"✨ Compose UI Engine v\" + version + \" Ready!\")\n}"
        ));
        projectList.add(new ProjectItem(
            "ZenBuild",
            "C++ • Native Project",
            "Yesterday",
            "⚙️",
            Color.parseColor("#8957E5"),
            "cpp",
            "ZenBuild.cpp",
            "// ZenBuild - Native C++ ARM64 Engine\n#include <iostream>\nusing namespace std;\n\nint main() {\n    int threads = 8;\n    int speed = threads * 125;\n    cout << \"⚡ ZenBuild C++ Native Engine\" << endl;\n    cout << \"🔥 Compiling Speed: \" << speed << \" MB/s\" << endl;\n    return 0;\n}"
        ));
        projectList.add(new ProjectItem(
            "MyApp",
            "Java • Android SDK",
            "2 days ago",
            "📱",
            Color.parseColor("#00ACC1"),
            "java",
            "MainActivity.java",
            "// MyApp - Native Android SDK Project\npackage com.bikash.myapp;\n\npublic class MainActivity {\n    public static void main(String[] args) {\n        String status = \"OxDroid Java SDK Active!\";\n        int apiLevel = 34;\n        System.out.println(\"📱 \" + status);\n        System.out.println(\"🛠️ Target Android API: \" + apiLevel);\n    }\n}"
        ));
        projectList.add(new ProjectItem(
            "PyCore AI",
            "Python 3 • Offline Script",
            "Today",
            "🐍",
            Color.parseColor("#D29922"),
            "python",
            "pycore_ai.py",
            "# PyCore AI - Native Python 3 Engine\nmodel = 'OxDroid-Mini-v1'\nprint('🐍 Python 3 Offline Engine Connected!')\nprint('🤖 Running AI Model:', model)\nfor step in range(3):\n    print('⚡ Training Epoch:', step)\nprint('✅ Model Ready on Android!')"
        ));
    }

    private View buildOriginalUI() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0B0E14"));
        root.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        // 1. TOP APP BAR (Hamburger, Title+Subtitle, Search, 3-Dots)
        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setGravity(Gravity.CENTER_VERTICAL);
        topBar.setPadding(dp(16), dp(14), dp(16), dp(10));

        TextView menuBtn = new TextView(this);
        menuBtn.setText("☰");
        menuBtn.setTextColor(Color.WHITE);
        menuBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24);
        menuBtn.setPadding(0, 0, dp(16), 0);
        addTouchEffect(menuBtn);
        menuBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showDrawerMenu();
            }
        });
        topBar.addView(menuBtn);

        LinearLayout titleCol = new LinearLayout(this);
        titleCol.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams titleLp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        titleCol.setLayoutParams(titleLp);

        TextView titleTv = new TextView(this);
        titleTv.setText("OxDroid Studio");
        titleTv.setTextColor(Color.WHITE);
        titleTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        titleTv.setTypeface(null, Typeface.BOLD);

        TextView subTitleTv = new TextView(this);
        subTitleTv.setText("Mobile Development Workspace");
        subTitleTv.setTextColor(Color.parseColor("#8B949E"));
        subTitleTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);

        titleCol.addView(titleTv);
        titleCol.addView(subTitleTv);
        topBar.addView(titleCol);

        TextView searchBtn = new TextView(this);
        searchBtn.setText("🔍");
        searchBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        searchBtn.setPadding(dp(10), dp(4), dp(10), dp(4));
        addTouchEffect(searchBtn);
        searchBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (searchBarContainer.getVisibility() == View.GONE) {
                    searchBarContainer.setVisibility(View.VISIBLE);
                    searchInput.requestFocus();
                } else {
                    searchInput.setText("");
                    searchBarContainer.setVisibility(View.GONE);
                }
            }
        });
        topBar.addView(searchBtn);

        TextView moreBtn = new TextView(this);
        moreBtn.setText("⋮");
        moreBtn.setTextColor(Color.WHITE);
        moreBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24);
        moreBtn.setPadding(dp(10), 0, dp(4), 0);
        addTouchEffect(moreBtn);
        moreBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showTopMoreMenu();
            }
        });
        topBar.addView(moreBtn);

        root.addView(topBar);

        // COLLAPSIBLE LIVE SEARCH BAR
        searchBarContainer = new LinearLayout(this);
        searchBarContainer.setOrientation(LinearLayout.HORIZONTAL);
        searchBarContainer.setPadding(dp(16), dp(4), dp(16), dp(8));
        searchBarContainer.setVisibility(View.GONE);

        searchInput = new EditText(this);
        searchInput.setHint("🔎 Search projects by name or language (Kotlin, C++, Python, Java)...");
        searchInput.setHintTextColor(Color.parseColor("#6E7681"));
        searchInput.setTextColor(Color.WHITE);
        searchInput.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        searchInput.setPadding(dp(14), dp(10), dp(14), dp(10));
        searchInput.setBackground(makeRoundedBg(Color.parseColor("#161B22"), dp(12), Color.parseColor("#30363D")));
        searchInput.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        searchInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                refreshProjectList(s.toString());
            }
            @Override
            public void afterTextChanged(Editable s) {}
        });
        searchBarContainer.addView(searchInput);
        root.addView(searchBarContainer);

        // MAIN SCROLLABLE BODY
        ScrollView scrollView = new ScrollView(this);
        LinearLayout.LayoutParams scrollLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f);
        scrollView.setLayoutParams(scrollLp);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(16), dp(6), dp(16), dp(20));

        // 2. HERO BANNER (res/drawable/hero_banner.png)
        LinearLayout bannerWrapper = new LinearLayout(this);
        bannerWrapper.setOrientation(LinearLayout.VERTICAL);
        bannerWrapper.setBackground(makeRoundedBg(Color.parseColor("#101935"), dp(18), Color.parseColor("#1F3264")));
        LinearLayout.LayoutParams bannerLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(175));
        bannerLp.setMargins(0, 0, 0, dp(18));
        bannerWrapper.setLayoutParams(bannerLp);

        ImageView bannerImg = new ImageView(this);
        bannerImg.setImageResource(R.drawable.hero_banner);
        bannerImg.setScaleType(ImageView.ScaleType.FIT_XY);
        bannerImg.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        addTouchEffect(bannerImg);
        bannerImg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openEditorWithProject(projectList.get(0));
            }
        });
        bannerWrapper.addView(bannerImg);
        content.addView(bannerWrapper);

        // 3. FOUR QUICK ACTION CARDS ROW
        LinearLayout cardsRow = new LinearLayout(this);
        cardsRow.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams rowLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        rowLp.setMargins(0, 0, 0, dp(22));
        cardsRow.setLayoutParams(rowLp);

        // Card 1: New Project (Blue)
        cardsRow.addView(buildActionCard(
            "➕", "New\nProject", "Start a new\napp",
            Color.parseColor("#1A6EF5"), Color.parseColor("#B8D4FF"),
            new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showNewProjectDialog();
                }
            }
        ));

        // Card 2: Open Project (Dark Grey)
        cardsRow.addView(buildActionCard(
            "📁", "Open\nProject", "Browse\nstorage",
            Color.parseColor("#21262D"), Color.parseColor("#8B949E"),
            new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showOpenProjectDialog();
                }
            }
        ));

        // Card 3: Clone Git (Purple)
        cardsRow.addView(buildActionCard(
            "🔗", "Git (Soon)", "Git client not ready",
            Color.parseColor("#6E2CF2"), Color.parseColor("#DCBFFF"),
            new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showCloneGitDialog();
                }
            }
        ));

        // Card 4: Samples (Green)
        cardsRow.addView(buildActionCard(
            "🎛️", "Samples", "Template\napps",
            Color.parseColor("#196C3E"), Color.parseColor("#AEEBC6"),
            new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showSamplesDialog();
                }
            }
        ));

        content.addView(cardsRow);

        // 4. RECENT PROJECTS HEADER (Recent Projects + See All >)
        LinearLayout recentHeader = new LinearLayout(this);
        recentHeader.setOrientation(LinearLayout.HORIZONTAL);
        recentHeader.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams rhLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        rhLp.setMargins(0, 0, 0, dp(12));
        recentHeader.setLayoutParams(rhLp);

        TextView recentTitle = new TextView(this);
        recentTitle.setText("Workspace & Starter Samples");
        recentTitle.setTextColor(Color.WHITE);
        recentTitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 19);
        recentTitle.setTypeface(null, Typeface.BOLD);
        recentTitle.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        recentHeader.addView(recentTitle);

        TextView seeAllBtn = new TextView(this);
        seeAllBtn.setText("See All ❯");
        seeAllBtn.setTextColor(Color.parseColor("#388BFD"));
        seeAllBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        seeAllBtn.setTypeface(null, Typeface.BOLD);
        addTouchEffect(seeAllBtn);
        seeAllBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                searchBarContainer.setVisibility(View.VISIBLE);
                searchInput.setText("");
                Toast.makeText(MainActivity.this, "Showing all " + projectList.size() + " workspace projects", Toast.LENGTH_SHORT).show();
            }
        });
        recentHeader.addView(seeAllBtn);
        content.addView(recentHeader);

        // 5. RECENT PROJECTS DYNAMIC CONTAINER
        projectsContainer = new LinearLayout(this);
        projectsContainer.setOrientation(LinearLayout.VERTICAL);
        content.addView(projectsContainer);
        refreshProjectList("");

        scrollView.addView(content);
        root.addView(scrollView);

        // 6. BOTTOM NAVIGATION BAR (Home, Projects, Editor, Settings)
        LinearLayout bottomNav = new LinearLayout(this);
        bottomNav.setOrientation(LinearLayout.HORIZONTAL);
        bottomNav.setGravity(Gravity.CENTER);
        bottomNav.setBackgroundColor(Color.parseColor("#0D1117"));
        bottomNav.setPadding(0, dp(8), 0, dp(8));

        bottomNav.addView(buildNavTab("🏠", "Home", true, new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                searchBarContainer.setVisibility(View.GONE);
                refreshProjectList("");
                Toast.makeText(MainActivity.this, "🏠 OxDroid Home", Toast.LENGTH_SHORT).show();
            }
        }));
        bottomNav.addView(buildNavTab("📁", "Projects", false, new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showOpenProjectDialog();
            }
        }));
        bottomNav.addView(buildNavTab("</>", "Editor", false, new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openEditorWithProject(projectList.get(0));
            }
        }));
        bottomNav.addView(buildNavTab("⚙️", "Settings", false, new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showSettingsDialog();
            }
        }));

        root.addView(bottomNav);
        return root;
    }

    private View buildActionCard(String icon, String title, String sub, int bgColor, int subColor, View.OnClickListener listener) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setGravity(Gravity.CENTER);
        card.setPadding(dp(6), dp(14), dp(6), dp(14));
        card.setBackground(makeRoundedBg(bgColor, dp(16), Color.parseColor("#30363D")));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(122), 1f);
        lp.setMargins(dp(4), 0, dp(4), 0);
        card.setLayoutParams(lp);

        TextView iconTv = new TextView(this);
        iconTv.setText(icon);
        iconTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
        iconTv.setGravity(Gravity.CENTER);

        TextView titleTv = new TextView(this);
        titleTv.setText(title);
        titleTv.setTextColor(Color.WHITE);
        titleTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        titleTv.setTypeface(null, Typeface.BOLD);
        titleTv.setGravity(Gravity.CENTER);
        titleTv.setPadding(0, dp(6), 0, dp(4));

        TextView subTv = new TextView(this);
        subTv.setText(sub);
        subTv.setTextColor(subColor);
        subTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 10);
        subTv.setGravity(Gravity.CENTER);

        card.addView(iconTv);
        card.addView(titleTv);
        card.addView(subTv);

        addTouchEffect(card);
        card.setOnClickListener(listener);
        return card;
    }

    private void refreshProjectList(String query) {
        projectsContainer.removeAllViews();
        String q = query.trim().toLowerCase();

        for (int i = 0; i < projectList.size(); i++) {
            final ProjectItem item = projectList.get(i);
            if (!q.isEmpty() && !item.name.toLowerCase().contains(q) && !item.subtitle.toLowerCase().contains(q)) {
                continue;
            }

            LinearLayout card = new LinearLayout(this);
            card.setOrientation(LinearLayout.HORIZONTAL);
            card.setGravity(Gravity.CENTER_VERTICAL);
            card.setPadding(dp(14), dp(14), dp(14), dp(14));
            card.setBackground(makeRoundedBg(Color.parseColor("#161B22"), dp(14), Color.parseColor("#21262D")));

            LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            cardLp.setMargins(0, 0, 0, dp(12));
            card.setLayoutParams(cardLp);

            // Colored Icon Box
            TextView iconBox = new TextView(this);
            iconBox.setText(item.icon);
            iconBox.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
            iconBox.setGravity(Gravity.CENTER);
            iconBox.setBackground(makeRoundedBg(item.badgeColor, dp(12), item.badgeColor));
            LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(dp(48), dp(48));
            iconLp.setMargins(0, 0, dp(14), 0);
            iconBox.setLayoutParams(iconLp);
            card.addView(iconBox);

            // Project Name + Language Info
            LinearLayout infoCol = new LinearLayout(this);
            infoCol.setOrientation(LinearLayout.VERTICAL);
            infoCol.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

            TextView nameTv = new TextView(this);
            nameTv.setText(item.name);
            nameTv.setTextColor(Color.WHITE);
            nameTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
            nameTv.setTypeface(null, Typeface.BOLD);

            TextView subTv = new TextView(this);
            subTv.setText(item.subtitle);
            subTv.setTextColor(Color.parseColor("#8B949E"));
            subTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
            subTv.setPadding(0, dp(3), 0, 0);

            infoCol.addView(nameTv);
            infoCol.addView(subTv);
            card.addView(infoCol);

            // Right Side: 3-dots + Date
            LinearLayout rightCol = new LinearLayout(this);
            rightCol.setOrientation(LinearLayout.VERTICAL);
            rightCol.setGravity(Gravity.END);

            TextView menuDots = new TextView(this);
            menuDots.setText("⋮");
            menuDots.setTextColor(Color.parseColor("#8B949E"));
            menuDots.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
            menuDots.setPadding(dp(12), 0, dp(4), dp(4));
            menuDots.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    showProjectItemMenu(item);
                }
            });

            TextView dateTv = new TextView(this);
            dateTv.setText(item.date);
            dateTv.setTextColor(Color.parseColor("#6E7681"));
            dateTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);

            rightCol.addView(menuDots);
            rightCol.addView(dateTv);
            card.addView(rightCol);

            addTouchEffect(card);
            card.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    openEditorWithProject(item);
                }
            });

            projectsContainer.addView(card);
        }
    }

    private View buildNavTab(String icon, String label, boolean active, View.OnClickListener listener) {
        LinearLayout tab = new LinearLayout(this);
        tab.setOrientation(LinearLayout.VERTICAL);
        tab.setGravity(Gravity.CENTER);
        tab.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView iconTv = new TextView(this);
        iconTv.setText(icon);
        iconTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        iconTv.setTextColor(active ? Color.parseColor("#388BFD") : Color.parseColor("#8B949E"));
        iconTv.setGravity(Gravity.CENTER);

        TextView labelTv = new TextView(this);
        labelTv.setText(label);
        labelTv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        labelTv.setTextColor(active ? Color.parseColor("#388BFD") : Color.parseColor("#8B949E"));
        labelTv.setPadding(0, dp(2), 0, 0);
        labelTv.setGravity(Gravity.CENTER);

        tab.addView(iconTv);
        tab.addView(labelTv);
        addTouchEffect(tab);
        tab.setOnClickListener(listener);
        return tab;
    }

    private void openEditorWithProject(ProjectItem item) {
        EngineCore.loadFile(this, item.fileName, item.starterCode);
        Intent intent = new Intent(this, EditorActivity.class);
        intent.putExtra("project_name", item.name);
        intent.putExtra("file_name", item.fileName);
        intent.putExtra("language", item.lang);
        intent.putExtra("starter_code", item.starterCode);
        startActivity(intent);
    }

    // DIALOG 1: NEW PROJECT CREATOR (Java, Kotlin, Python, C++)
    private void showNewProjectDialog() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(20), dp(16), dp(20), dp(8));

        final EditText nameEt = new EditText(this);
        nameEt.setHint("Enter Project Name (e.g. SuperApp)");
        box.addView(nameEt);

        final String[] langs = {
            "☕ Android App (Java)",
            "▶ Video App Starter (Java)",
            "🐍 Python 3 (Native Script)",
            "⚡ C++ (ARM64 Native Binary)"
        };

        new AlertDialog.Builder(this)
            .setTitle("➕ Create New Project")
            .setView(box)
            .setItems(langs, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    String rawName = nameEt.getText().toString().trim();
                    if (rawName.isEmpty()) rawName = "OxProject" + (projectList.size() + 1);

                    ProjectItem newItem;
                    if (which == 0) {
                        newItem = new ProjectItem(rawName, "Java • Android SDK", "Just now", "☕",
                            Color.parseColor("#00ACC1"), "java", rawName + ".java",
                            "public class " + rawName.replace(" ", "") + " {\n    public static void main(String[] args) {\n        System.out.println(\"🚀 Welcome to " + rawName + " (Java)!\");\n    }\n}");
                    } else if (which == 1) {
                        newItem = new ProjectItem(rawName, "Video App • Android Java", "Just now", "▶",
                            Color.parseColor("#E53935"), "java", rawName + ".java",
                            "// Video App Starter\n// Use the generated Android project tree and connect your authorized backend API.\n");
                    } else if (which == 2) {
                        newItem = new ProjectItem(rawName, "Python 3 • Native Script", "Just now", "🐍",
                            Color.parseColor("#D29922"), "python", rawName.toLowerCase() + ".py",
                            "# " + rawName + " - Python 3\nname = '" + rawName + "'\nprint('🐍 Running Python Project:', name)\nfor i in range(3):\n    print('Step:', i)");
                    } else {
                        newItem = new ProjectItem(rawName, "C++ • Native Project", "Just now", "⚡",
                            Color.parseColor("#8957E5"), "cpp", rawName.toLowerCase() + ".cpp",
                            "#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << \"⚡ C++ Project " + rawName + " Running!\" << endl;\n    return 0;\n}");
                    }

                    if (which == 0 || which == 1) {
                        String generatedFile = ProjectScaffold.create(MainActivity.this, rawName, which == 1);
                        newItem.fileName = generatedFile;
                        newItem.starterCode = EngineCore.loadFile(MainActivity.this, generatedFile, newItem.starterCode);
                    }
                    projectList.add(0, newItem);
                    refreshProjectList("");
                    openEditorWithProject(newItem);
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    // DIALOG 2: OPEN PROJECT / STORAGE EXPLORER
    private void showOpenProjectDialog() {
        File ws = EngineCore.getWorkspaceDir(this);
        File[] files = ws.listFiles();
        final List<String> names = new ArrayList<String>();
        for (ProjectItem p : projectList) {
            names.add("📦 " + p.name + " (" + p.fileName + ")");
        }
        if (files != null) {
            for (File f : files) {
                String label = "📄 " + f.getName();
                if (!names.contains(label)) names.add(label);
            }
        }

        String[] arr = names.toArray(new String[0]);
        new AlertDialog.Builder(this)
            .setTitle("📁 Open Workspace Project")
            .setItems(arr, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if (which < projectList.size()) {
                        openEditorWithProject(projectList.get(which));
                    } else {
                        String picked = names.get(which).replace("📄 ", "").trim();
                        String lang = picked.endsWith(".py") ? "python" : picked.endsWith(".cpp") ? "cpp" : picked.endsWith(".kt") ? "kotlin" : "java";
                        String content = EngineCore.loadFile(MainActivity.this, picked, "// File: " + picked);
                        ProjectItem temp = new ProjectItem(picked, lang.toUpperCase() + " • Workspace File", "Opened", "📄",
                            Color.parseColor("#1A6EF5"), lang, picked, content);
                        openEditorWithProject(temp);
                    }
                }
            })
            .setNegativeButton("Close", null)
            .show();
    }

    // DIALOG 3: GIT CLONE (not implemented yet; never report a fake clone)
    private void showCloneGitDialog() {
        new AlertDialog.Builder(this)
            .setTitle("Git Clone Not Available Yet")
            .setMessage("This version does not include a Git transport/client or credential handling. No repository has been downloaded and no files have been created. Git clone will be enabled after a real client is integrated and tested.")
            .setPositiveButton("OK", null)
            .show();
    }

    // DIALOG 4: SAMPLES & TEMPLATES
    private void showSamplesDialog() {
        final String[] samples = {
            "🐍 Python: Smart Calculator & Loop Engine",
            "⚡ C++: High-Speed Matrix Benchmark",
            "🟣 Kotlin: Android Compose Counter App",
            "☕ Java: Android SDK Toast & Lifecycle Demo",
            "🖥️ Linux Shell: Android CPU & RAM Inspector"
        };

        new AlertDialog.Builder(this)
            .setTitle("🎛️ Built-in Sample Templates")
            .setItems(samples, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    ProjectItem sample;
                    if (which == 0) {
                        sample = new ProjectItem("PyCalculator", "Python 3 • Sample", "Template", "🐍",
                            Color.parseColor("#D29922"), "python", "sample_calc.py",
                            "# Python Smart Calculator Sample\na = 120\nb = 45\ntotal = a + b\nmult = a * 2\nprint('🐍 Python Native Calculator')\nprint('Addition (120 + 45):', total)\nprint('Multiply (120 * 2):', mult)\nfor i in range(4):\n    print('Iteration count:', i)");
                    } else if (which == 1) {
                        sample = new ProjectItem("CppBenchmark", "C++ • Sample", "Template", "⚡",
                            Color.parseColor("#8957E5"), "cpp", "benchmark.cpp",
                            "#include <iostream>\nusing namespace std;\n\nint main() {\n    int cores = 8;\n    int score = cores * 950;\n    cout << \"⚡ ARM64 C++ Benchmark Started\" << endl;\n    cout << \"🏆 Multi-Core Score: \" << score << \" pts\" << endl;\n    return 0;\n}");
                    } else if (which == 2) {
                        sample = new ProjectItem("ComposeCounter", "Kotlin • Sample", "Template", "🟣",
                            Color.parseColor("#2EA043"), "kotlin", "Counter.kt",
                            "fun main() {\n    var count = 10\n    println(\"🟣 Jetpack Compose State Demo\")\n    println(\"Initial Counter Value: \" + count)\n}");
                    } else if (which == 3) {
                        sample = new ProjectItem("AndroidSDKDemo", "Java • Sample", "Template", "☕",
                            Color.parseColor("#00ACC1"), "java", "SdkDemo.java",
                            "public class SdkDemo {\n    public static void main(String[] args) {\n        System.out.println(\"☕ Android SDK Lifecycle Initialized\");\n        System.out.println(\"✅ Activity onCreate() -> onResume() Complete!\");\n    }\n}");
                    } else {
                        sample = new ProjectItem("SystemInspector", "Shell • Linux", "Template", "🖥️",
                            Color.parseColor("#196C3E"), "shell", "sysinfo.sh",
                            "echo '=== OxDroid Linux Kernel Info ==='\nuname -a\necho '=== Android Architecture ==='\ngetprop ro.product.cpu.abi\necho '=== Memory Status ==='\nfree -m 2>/dev/null || cat /proc/meminfo | head -n 5");
                    }
                    openEditorWithProject(sample);
                }
            })
            .setNegativeButton("Close", null)
            .show();
    }

    // PROJECT 3-DOTS CONTEXT MENU
    private void showProjectItemMenu(final ProjectItem item) {
        String[] options = {
            "📝 Open in Code Editor",
            "▶️ Run Project Directly",
            "🗑️ Remove from List"
        };
        new AlertDialog.Builder(this)
            .setTitle(item.icon + " " + item.name)
            .setItems(options, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if (which == 0) {
                        openEditorWithProject(item);
                    } else if (which == 1) {
                        String out = EngineCore.executeCode(MainActivity.this, item.lang, item.fileName, item.starterCode);
                        new AlertDialog.Builder(MainActivity.this)
                            .setTitle("▶️ Execution Output: " + item.name)
                            .setMessage(out)
                            .setPositiveButton("Open Editor", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface d, int w) {
                                    openEditorWithProject(item);
                                }
                            })
                            .setNegativeButton("Close", null)
                            .show();
                    } else {
                        projectList.remove(item);
                        refreshProjectList("");
                        Toast.makeText(MainActivity.this, "Removed " + item.name, Toast.LENGTH_SHORT).show();
                    }
                }
            })
            .show();
    }

    // DRAWER MENU (HAMBURGER ☰)
    private void showDrawerMenu() {
        String[] items = {
            "🧰 Runtime & Build Toolchain Status",
            "🖥️ Open Interactive Linux Shell Terminal",
            "🛡️ Ironclad Open-Source Guarantee",
            "⚙️ IDE Settings & Preferences"
        };
        showMasterNeonDialog(items, "ui/green_neon.png", true, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if (which == 0) {
                        String abi = EngineCore.runShellCommand("getprop ro.product.cpu.abi");
                        new AlertDialog.Builder(MainActivity.this)
                            .setTitle("Runtime & Build Toolchain Status")
                            .setMessage("CPU Architecture: " + abi + "\n\n" +
                                "On-device Android APK build: NOT AVAILABLE yet.\n" +
                                "Python/C++/Java/Kotlin fallback: limited evaluator, not a full language compiler/runtime.\n" +
                                "Python/C++ daemon: used only if a real local service responds.\n" +
                                "Shell: Android /system/bin/sh is available.")
                            .setPositiveButton("Awesome", null)
                            .show();
                    } else if (which == 1) {
                        ProjectItem term = new ProjectItem("LinuxTerminal", "Bash • Native Shell", "Live", "🖥️",
                            Color.parseColor("#196C3E"), "shell", "terminal.sh",
                            "echo '=== [ OxDroid 10MB Linux Engine Active ] ==='\nuname -a\necho ' Core: ARM64 RootFS + Toybox Ready'\nls -la");
                        openEditorWithProject(term);
                    } else if (which == 2) {
                        new AlertDialog.Builder(MainActivity.this)
                            .setTitle("🛡️ Ironclad Open-Source Promise")
                            .setMessage("OxDroid Studio belongs to students and developers worldwide.\n\n100% Free & Open Source Forever. It will NEVER be sold to any company.")
                            .setPositiveButton("OK", null)
                            .show();
                    } else {
                        showSettingsDialog();
                    }
                }
            });
    }

    // TOP RIGHT 3-DOTS MENU
    private void showTopMoreMenu() {
        String[] opts = {
            "🔄 Refresh Workspace",
            "🔍 Toggle Search Bar",
            "🧹 Clear Build Cache",
            "⚙️ Open Settings"
        };
        showMasterNeonDialog(opts, "ui/blue_neon.png", false, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if (which == 0) {
                        refreshProjectList("");
                        Toast.makeText(MainActivity.this, "✅ Workspace refreshed!", Toast.LENGTH_SHORT).show();
                    } else if (which == 1) {
                        searchBarContainer.setVisibility(
                            searchBarContainer.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
                    } else if (which == 2) {
                        new AlertDialog.Builder(MainActivity.this)
                            .setTitle("Build Cache Not Connected")
                            .setMessage("The current app does not have an on-device build cache connected to this action. No files were deleted.")
                            .setPositiveButton("OK", null)
                            .show();
                    } else {
                        showSettingsDialog();
                    }
                }
            });
    }

    // SETTINGS DIALOG
    private void showSettingsDialog() {
        final int currentFont = prefs.getInt("font_size", 15);
        final boolean autoSave = prefs.getBoolean("auto_save", true);

        String[] settingsItems = {
            "🔤 Editor Font Size (Current: " + currentFont + "sp)",
            "💾 Auto-Save Code on Run (" + (autoSave ? "ON" : "OFF") + ")",
            "🧰 Check Android Build Toolchain Status",
            "🔄 Reset Default Sample Projects"
        };

        showMasterNeonDialog(settingsItems, "ui/blue_neon.png", false, new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    if (which == 0) {
                        int nextSize = (currentFont == 15) ? 18 : (currentFont == 18) ? 13 : 15;
                        prefs.edit().putInt("font_size", nextSize).apply();
                        Toast.makeText(MainActivity.this, "Font size set to " + nextSize + "sp", Toast.LENGTH_SHORT).show();
                    } else if (which == 1) {
                        prefs.edit().putBoolean("auto_save", !autoSave).apply();
                        Toast.makeText(MainActivity.this, "Auto-Save set to " + (!autoSave ? "ON" : "OFF"), Toast.LENGTH_SHORT).show();
                    } else if (which == 2) {
                        new AlertDialog.Builder(MainActivity.this)
                            .setTitle("Android Build Toolchain Status")
                            .setMessage(EngineCore.buildToolStatus(MainActivity.this))
                            .setPositiveButton("OK", null)
                            .show();
                    } else {
                        initDefaultProjects();
                        refreshProjectList("");
                        Toast.makeText(MainActivity.this, "✅ Default projects restored!", Toast.LENGTH_SHORT).show();
                    }
                }
            });
    }

    private GradientDrawable makeRoundedBg(int fillColor, int radiusPx, int strokeColor) {
        GradientDrawable gd = new GradientDrawable();
        gd.setColor(fillColor);
        gd.setCornerRadius(radiusPx);
        gd.setStroke(dp(1), strokeColor);
        return gd;
    }

    private void addTouchEffect(final View view) {
        view.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    v.setAlpha(0.7f);
                } else if (event.getAction() == MotionEvent.ACTION_UP || event.getAction() == MotionEvent.ACTION_CANCEL) {
                    v.setAlpha(1.0f);
                }
                return false;
            }
        });
    }

    private int dp(int val) {
        return (int) TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, val, getResources().getDisplayMetrics());
    }

    private void showMasterNeonDialog(String[] items, String assetName, final boolean hasBuiltInText, final android.content.DialogInterface.OnClickListener listener) {
        try {
            final android.app.Dialog dialog = new android.app.Dialog(this);
            dialog.requestWindowFeature(android.view.Window.FEATURE_NO_TITLE);
            if (dialog.getWindow() != null) {
                dialog.getWindow().setBackgroundDrawable(new android.graphics.drawable.ColorDrawable(0));
            }

            android.widget.LinearLayout base = new android.widget.LinearLayout(this);
            base.setOrientation(android.widget.LinearLayout.VERTICAL);

            int imgW = 1536, imgH = 1024;
            try {
                java.io.InputStream is = getAssets().open(assetName);
                android.graphics.Bitmap bmp = android.graphics.BitmapFactory.decodeStream(is);
                is.close();
                if (bmp != null) {
                    imgW = bmp.getWidth();
                    imgH = bmp.getHeight();
                    base.setBackground(new android.graphics.drawable.BitmapDrawable(getResources(), bmp));
                }
            } catch (Exception e) {
                base.setBackgroundColor(android.graphics.Color.parseColor("#121829"));
            }

            int w = (int) (getResources().getDisplayMetrics().widthPixels * 0.96f);
            int h = (int) (w * ((float) imgH / (float) imgW));

            if (hasBuiltInText) {
                base.setWeightSum(5f);
                int padH = (int)(w * 0.08f);
                int padV = (int)(h * 0.09f);
                base.setPadding(padH, padV, padH, padV);

                android.view.View headerSlot = new android.view.View(this);
                headerSlot.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                    android.view.ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
                base.addView(headerSlot);

                for (int i = 0; i < items.length && i < 4; i++) {
                    final int idx = i;
                    android.view.View touchZone = new android.view.View(this);
                    touchZone.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                        android.view.ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
                    touchZone.setOnClickListener(new android.view.View.OnClickListener() {
                        @Override
                        public void onClick(android.view.View v) {
                            try { dialog.dismiss(); } catch (Exception ignored) {}
                            if (listener != null) {
                                listener.onClick(dialog, idx);
                            }
                        }
                    });
                    base.addView(touchZone);
                }
            } else {
                base.setWeightSum(items.length);
                int padLeft = (int)(w * 0.22f);
                int padRight = (int)(w * 0.06f);
                int padTop = (int)(h * 0.065f);
                base.setPadding(padLeft, padTop, padRight, padTop);

                for (int i = 0; i < items.length; i++) {
                    final int idx = i;
                    android.widget.TextView tv = new android.widget.TextView(this);
                    String cleanText = items[i].replaceAll("^[^a-zA-Z0-9]+", "").trim();
                    tv.setText(cleanText);
                    tv.setTextColor(android.graphics.Color.WHITE);
                    tv.setTextSize(14.5f);
                    tv.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
                    tv.setGravity(android.view.Gravity.CENTER_VERTICAL);
                    tv.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                        android.view.ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

                    tv.setOnClickListener(new android.view.View.OnClickListener() {
                        @Override
                        public void onClick(android.view.View v) {
                            try { dialog.dismiss(); } catch (Exception ignored) {}
                            if (listener != null) {
                                listener.onClick(dialog, idx);
                            }
                        }
                    });
                    base.addView(tv);
                }
            }

            dialog.setContentView(base, new android.view.ViewGroup.LayoutParams(w, h));
            dialog.show();
            if (dialog.getWindow() != null) {
                dialog.getWindow().setLayout(w, h);
            }
        } catch (Exception ignored) {}
    }

}
