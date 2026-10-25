package com.bikash.mobileide;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.widget.*;
import android.view.*;
import android.graphics.Color;
import android.graphics.drawable.*;
import android.content.res.ColorStateList;
import android.graphics.Typeface;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        
        RelativeLayout rootLayout = new RelativeLayout(this);
        rootLayout.setBackgroundColor(Color.parseColor("#0B0F15"));

        ScrollView scrollView = new ScrollView(this);
        RelativeLayout.LayoutParams scrollParams = new RelativeLayout.LayoutParams(-1, -1);
        scrollParams.bottomMargin = 140; 
        scrollView.setLayoutParams(scrollParams);
        
        LinearLayout mainLayout = new LinearLayout(this);
        mainLayout.setOrientation(LinearLayout.VERTICAL);
        mainLayout.setPadding(30, 20, 30, 40);
        scrollView.addView(mainLayout);

        // 1. TOP APP BAR
        RelativeLayout topBar = new RelativeLayout(this);
        TextView menuIcon = new TextView(this);
        menuIcon.setId(View.generateViewId());
        menuIcon.setText("≡");
        menuIcon.setTextColor(Color.WHITE);
        menuIcon.setTextSize(30);
        addPremiumTouchAnimation(menuIcon, null);
        
        LinearLayout titleLayout = new LinearLayout(this);
        titleLayout.setOrientation(LinearLayout.VERTICAL);
        titleLayout.setPadding(30, 0, 0, 0);
        RelativeLayout.LayoutParams titleParams = new RelativeLayout.LayoutParams(-2, -2);
        titleParams.addRule(RelativeLayout.RIGHT_OF, menuIcon.getId());
        titleLayout.setLayoutParams(titleParams);
        
        TextView appTitle = new TextView(this);
        appTitle.setText("OxDroid Studio");
        appTitle.setTextColor(Color.WHITE);
        appTitle.setTextSize(18);
        appTitle.setTypeface(null, Typeface.BOLD);
        titleLayout.addView(appTitle);
        
        TextView subTitle = new TextView(this);
        subTitle.setText("Mobile Development Workspace");
        subTitle.setTextColor(Color.GRAY);
        subTitle.setTextSize(11);
        titleLayout.addView(subTitle);
        
        LinearLayout rightIcons = new LinearLayout(this);
        rightIcons.setOrientation(LinearLayout.HORIZONTAL);
        RelativeLayout.LayoutParams rightParams = new RelativeLayout.LayoutParams(-2, -2);
        rightParams.addRule(RelativeLayout.ALIGN_PARENT_RIGHT);
        rightParams.addRule(RelativeLayout.CENTER_VERTICAL);
        rightIcons.setLayoutParams(rightParams);
        
        TextView searchIcon = new TextView(this);
        searchIcon.setText("🔍");
        searchIcon.setTextColor(Color.WHITE);
        searchIcon.setTextSize(18);
        searchIcon.setPadding(0, 0, 30, 0);
        addPremiumTouchAnimation(searchIcon, null);
        
        TextView dotMenu = new TextView(this);
        dotMenu.setText("⋮");
        dotMenu.setTextColor(Color.WHITE);
        dotMenu.setTextSize(22);
        addPremiumTouchAnimation(dotMenu, null);
        
        rightIcons.addView(searchIcon);
        rightIcons.addView(dotMenu);
        
        topBar.addView(menuIcon);
        topBar.addView(titleLayout);
        topBar.addView(rightIcons);
        mainLayout.addView(topBar);

        // 2. HERO BANNER
        ImageView banner = new ImageView(this);
        int bannerResId = getResources().getIdentifier("hero_banner", "drawable", getPackageName());
        if(bannerResId != 0) banner.setImageResource(bannerResId);
        banner.setScaleType(ImageView.ScaleType.FIT_XY);
        LinearLayout.LayoutParams bannerParams = new LinearLayout.LayoutParams(-1, 350); 
        bannerParams.setMargins(0, 40, 0, 40);
        banner.setLayoutParams(bannerParams);
        GradientDrawable shape = new GradientDrawable();
        shape.setShape(GradientDrawable.RECTANGLE);
        shape.setCornerRadius(30f);
        banner.setBackground(shape);
        banner.setClipToOutline(true);
        mainLayout.addView(banner);

        // 3. MAIN ACTION BUTTONS
        LinearLayout actionRow = new LinearLayout(this);
        actionRow.setOrientation(LinearLayout.HORIZONTAL);
        actionRow.addView(createActionCard("➕", "New Project", "Start a new app", "#1C68F3", new Runnable() {
            public void run() { showLanguageSelectionDialog(); }
        }));
        actionRow.addView(createActionCard("📁", "Open Project", "Browse storage", "#23252E", null));
        actionRow.addView(createActionCard("🔗", "Clone Git", "From repository", "#642299", null));
        actionRow.addView(createActionCard("🎛", "Samples", "Template apps", "#115E39", null));
        mainLayout.addView(actionRow);

        // 4. RECENT PROJECTS
        RelativeLayout recentHeader = new RelativeLayout(this);
        recentHeader.setPadding(0, 40, 0, 10);
        TextView recentTitle = new TextView(this);
        recentTitle.setText("Recent Projects");
        recentTitle.setTextColor(Color.WHITE);
        recentTitle.setTextSize(18);
        recentTitle.setTypeface(null, Typeface.BOLD);
        
        TextView seeAll = new TextView(this);
        seeAll.setText("See All ❯");
        seeAll.setTextColor(Color.parseColor("#1C68F3"));
        seeAll.setTextSize(14);
        RelativeLayout.LayoutParams seeAllParams = new RelativeLayout.LayoutParams(-2, -2);
        seeAllParams.addRule(RelativeLayout.ALIGN_PARENT_RIGHT);
        seeAllParams.addRule(RelativeLayout.CENTER_VERTICAL);
        seeAll.setLayoutParams(seeAllParams);
        addPremiumTouchAnimation(seeAll, null);
        
        recentHeader.addView(recentTitle);
        recentHeader.addView(seeAll);
        mainLayout.addView(recentHeader);

        mainLayout.addView(createProjectItem("🤖", "Androidify", "Kotlin • Jetpack Compose", "Today", "#11A955"));
        mainLayout.addView(createProjectItem("⚙️", "ZenBuild", "C++ • Native Project", "Yesterday", "#8957E5"));
        mainLayout.addView(createProjectItem("🎮", "MyApp", "Java • Android App", "3 days ago", "#00B4D8"));
        mainLayout.addView(createProjectItem("▶️", "PlayerX", "Kotlin • Media App", "5 days ago", "#E53935"));

        // 5. QUICK TOOLS
        TextView toolsTitle = new TextView(this);
        toolsTitle.setText("Quick Tools");
        toolsTitle.setTextColor(Color.WHITE);
        toolsTitle.setTextSize(18);
        toolsTitle.setTypeface(null, Typeface.BOLD);
        toolsTitle.setPadding(0, 40, 0, 10);
        mainLayout.addView(toolsTitle);

        LinearLayout toolsRow1 = new LinearLayout(this);
        toolsRow1.setOrientation(LinearLayout.HORIZONTAL);
        toolsRow1.addView(createActionCard("❯_", "Terminal", "Command line", "#161B22", new Runnable() {
            public void run() { openNativeTerminal(); }
        }));
        toolsRow1.addView(createActionCard("🔨", "Build", "Build & Run", "#161B22", null));
        toolsRow1.addView(createActionCard("▶", "Run App", "On Device", "#161B22", null));
        toolsRow1.addView(createActionCard("📱", "Devices", "Manage ADB", "#161B22", null));
        mainLayout.addView(toolsRow1);

        LinearLayout toolsRow2 = new LinearLayout(this);
        toolsRow2.setOrientation(LinearLayout.HORIZONTAL);
        toolsRow2.addView(createActionCard("🔗", "Git", "Version Control", "#161B22", null));
        toolsRow2.addView(createActionCard("📁", "SDK", "SDK Manager", "#161B22", null));
        toolsRow2.addView(createActionCard("🗂", "Logcat", "View Logs", "#161B22", null));
        toolsRow2.addView(createActionCard("⚙️", "Settings", "IDE Settings", "#161B22", null));
        mainLayout.addView(toolsRow2);

        // 6. STORAGE USAGE SECTION
        LinearLayout storageBox = new LinearLayout(this);
        storageBox.setOrientation(LinearLayout.VERTICAL);
        storageBox.setBackground(getRippleDrawable("#161B22", 20f));
        storageBox.setPadding(40, 30, 40, 30);
        LinearLayout.LayoutParams storageParams = new LinearLayout.LayoutParams(-1, -2);
        storageParams.setMargins(0, 40, 0, 20);
        storageBox.setLayoutParams(storageParams);
        addPremiumTouchAnimation(storageBox, null);

        RelativeLayout storeHeader = new RelativeLayout(this);
        LinearLayout storeTitleLayout = new LinearLayout(this);
        storeTitleLayout.setOrientation(LinearLayout.HORIZONTAL);
        TextView hddIcon = new TextView(this);
        hddIcon.setText("💽  ");
        hddIcon.setTextColor(Color.WHITE);
        TextView storeText = new TextView(this);
        storeText.setText("Storage Usage");
        storeText.setTextColor(Color.WHITE);
        storeText.setTypeface(null, Typeface.BOLD);
        storeTitleLayout.addView(hddIcon);
        storeTitleLayout.addView(storeText);

        TextView manageBtn = new TextView(this);
        manageBtn.setText("Manage ❯");
        manageBtn.setTextColor(Color.parseColor("#1C68F3"));
        RelativeLayout.LayoutParams manParams = new RelativeLayout.LayoutParams(-2, -2);
        manParams.addRule(RelativeLayout.ALIGN_PARENT_RIGHT);
        manageBtn.setLayoutParams(manParams);
        addPremiumTouchAnimation(manageBtn, null);

        storeHeader.addView(storeTitleLayout);
        storeHeader.addView(manageBtn);
        storageBox.addView(storeHeader);

        ProgressBar progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setProgress(30);
        progress.setProgressTintList(ColorStateList.valueOf(Color.parseColor("#1C68F3")));
        progress.setPadding(0, 20, 0, 10);
        storageBox.addView(progress);

        TextView storeSub = new TextView(this);
        storeSub.setText("18.4 GB used / 64 GB");
        storeSub.setTextColor(Color.GRAY);
        storeSub.setTextSize(10);
        storageBox.addView(storeSub);

        mainLayout.addView(storageBox);
        rootLayout.addView(scrollView);

        // 7. BOTTOM NAVIGATION BAR
        LinearLayout bottomNav = new LinearLayout(this);
        bottomNav.setOrientation(LinearLayout.HORIZONTAL);
        bottomNav.setBackgroundColor(Color.parseColor("#0B0F15"));
        RelativeLayout.LayoutParams navParams = new RelativeLayout.LayoutParams(-1, 140);
        navParams.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM);
        bottomNav.setLayoutParams(navParams);
        
        bottomNav.addView(createNavBtn("🏠\nHome", true));
        bottomNav.addView(createNavBtn("📁\nProjects", false));
        bottomNav.addView(createNavBtn("</>\nEditor", false));
        bottomNav.addView(createNavBtn("⚙️\nSettings", false));
        
        rootLayout.addView(bottomNav);
        setContentView(rootLayout);
    }

    // ==========================================
    // DIALOGS & TERMINAL ENGINE
    // ==========================================
    private void showLanguageSelectionDialog() {
        final Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        
        LinearLayout dialogLayout = new LinearLayout(this);
        dialogLayout.setOrientation(LinearLayout.VERTICAL);
        dialogLayout.setPadding(50, 50, 50, 50);
        dialogLayout.setBackground(getRippleDrawable("#1E1E1E", 30f));

        TextView title = new TextView(this);
        title.setText("Create New Project");
        title.setTextColor(Color.WHITE);
        title.setTextSize(22);
        title.setTypeface(null, Typeface.BOLD);
        dialogLayout.addView(title);

        TextView subTitle = new TextView(this);
        subTitle.setText("Choose your primary programming language.");
        subTitle.setTextColor(Color.parseColor("#8B949E"));
        subTitle.setPadding(0, 10, 0, 40);
        dialogLayout.addView(subTitle);

        Button javaBtn = new Button(this);
        javaBtn.setText("☕ Java (Native Android)");
        javaBtn.setBackground(getRippleDrawable("#1C68F3", 15f));
        javaBtn.setTextColor(Color.WHITE);
        javaBtn.setPadding(0, 30, 0, 30);
        addPremiumTouchAnimation(javaBtn, new Runnable() {
            public void run() {
                android.content.Intent intent = new android.content.Intent(MainActivity.this, EditorActivity.class); startActivity(intent);
                dialog.dismiss();
            }
        });
        dialogLayout.addView(javaBtn);

        Space space = new Space(this);
        space.setLayoutParams(new LinearLayout.LayoutParams(1, 20));
        dialogLayout.addView(space);

        Button pythonBtn = new Button(this);
        pythonBtn.setText("🐍 Python (Wrapper Engine)");
        pythonBtn.setBackground(getRippleDrawable("#11A955", 15f));
        pythonBtn.setTextColor(Color.WHITE);
        pythonBtn.setPadding(0, 30, 0, 30);
        addPremiumTouchAnimation(pythonBtn, new Runnable() {
            public void run() {
                Toast.makeText(MainActivity.this, "Injecting Python Base APK...", Toast.LENGTH_SHORT).show();
                dialog.dismiss();
            }
        });
        dialogLayout.addView(pythonBtn);

        dialog.setContentView(dialogLayout);
        dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        dialog.show();
    }

    private void openNativeTerminal() {
        final Dialog termDialog = new Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen);
        LinearLayout termLayout = new LinearLayout(this);
        termLayout.setOrientation(LinearLayout.VERTICAL);
        termLayout.setBackgroundColor(Color.BLACK);

        final TextView output = new TextView(this);
        output.setTextColor(Color.GREEN);
        output.setTypeface(Typeface.MONOSPACE);
        output.setText("OxDroid Native Engine v1.0\nReady.\n");
        
        ScrollView outScroll = new ScrollView(this);
        outScroll.setLayoutParams(new LinearLayout.LayoutParams(-1, 0, 1));
        outScroll.addView(output);
        
        final EditText input = new EditText(this);
        input.setHint("Type native linux command (e.g. ls /)");
        input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.GRAY);

        Button runBtn = new Button(this);
        runBtn.setText("RUN");
        runBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                String cmd = input.getText().toString();
                output.append("\n$ " + cmd + "\n");
                input.setText("");
                try {
                    Process process = Runtime.getRuntime().exec(new String[]{"sh", "-c", cmd});
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    BufferedReader error = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                    String line;
                    while ((line = reader.readLine()) != null) output.append(line + "\n");
                    while ((line = error.readLine()) != null) output.append("[ERR] " + line + "\n");
                    process.waitFor();
                } catch (Exception e) {
                    output.append("Engine Error: " + e.getMessage() + "\n");
                }
            }
        });

        termLayout.addView(outScroll);
        termLayout.addView(input);
        termLayout.addView(runBtn);
        termDialog.setContentView(termLayout);
        termDialog.show();
    }

    // ==========================================
    // UI BUILDER HELPERS & ANIMATIONS
    // ==========================================
    private void addPremiumTouchAnimation(final View view, final Runnable action) {
        view.setClickable(true);
        view.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN) {
                    v.animate().scaleX(0.92f).scaleY(0.92f).setDuration(100).start();
                } else if (event.getAction() == MotionEvent.ACTION_UP || event.getAction() == MotionEvent.ACTION_CANCEL) {
                    v.animate().scaleX(1f).scaleY(1f).setDuration(100).start();
                }
                return false; 
            }
        });
        view.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (action != null) action.run();
                else Toast.makeText(MainActivity.this, "Clicked!", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private View createNavBtn(String text, boolean isActive) {
        TextView btn = new TextView(this);
        btn.setText(text);
        btn.setTextColor(isActive ? Color.parseColor("#1C68F3") : Color.GRAY);
        btn.setGravity(Gravity.CENTER);
        btn.setLayoutParams(new LinearLayout.LayoutParams(0, -1, 1.0f));
        addPremiumTouchAnimation(btn, null);
        return btn;
    }

    private View createActionCard(String iconStr, String titleStr, String subStr, String bgColor, final Runnable action) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(15, 30, 15, 30);
        card.setGravity(Gravity.CENTER);
        
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f);
        params.setMargins(10, 10, 10, 10);
        card.setLayoutParams(params);
        card.setBackground(getRippleDrawable(bgColor, 25f));
        addPremiumTouchAnimation(card, action);
        
        TextView icon = new TextView(this);
        icon.setText(iconStr);
        icon.setTextSize(22);
        icon.setGravity(Gravity.CENTER);
        card.addView(icon);
        
        TextView title = new TextView(this);
        title.setText(titleStr);
        title.setTextColor(Color.WHITE);
        title.setTextSize(11);
        title.setTypeface(null, Typeface.BOLD);
        title.setPadding(0, 10, 0, 5);
        title.setGravity(Gravity.CENTER);
        card.addView(title);
        
        TextView sub = new TextView(this);
        sub.setText(subStr);
        sub.setTextColor(Color.parseColor("#8B949E"));
        sub.setTextSize(9);
        sub.setGravity(Gravity.CENTER);
        card.addView(sub);
        return card;
    }

    private View createProjectItem(String iconStr, final String titleStr, String subStr, String timeStr, String iconBgColor) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(20, 25, 20, 25);
        row.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.setMargins(0, 10, 0, 10);
        row.setLayoutParams(params);
        row.setBackground(getRippleDrawable("#161B22", 20f));
        
        addPremiumTouchAnimation(row, new Runnable() {
            public void run() { if(titleStr.equals("Androidify") || titleStr.equals("ZenBuild")){ android.content.Intent i = new android.content.Intent(MainActivity.this, EditorActivity.class); startActivity(i); } else { Toast.makeText(MainActivity.this, "Opening " + titleStr + "...", Toast.LENGTH_SHORT).show(); } }
        });
        
        TextView icon = new TextView(this);
        icon.setText(iconStr);
        icon.setTextSize(20);
        icon.setGravity(Gravity.CENTER);
        icon.setBackground(getRippleDrawable(iconBgColor, 15f));
        icon.setLayoutParams(new LinearLayout.LayoutParams(100, 100));
        row.addView(icon);
        
        LinearLayout textLayout = new LinearLayout(this);
        textLayout.setOrientation(LinearLayout.VERTICAL);
        textLayout.setPadding(30, 0, 0, 0);
        textLayout.setLayoutParams(new LinearLayout.LayoutParams(0, -2, 1));
        
        TextView title = new TextView(this);
        title.setText(titleStr);
        title.setTextColor(Color.WHITE);
        title.setTextSize(14);
        title.setTypeface(null, Typeface.BOLD);
        textLayout.addView(title);
        
        TextView sub = new TextView(this);
        sub.setText(subStr);
        sub.setTextColor(Color.parseColor("#8B949E"));
        sub.setTextSize(11);
        textLayout.addView(sub);
        row.addView(textLayout);
        
        LinearLayout rightLayout = new LinearLayout(this);
        rightLayout.setOrientation(LinearLayout.VERTICAL);
        rightLayout.setGravity(Gravity.RIGHT);
        
        TextView dotMenu = new TextView(this);
        dotMenu.setText("⋮");
        dotMenu.setTextColor(Color.WHITE);
        dotMenu.setTextSize(18);
        dotMenu.setGravity(Gravity.RIGHT);
        rightLayout.addView(dotMenu);
        
        TextView time = new TextView(this);
        time.setText(timeStr);
        time.setTextColor(Color.parseColor("#8B949E"));
        time.setTextSize(10);
        rightLayout.addView(time);
        
        row.addView(rightLayout);
        return row;
    }

    private Drawable getRippleDrawable(String colorHex, float radius) {
        GradientDrawable shape = new GradientDrawable();
        shape.setShape(GradientDrawable.RECTANGLE);
        shape.setCornerRadius(radius);
        shape.setColor(Color.parseColor(colorHex));
        return new RippleDrawable(ColorStateList.valueOf(Color.parseColor("#40FFFFFF")), shape, null);
    }
}
