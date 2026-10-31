package com.bikash.mobileide;

import android.content.Context;
import java.util.Locale;

/** Creates a real, portable Android project tree inside the IDE workspace. */
public final class ProjectScaffold {
    private ProjectScaffold() {}

    public static String create(Context context, String rawName, boolean videoTemplate) {
        if (ProjectName.validationError(rawName) != null) return null;
        String slug = ProjectName.fileStem(rawName);
        String packageName = "com.oxdroid." + slug.replace('-', '_');
        String root = "projects/" + slug + "/";
        String javaPath = root + "app/src/main/java/" + packageName.replace('.', '/') + "/MainActivity.java";

        EngineCore.saveFile(context, root + "settings.gradle", "pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }\n" +
                "dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }\n" +
                "rootProject.name = '" + slug + "'\ninclude ':app'\n");
        EngineCore.saveFile(context, root + "build.gradle", "plugins {\n" +
                "    id 'com.android.application' version '8.2.2' apply false\n" +
                "}\n");
        EngineCore.saveFile(context, root + "app/build.gradle", "plugins { id 'com.android.application' }\n\n" +
                "android { namespace '" + packageName + "'; compileSdk 34\n" +
                "    defaultConfig { applicationId '" + packageName + "'; minSdk 24; targetSdk 34; versionCode 1; versionName '1.0' }\n" +
                "}\n");
        EngineCore.saveFile(context, root + "app/src/main/AndroidManifest.xml", "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
                "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" package=\"" + packageName + "\">\n" +
                "    <uses-permission android:name=\"android.permission.INTERNET\" />\n" +
                "    <application android:theme=\"@style/AppTheme\" android:label=\"" + escapeXml(rawName) + "\">\n" +
                "        <activity android:name=\".MainActivity\" android:exported=\"true\">\n" +
                "            <intent-filter><action android:name=\"android.intent.action.MAIN\" /><category android:name=\"android.intent.category.LAUNCHER\" /></intent-filter>\n" +
                "        </activity>\n" +
                "    </application>\n" +
                "</manifest>\n");
        EngineCore.saveFile(context, root + "app/src/main/res/values/styles.xml", "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<resources><style name=\"AppTheme\" parent=\"android:style/Theme.Material.Light.NoActionBar\"><item name=\"android:fontFamily\">sans</item><item name=\"android:colorAccent\">#3F51B5</item></style></resources>\n");
        EngineCore.saveFile(context, root + "app/src/main/res/values/strings.xml", "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<resources><string name=\"app_name\">" + escapeXml(rawName) + "</string></resources>\n");
        EngineCore.saveFile(context, root + "app/src/main/res/layout/activity_main.xml", videoTemplate ? videoLayout() : basicLayout());
        EngineCore.saveFile(context, javaPath, videoTemplate ? videoActivity(packageName) : basicActivity(packageName));
        return javaPath;
    }

    private static String basicActivity(String packageName) {
        return "package " + packageName + ";\n\nimport android.app.Activity;\nimport android.os.Bundle;\n\npublic class MainActivity extends Activity {\n    @Override public void onCreate(Bundle state) {\n        super.onCreate(state);\n        setContentView(R.layout.activity_main);\n    }\n}\n";
    }

    private static String videoActivity(String packageName) {
        return "package " + packageName + ";\n\n" +
                "import android.app.Activity;\nimport android.os.Bundle;\nimport android.widget.Toast;\n\n" +
                "public class MainActivity extends Activity {\n" +
                "    @Override public void onCreate(Bundle state) {\n" +
                "        super.onCreate(state); setContentView(R.layout.activity_main);\n" +
                "        findViewById(R.id.search_button).setOnClickListener(v -> Toast.makeText(this, \"Connect your API to search videos\", Toast.LENGTH_SHORT).show());\n" +
                "        findViewById(R.id.upload_button).setOnClickListener(v -> Toast.makeText(this, \"Add your authorized upload endpoint\", Toast.LENGTH_SHORT).show());\n" +
                "    }\n}\n";
    }

    private static String basicLayout() {
        return "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<LinearLayout xmlns:android=\"http://schemas.android.com/apk/res/android\" android:orientation=\"vertical\" android:padding=\"24dp\" android:layout_width=\"match_parent\" android:layout_height=\"match_parent\"><TextView android:text=\"Welcome to your Android app\" android:textSize=\"24sp\" android:layout_width=\"wrap_content\" android:layout_height=\"wrap_content\" /></LinearLayout>\n";
    }

    private static String videoLayout() {
        return "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" +
                "<ScrollView xmlns:android=\"http://schemas.android.com/apk/res/android\" android:layout_width=\"match_parent\" android:layout_height=\"match_parent\"><LinearLayout android:orientation=\"vertical\" android:padding=\"16dp\" android:layout_width=\"match_parent\" android:layout_height=\"wrap_content\">" +
                "<TextView android:text=\"Video App Starter\" android:textStyle=\"bold\" android:textSize=\"26sp\" android:layout_width=\"wrap_content\" android:layout_height=\"wrap_content\" />" +
                "<EditText android:id=\"@+id/search_input\" android:hint=\"Search videos\" android:layout_width=\"match_parent\" android:layout_height=\"wrap_content\" />" +
                "<Button android:id=\"@+id/search_button\" android:text=\"Search\" android:layout_width=\"match_parent\" android:layout_height=\"wrap_content\" />" +
                "<TextView android:text=\"Connect a backend API for feed, auth, upload, comments and subscriptions.\" android:paddingTop=\"20dp\" android:layout_width=\"match_parent\" android:layout_height=\"wrap_content\" />" +
                "<Button android:id=\"@+id/upload_button\" android:text=\"Upload authorized video\" android:layout_width=\"match_parent\" android:layout_height=\"wrap_content\" />" +
                "</LinearLayout></ScrollView>\n";
    }

    private static String escapeXml(String value) {
        return value == null ? "New App" : value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;");
    }
}
