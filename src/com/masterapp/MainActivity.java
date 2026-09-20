package com.masterapp;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.util.ArrayList;
import dalvik.system.DexClassLoader;

public class MainActivity extends Activity {

    private WebView editorWebView, previewWebView;
    private String currentFileName = "Main.java";
    private boolean isPreviewVisible = false;
    private DexClassLoader interpreterLoader = null;
    private static final int EDITOR_LOAD_TIMEOUT = 5000; // 5 seconds timeout
    private Handler editorLoadHandler = new Handler();
    private Runnable editorLoadRunnable;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Edge-to-edge with insets awareness
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            getWindow().setDecorFitsSystemWindows(false);
        } else {
            getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        }

        editorWebView = findViewById(R.id.editorWebView);
        previewWebView = findViewById(R.id.previewWebView);

        editorWebView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        previewWebView.setLayerType(View.LAYER_TYPE_HARDWARE, null);

        configureWebView(editorWebView, true);
        configureWebView(previewWebView, false);

        setupLinux();
    }

    private void configureWebView(WebView webView, boolean isEditor) {
        if (webView == null) return;

        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webSettings.setDomStorageEnabled(true);
        webSettings.setAllowFileAccess(true);
        webSettings.setAllowContentAccess(true);
        webSettings.setLoadWithOverviewMode(true);
        webSettings.setUseWideViewPort(true);
        webSettings.setMediaPlaybackRequiresUserGesture(false);

        // Set Content Security Policy
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                if (isEditor) {
                    view.evaluateJavascript(
                        "document.addEventListener('DOMContentLoaded', function() {" +
                        "   var meta = document.createElement('meta');" +
                        "   meta.httpEquiv = 'Content-Security-Policy';" +
                        "   meta.content = \"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; media-src 'self'; object-src 'none'; frame-src 'none';\";" +
                        "   document.head.appendChild(meta);" +
                        "});", null);
                }
            }
        });

        if (isEditor) {
            webView.addJavascriptInterface(new AndroidBridge(), "Android");
            webView.loadUrl("file:///android_asset/editor/index.html");

            // Set up timeout for editor loading
            editorLoadRunnable = new Runnable() {
                @Override
                public void run() {
                    if (editorWebView != null && !editorWebView.isShown()) {
                        log("Error: Editor loading timed out. Showing fallback UI.");
                        // You can add code here to show a fallback UI or message
                    }
                }
            };
            editorLoadHandler.postDelayed(editorLoadRunnable, EDITOR_LOAD_TIMEOUT);
        }
    }

    // ============ LOG to HTML ============
    private String esc(String s) {
        return s.replace("\\", "\\\\").replace("'", "\\'")
                .replace("\n", "\\n").replace("\r", "");
    }

    private void log(final String msg) {
        if (editorWebView == null) return;
        final String safe = esc(msg);
        editorWebView.post(new Runnable() {
            @Override
            public void run() {
                editorWebView.evaluateJavascript("appendTerminal('" + safe + "')", null);
            }
        });
    }

    // ============ INTERPRETERS ============
    private DexClassLoader getInterpreterLoader() throws Exception {
        if (interpreterLoader != null) return interpreterLoader;
        File dexDir = getDir("dex", MODE_PRIVATE);
        File dexFile = new File(dexDir, "interpreters.dex");
        if (!dexFile.exists() || dexFile.length() < 1000000) {
            copyAssetToFile("libs/interpreters.dex", dexFile);
        }
        interpreterLoader = new DexClassLoader(
            dexFile.getAbsolutePath(), dexDir.getAbsolutePath(),
            null, getClassLoader());
        return interpreterLoader;
    }

    private void copyAssetToFile(String assetPath, File destFile) throws Exception {
        try (InputStream is = getAssets().open(assetPath);
             FileOutputStream fos = new FileOutputStream(destFile)) {
            byte[] buf = new byte[8192];
            int len;
            while ((len = is.read(buf)) > 0) {
                fos.write(buf, 0, len);
            }
        }
    }

    // ============ FILE HELPERS ============
    private File currentFile() {
        File dir = new File(getFilesDir(), "projects");
        dir.mkdirs();
        if (currentFileName == null || currentFileName.length() == 0) {
            currentFileName = "Main.java";
        }
        return new File(dir, currentFileName);
    }

    private File specificFile(String name) {
        File dir = new File(getFilesDir(), "projects");
        dir.mkdirs();
        return new File(dir, name);
    }

    // ============ LINUX HELPERS ============
    private boolean isLinuxReady() {
        File rootfs = new File(getFilesDir(), "linux/rootfs");
        File proot = new File(getFilesDir(), "linux/proot");
        return rootfs.exists() && proot.exists();
    }

    private void copyCodeToLinux() throws Exception {
        File rootfs = new File(getFilesDir(), "linux/rootfs");
        File tmpDir = new File(rootfs, "tmp");
        tmpDir.mkdirs();
        File dest = new File(tmpDir, currentFileName);

        try (BufferedReader br = new BufferedReader(new FileReader(currentFile()));
             FileWriter fw = new FileWriter(dest)) {
            String line;
            while ((line = br.readLine()) != null) {
                fw.write(line + "\n");
            }
        }
    }

    private void execInLinux(String command) throws Exception {
        File linuxDir = new File(getFilesDir(), "linux");
        File rootfs = new File(linuxDir, "rootfs");
        File proot = new File(linuxDir, "proot");
        File resolv = new File(rootfs, "etc/resolv.conf");
        resolv.getParentFile().mkdirs();

        if (!resolv.exists()) {
            try (FileWriter fw = new FileWriter(resolv)) {
                fw.write("nameserver 8.8.8.8\nnameserver 1.1.1.1\n");
            }
        }

        ProcessBuilder pb = new ProcessBuilder(
            proot.getAbsolutePath(), "--link2symlink", "-0",
            "-r", rootfs.getAbsolutePath(),
            "-b", "/dev", "-b", "/proc", "-b", "/sys",
            "-w", "/root", "/bin/sh", "-c", command);
        pb.environment().put("PROOT_TMP_DIR", getCacheDir().getAbsolutePath());
        pb.environment().put("PROOT_NO_SECCOMP", "1");
        pb.environment().put("HOME", "/root");
        pb.environment().put("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin");
        pb.environment().put("TERM", "xterm-256color");
        pb.redirectErrorStream(true);

        Process p = pb.start();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = br.readLine()) != null) {
                log(line);
            }
        }
        p.waitFor();
    }

    private void setupLinux() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    File linuxDir = new File(getFilesDir(), "linux");
                    File rootfs = new File(linuxDir, "rootfs");
                    File proot = new File(linuxDir, "proot");
                    File tar = new File(linuxDir, "alpine_data");
                    linuxDir.mkdirs();

                    if (rootfs.exists() && rootfs.listFiles() != null && rootfs.listFiles().length > 0) {
                        return;
                    }

                    if (!proot.exists() || proot.length() < 100000) {
                        copyAssetToFile("linux/proot_bin", proot);
                        proot.setExecutable(true);
                    }

                    if (!tar.exists() || tar.length() < 1000000) {
                        copyAssetToFile("linux/alpine_data", tar);
                    }

                    rootfs.mkdirs();
                    Process p = Runtime.getRuntime().exec(new String[]{"sh", "-c",
                        "cd " + rootfs.getAbsolutePath() + " && tar -xzf " + tar.getAbsolutePath()});
                    p.waitFor();
                    log("✓ Linux environment ready");
                } catch (final Exception e) {
                    log("Error: Linux setup error: " + e.getMessage());
                }
            }
        }).start();
    }

    // ============ CODE RUNNERS ============
    private void runCode() {
        saveCurrentFile();
        String lower = currentFileName.toLowerCase();
        if (lower.endsWith(".java") || lower.endsWith(".bsh")) {
            runJavaBeanShell();
        } else if (lower.endsWith(".js")) {
            runRhino();
        } else if (lower.endsWith(".html") || lower.endsWith(".htm")) {
            togglePreview();
        } else if (lower.endsWith(".py")) {
            runPython();
        } else if (lower.endsWith(".c") || lower.endsWith(".cpp") || lower.endsWith(".cc")) {
            runCpp();
        } else if (lower.endsWith(".sh")) {
            runShell();
        } else {
            log("Error: Unsupported file type: " + currentFileName);
        }
    }

    private void runPython() {
        log("> Running " + currentFileName + " with Python 3...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (!isLinuxReady()) {
                        log("Error: Linux not ready yet");
                        return;
                    }
                    copyCodeToLinux();
                    execInLinux("if ! which python3 > /dev/null 2>&1; then echo 'Installing Python...'; apk add --no-cache python3; fi; python3 /tmp/" + currentFileName);
                } catch (final Exception e) {
                    log("Error: Python error: " + e.getMessage());
                }
            }
        }).start();
    }

    private void runCpp() {
        log("> Compiling " + currentFileName + "...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (!isLinuxReady()) {
                        log("Error: Linux not ready yet");
                        return;
                    }
                    copyCodeToLinux();
                    boolean isC = currentFileName.toLowerCase().endsWith(".c");
                    String cc = isC ? "gcc" : "g++";
                    execInLinux("if ! which " + cc + " > /dev/null 2>&1; then echo 'Installing compiler...'; apk add --no-cache g++ gcc; fi; " +
                        cc + " /tmp/" + currentFileName + " -o /tmp/a.out && /tmp/a.out");
                } catch (final Exception e) {
                    log("Error: C/C++ error: " + e.getMessage());
                }
            }
        }).start();
    }

    private void runShell() {
        log("> Running " + currentFileName + "...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (!isLinuxReady()) {
                        log("Error: Linux not ready yet");
                        return;
                    }
                    copyCodeToLinux();
                    execInLinux("sh /tmp/" + currentFileName);
                } catch (final Exception e) {
                    log("Error: Shell error: " + e.getMessage());
                }
            }
        }).start();
    }

    private void runJavaBeanShell() {
        log("> Running " + currentFileName + " with BeanShell...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                PrintStream oldOut = System.out;
                PrintStream oldErr = System.err;
                try {
                    DexClassLoader loader = getInterpreterLoader();
                    Class<?> c = loader.loadClass("bsh.Interpreter");
                    Object itp = c.newInstance();
                    System.setOut(new PrintStream(baos));
                    System.setErr(new PrintStream(baos));
                    try (BufferedReader br = new BufferedReader(new FileReader(currentFile()))) {
                        StringBuilder sb = new StringBuilder();
                        String ln;
                        while ((ln = br.readLine()) != null) {
                            sb.append(ln).append("\n");
                        }
                        Method m = c.getMethod("eval", String.class);
                        m.invoke(itp, sb.toString());
                    }
                } catch (Exception e) {
                    try {
                        baos.write(("\nError: " + e).getBytes());
                    } catch (Exception x) {
                        // Ignore
                    }
                } finally {
                    System.setOut(oldOut);
                    System.setErr(oldErr);
                }
                final String out = baos.toString().trim();
                log(out.length() == 0 ? "✓ Executed (no output)" : out);
            }
        }).start();
    }

    private void runRhino() {
        log("> Running " + currentFileName + " with Rhino...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                PrintStream oldOut = System.out;
                PrintStream oldErr = System.err;
                try {
                    DexClassLoader loader = getInterpreterLoader();
                    Class<?> ctxC = loader.loadClass("org.mozilla.javascript.Context");
                    Class<?> scC = loader.loadClass("org.mozilla.javascript.Scriptable");
                    System.setOut(new PrintStream(baos));
                    System.setErr(new PrintStream(baos));
                    try (BufferedReader br = new BufferedReader(new FileReader(currentFile()))) {
                        StringBuilder sb = new StringBuilder();
                        String ln;
                        while ((ln = br.readLine()) != null) {
                            sb.append(ln).append("\n");
                        }
                        Object cx = ctxC.getMethod("enter").invoke(null);
                        Object scope = ctxC.getMethod("initStandardObjects").invoke(cx);
                        ctxC.getMethod("evaluateString", scC, String.class, String.class, int.class, null)
                            .invoke(cx, scope, sb.toString(), "script", 1, null);
                        ctxC.getMethod("exit").invoke(null);
                    }
                } catch (Exception e) {
                    try {
                        baos.write(("\nError: " + e).getBytes());
                    } catch (Exception x) {
                        // Ignore
                    }
                } finally {
                    System.setOut(oldOut);
                    System.setErr(oldErr);
                }
                final String out = baos.toString().trim();
                log(out.length() == 0 ? "✓ Executed (no output)" : out);
            }
        }).start();
    }

    // ============ FILE I/O ============
    private void saveCurrentFile() {
        if (editorWebView == null) return;
        editorWebView.evaluateJavascript("getCode()", new android.webkit.ValueCallback<String>() {
            @Override
            public void onReceiveValue(String code) {
                if (code != null && code.length() > 1) {
                    code = code.substring(1, code.length() - 1)
                        .replace("\\n", "\n").replace("\\\"", "\"")
                        .replace("\\\\", "\\");
                } else {
                    code = "";
                }
                try (FileWriter fw = new FileWriter(currentFile())) {
                    fw.write(code);
                } catch (Exception e) {
                    log("Error: Failed to save file: " + e.getMessage());
                }
            }
        });
    }

    private void togglePreview() {
        if (editorWebView == null || previewWebView == null) return;
        isPreviewVisible = !isPreviewVisible;
        if (isPreviewVisible) {
            editorWebView.evaluateJavascript("getCode()", new android.webkit.ValueCallback<String>() {
                @Override
                public void onReceiveValue(String code) {
                    if (code != null && code.length() > 1) {
                        code = code.substring(1, code.length() - 1)
                            .replace("\\n", "\n").replace("\\\"", "\"")
                            .replace("\\\\", "\\");
                    } else {
                        code = "";
                    }
                    previewWebView.loadDataWithBaseURL(null, code, "text/html", "UTF-8", null);
                    previewWebView.setVisibility(View.VISIBLE);
                    editorWebView.setVisibility(View.GONE);
                    log("Preview ON");
                }
            });
        } else {
            editorWebView.setVisibility(View.VISIBLE);
            previewWebView.setVisibility(View.GONE);
            log("Preview OFF");
        }
    }

    private void refreshFileList() {
        if (editorWebView == null) return;
        File dir = new File(getFilesDir(), "projects");
        File[] files = dir.listFiles();
        ArrayList<String> names = new ArrayList<>();
        if (files != null) {
            for (File f : files) {
                names.add(f.getName());
            }
        }
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < names.size(); i++) {
            json.append("\"").append(names.get(i)).append("\"");
            if (i < names.size() - 1) {
                json.append(",");
            }
        }
        json.append("]");
        final String js = json.toString();
        editorWebView.post(new Runnable() {
            @Override
            public void run() {
                editorWebView.evaluateJavascript("updateFileList('" + js + "')", null);
            }
        });
    }

    @Override
    public void onBackPressed() {
        if (editorWebView == null) {
            finish();
            return;
        }
        editorWebView.evaluateJavascript(
            "(function(){" +
            "if(window.__smartTaskBack){try{var r=window.__smartTaskBack();if(r==='1'||r===1)return '1';}catch(e){}}" +
            "return '0';})()",
            new android.webkit.ValueCallback<String>() {
                @Override
                public void onReceiveValue(String r) {
                    if (r == null || r.equals("0") || r.equals("\"0\"") || r.equals("null")) {
                        finish();
                    }
                }
            });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (editorWebView != null) {
            editorWebView.removeAllViews();
            editorWebView.destroy();
            editorWebView = null;
        }
        if (previewWebView != null) {
            previewWebView.removeAllViews();
            previewWebView.destroy();
            previewWebView = null;
        }
        if (editorLoadHandler != null && editorLoadRunnable != null) {
            editorLoadHandler.removeCallbacks(editorLoadRunnable);
        }
    }

    // ============ JS BRIDGE ============
    public class AndroidBridge {

        @JavascriptInterface
        public void runCodeClick() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    runCode();
                }
            });
        }

        @JavascriptInterface
        public void buildAPKClick() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    runCode();
                }
            });
        }

        @JavascriptInterface
        public void togglePreviewClick() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (editorWebView != null) {
                        editorWebView.evaluateJavascript("togglePreview()", null);
                    }
                }
            });
        }

        @JavascriptInterface
        public void logMsg(final String m) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    log(m);
                }
            });
        }

        @JavascriptInterface
        public void clearTerminal() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (editorWebView != null) {
                        editorWebView.evaluateJavascript("clearTerminal()", null);
                    }
                }
            });
        }

        @JavascriptInterface
        public void listFiles() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    refreshFileList();
                }
            });
        }

        @JavascriptInterface
        public void openFile(final String filename) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        File f = specificFile(filename);
                        if (!f.exists()) {
                            log("Error: File does not exist: " + filename);
                            return;
                        }
                        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                            StringBuilder sb = new StringBuilder();
                            String ln;
                            while ((ln = br.readLine()) != null) {
                                sb.append(ln).append("\n");
                            }
                            currentFileName = filename;
                            String content = esc(sb.toString());
                            if (editorWebView != null) {
                                editorWebView.evaluateJavascript("setCode('" + content + "');setActiveTab('" + esc(filename) + "');updateCurrentFileLabel('" + esc(filename) + "');", null);
                            }
                            log("Opened: " + filename);
                        }
                    } catch (Exception e) {
                        log("Error: Open error: " + e.getMessage());
                    }
                }
            });
        }

        @JavascriptInterface
        public void saveSpecificFile(final String filename, final String code) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try (FileWriter fw = new FileWriter(specificFile(filename))) {
                        fw.write(code);
                    } catch (Exception e) {
                        log("Error: Failed to save file: " + e.getMessage());
                    }
                }
            });
        }

        @JavascriptInterface
        public void promptNewFile() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    final EditText input = new EditText(MainActivity.this);
                    input.setHint("जैसे: Test.java");
                    input.setTextColor(0xFFDFE1E5);
                    new AlertDialog.Builder(MainActivity.this)
                        .setTitle("New File")
                        .setView(input)
                        .setPositiveButton("Create", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface d, int w) {
                                String name = input.getText().toString().trim();
                                if (name.length() == 0) {
                                    name = "NewFile.java";
                                }
                                try (FileWriter fw = new FileWriter(specificFile(name))) {
                                    fw.write("");
                                    log("Created: " + name);
                                    refreshFileList();
                                    if (editorWebView != null) {
                                        editorWebView.evaluateJavascript("setActiveTab('" + esc(name) + "');setCode('');updateCurrentFileLabel('" + esc(name) + "');", null);
                                    }
                                    currentFileName = name;
                                } catch (Exception e) {
                                    log("Error: Create error: " + e.getMessage());
                                }
                            }
                        })
                        .setNegativeButton("Cancel", null).show();
                }
            });
        }

        @JavascriptInterface
        public void renameFile(final String oldName) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    final EditText input = new EditText(MainActivity.this);
                    input.setText(oldName);
                    input.setTextColor(0xFFDFE1E5);
                    new AlertDialog.Builder(MainActivity.this)
                        .setTitle("Rename")
                        .setView(input)
                        .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface d, int w) {
                                String nn = input.getText().toString().trim();
                                if (nn.length() == 0 || nn.equals(oldName)) {
                                    return;
                                }
                                if (specificFile(oldName).renameTo(specificFile(nn))) {
                                    log("Renamed: " + oldName + " → " + nn);
                                    if (currentFileName.equals(oldName)) {
                                        currentFileName = nn;
                                    }
                                    refreshFileList();
                                } else {
                                    log("Error: Failed to rename file: " + oldName);
                                }
                            }
                        })
                        .setNegativeButton("Cancel", null).show();
                }
            });
        }

        @JavascriptInterface
        public void deleteFile(final String name) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    new AlertDialog.Builder(MainActivity.this)
                        .setTitle("Delete " + name + "?")
                        .setPositiveButton("Delete", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface d, int w) {
                                if (specificFile(name).delete()) {
                                    log("Deleted: " + name);
                                    refreshFileList();
                                } else {
                                    log("Error: Failed to delete file: " + name);
                                }
                            }
                        })
                        .setNegativeButton("Cancel", null).show();
                }
            });
        }

        @JavascriptInterface
        public void getProjectFiles() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    refreshFileList();
                }
            });
        }

        @JavascriptInterface
        public String getDeviceInfo() {
            return "{\"sdk\":" + Build.VERSION.SDK_INT + ",\"release\":\"" + Build.VERSION.RELEASE + "\",\"model\":\"" + Build.MODEL + "\"}";
        }
    }
}