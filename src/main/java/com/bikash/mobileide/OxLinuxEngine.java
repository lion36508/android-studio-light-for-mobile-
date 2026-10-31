package com.bikash.mobileide;

import android.content.Context;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class OxLinuxEngine {
    private final Context context;
    private final File rootDir;

    public OxLinuxEngine(Context ctx) {
        this.context = ctx;
        this.rootDir = new File(ctx.getFilesDir(), "oxlinux");
        extractIfNeeded();
    }

    private void extractIfNeeded() {
        File readyFlag = new File(rootDir, ".ready");
        if (readyFlag.exists()) return;
        try {
            if (!rootDir.exists()) rootDir.mkdirs();
            InputStream is = context.getAssets().open("linux_env/oxlinux_rootfs.bin");
            ZipInputStream zis = new ZipInputStream(is);
            ZipEntry entry;
            byte[] buffer = new byte[8192];
            File canonicalRoot = rootDir.getCanonicalFile();
            while ((entry = zis.getNextEntry()) != null) {
                File outFile = new File(canonicalRoot, entry.getName()).getCanonicalFile();
                if (!outFile.getPath().startsWith(canonicalRoot.getPath() + File.separator)) {
                    throw new SecurityException("Unsafe rootfs entry: " + entry.getName());
                }
                if (entry.isDirectory()) {
                    outFile.mkdirs();
                } else {
                    outFile.getParentFile().mkdirs();
                    FileOutputStream fos = new FileOutputStream(outFile);
                    int len;
                    while ((len = zis.read(buffer)) > 0) {
                        fos.write(buffer, 0, len);
                    }
                    fos.close();
                    if (outFile.getName().endsWith(".sh") || outFile.getParentFile().getName().equals("bin")) {
                        outFile.setExecutable(true, false);
                    }
                }
            }
            zis.close();
            readyFlag.createNewFile();
        } catch (Exception ignored) {}
    }

    public CommandRunner.Result execResult(String cmd, long timeoutMillis, int maxOutputChars) {
        ProcessBuilder builder = new ProcessBuilder(
            "/system/bin/sh", "-c", cmd == null ? "" : cmd
        );
        builder.directory(rootDir);
        Map<String, String> env = builder.environment();
        env.put("HOME", rootDir.getAbsolutePath());
        env.put("PATH", new File(rootDir, "bin").getAbsolutePath() + ":/system/bin:/system/xbin");
        return CommandRunner.run(builder, timeoutMillis, maxOutputChars);
    }

    public String exec(String cmd) {
        return execResult(
            cmd,
            CommandRunner.DEFAULT_TIMEOUT_MS,
            CommandRunner.DEFAULT_MAX_OUTPUT_CHARS
        ).toDisplayString();
    }
}
