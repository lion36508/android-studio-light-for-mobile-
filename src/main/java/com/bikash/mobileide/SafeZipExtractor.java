package com.bikash.mobileide;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Extracts a ZIP archive into a sibling staging directory, then promotes it
 * with same-filesystem directory renames. Archive paths and expanded sizes
 * are validated before any file is written outside the staging directory.
 */
public final class SafeZipExtractor {
    public static final int DEFAULT_MAX_ENTRIES = 20_000;
    public static final long DEFAULT_MAX_ARCHIVE_BYTES = 128L * 1024L * 1024L;
    public static final long DEFAULT_MAX_ENTRY_BYTES = 128L * 1024L * 1024L;
    public static final long DEFAULT_MAX_TOTAL_BYTES = 512L * 1024L * 1024L;
    public static final int DEFAULT_MAX_COMPRESSION_RATIO = 1000;
    public static final int DEFAULT_MAX_PATH_LENGTH = 1024;
    public static final int DEFAULT_MAX_PATH_DEPTH = 64;

    private SafeZipExtractor() {}

    public static final class Limits {
        public final int maxEntries;
        public final long maxArchiveBytes;
        public final long maxEntryBytes;
        public final long maxTotalBytes;
        public final int maxCompressionRatio;
        public final int maxPathLength;
        public final int maxPathDepth;

        public Limits(
            int maxEntries,
            long maxArchiveBytes,
            long maxEntryBytes,
            long maxTotalBytes,
            int maxCompressionRatio,
            int maxPathLength,
            int maxPathDepth
        ) {
            if (maxEntries < 1 || maxArchiveBytes < 1L || maxEntryBytes < 1L
                || maxTotalBytes < 1L || maxCompressionRatio < 1
                || maxPathLength < 1 || maxPathDepth < 1) {
                throw new IllegalArgumentException("All ZIP extraction limits must be positive.");
            }
            this.maxEntries = maxEntries;
            this.maxArchiveBytes = maxArchiveBytes;
            this.maxEntryBytes = maxEntryBytes;
            this.maxTotalBytes = maxTotalBytes;
            this.maxCompressionRatio = maxCompressionRatio;
            this.maxPathLength = maxPathLength;
            this.maxPathDepth = maxPathDepth;
        }

        public static Limits defaults() {
            return new Limits(
                DEFAULT_MAX_ENTRIES,
                DEFAULT_MAX_ARCHIVE_BYTES,
                DEFAULT_MAX_ENTRY_BYTES,
                DEFAULT_MAX_TOTAL_BYTES,
                DEFAULT_MAX_COMPRESSION_RATIO,
                DEFAULT_MAX_PATH_LENGTH,
                DEFAULT_MAX_PATH_DEPTH
            );
        }
    }

    public static final class Result {
        public final boolean success;
        public final int entriesProcessed;
        public final long bytesExtracted;
        public final String message;

        private Result(boolean success, int entriesProcessed, long bytesExtracted, String message) {
            this.success = success;
            this.entriesProcessed = entriesProcessed;
            this.bytesExtracted = bytesExtracted;
            this.message = message;
        }
    }

    public static boolean isReady(File rootDirectory) {
        if (rootDirectory == null || !rootDirectory.isDirectory()) return false;
        File marker = new File(rootDirectory, ".ready");
        return marker.isFile() && marker.length() > 0L;
    }

    public static Result extract(InputStream source, File targetRoot) {
        return extract(source, targetRoot, Limits.defaults());
    }

    /**
     * The source stream is closed by this method, whether extraction succeeds or fails.
     */
    public static Result extract(InputStream source, File targetRoot, Limits limits) {
        int entriesProcessed = 0;
        long bytesExtracted = 0L;
        File archiveTemp = null;
        File staging = null;
        File backup = null;
        boolean backupCreated = false;
        String message = null;

        if (source == null) {
            return failure(entriesProcessed, bytesExtracted, "Rootfs archive stream is unavailable.");
        }
        if (targetRoot == null || limits == null) {
            closeQuietly(source);
            return failure(entriesProcessed, bytesExtracted, "Extraction target and limits are required.");
        }

        File absoluteTarget = targetRoot.getAbsoluteFile();
        File parent = absoluteTarget.getParentFile();
        if (parent == null) {
            closeQuietly(source);
            return failure(entriesProcessed, bytesExtracted, "Rootfs target has no parent directory.");
        }

        try {
            ensureDirectory(parent);
            String token = UUID.randomUUID().toString();
            String prefix = "." + absoluteTarget.getName() + "-" + token;
            archiveTemp = new File(parent, prefix + ".archive.tmp");
            staging = new File(parent, prefix + ".staging");
            backup = new File(parent, prefix + ".backup");

            copyArchiveBounded(source, archiveTemp, limits.maxArchiveBytes);
            ensureNewDirectory(staging);

            File canonicalStage = staging.getCanonicalFile();
            String stagePrefix = canonicalStage.getPath() + File.separator;
            Set<String> seenPaths = new HashSet<String>();

            try (ZipFile zip = new ZipFile(archiveTemp)) {
                Enumeration<? extends ZipEntry> entries = zip.entries();
                while (entries.hasMoreElements()) {
                    ZipEntry entry = entries.nextElement();
                    entriesProcessed++;
                    if (entriesProcessed > limits.maxEntries) {
                        throw new IOException("Archive exceeded the entry-count limit of " + limits.maxEntries + ".");
                    }

                    String entryName = entry.getName();
                    validateEntryName(entryName, limits);
                    File output = new File(canonicalStage, entryName).getCanonicalFile();
                    String outputPath = output.getPath();

                    if (!outputPath.startsWith(stagePrefix)) {
                        throw new IOException("Archive entry escaped the staging directory.");
                    }
                    if (output.equals(canonicalStage)) {
                        throw new IOException("Archive entry may not overwrite the staging root.");
                    }
                    if (".ready".equals(entryName)) {
                        throw new IOException("Archive entry uses the reserved readiness marker name.");
                    }

                    boolean directoryEntry = entry.isDirectory();
                    if (!seenPaths.add(outputPath)) {
                        if (directoryEntry && output.isDirectory()) continue;
                        throw new IOException("Archive contains a duplicate entry path: " + entryName);
                    }

                    if (directoryEntry) {
                        ensureDirectory(output);
                        continue;
                    }

                    long expectedSize = entry.getSize();
                    long compressedSize = entry.getCompressedSize();
                    if (expectedSize < 0L || compressedSize < 0L) {
                        throw new IOException("Archive entry has an unknown size: " + entryName);
                    }
                    if (expectedSize > limits.maxEntryBytes) {
                        throw new IOException("Archive entry exceeds the per-file size limit: " + entryName);
                    }
                    if (expectedSize > 0L && compressedSize == 0L) {
                        throw new IOException("Archive entry has an invalid compressed size: " + entryName);
                    }
                    if (compressedSize > 0L
                        && expectedSize > compressedSize * (long) limits.maxCompressionRatio) {
                        throw new IOException("Archive entry exceeds the compression-ratio limit: " + entryName);
                    }

                    File outputParent = output.getParentFile();
                    if (outputParent == null) {
                        throw new IOException("Archive entry has no parent directory: " + entryName);
                    }
                    ensureDirectory(outputParent);
                    if (output.exists()) {
                        throw new IOException("Archive entry would overwrite an existing path: " + entryName);
                    }

                    long entryBytes = 0L;
                    try (
                        InputStream input = new BufferedInputStream(zip.getInputStream(entry));
                        BufferedOutputStream fileOutput = new BufferedOutputStream(new FileOutputStream(output))
                    ) {
                        byte[] buffer = new byte[32 * 1024];
                        int count;
                        while ((count = input.read(buffer)) != -1) {
                            if (entryBytes > limits.maxEntryBytes - count) {
                                throw new IOException("Archive entry expanded beyond the per-file limit: " + entryName);
                            }
                            if (bytesExtracted > limits.maxTotalBytes - count) {
                                throw new IOException("Archive expanded beyond the total extraction limit.");
                            }
                            fileOutput.write(buffer, 0, count);
                            entryBytes += count;
                            bytesExtracted += count;
                        }
                    }

                    if (entryBytes != expectedSize) {
                        throw new IOException("Archive entry size did not match its directory record: " + entryName);
                    }

                    if (entryName.endsWith(".sh") || entryName.startsWith("bin/")) {
                        output.setExecutable(true, false);
                    }
                    if (entry.getTime() > 0L) {
                        output.setLastModified(entry.getTime());
                    }
                }
            }

            writeReadyMarker(staging, entriesProcessed, bytesExtracted);

            if (absoluteTarget.exists()) {
                if (!absoluteTarget.renameTo(backup)) {
                    throw new IOException("Could not move the existing rootfs aside for safe replacement.");
                }
                backupCreated = true;
            }

            if (!staging.renameTo(absoluteTarget)) {
                throw new IOException("Could not atomically promote the extracted rootfs.");
            }

            String successMessage = "Extracted " + entriesProcessed + " entries (" + bytesExtracted + " bytes).";
            if (backupCreated) {
                if (deleteRecursivelySafely(backup)) {
                    backupCreated = false;
                } else {
                    successMessage += " Previous rootfs backup remains for manual cleanup.";
                }
            }
            return new Result(true, entriesProcessed, bytesExtracted, successMessage);
        } catch (Exception error) {
            message = readableMessage(error);
            if (backupCreated && !absoluteTarget.exists() && backup != null && backup.exists()) {
                if (backup.renameTo(absoluteTarget)) {
                    backupCreated = false;
                } else {
                    message += " The previous rootfs backup could not be restored and remains at "
                        + backup.getAbsolutePath() + ".";
                }
            }
            return failure(entriesProcessed, bytesExtracted, message);
        } finally {
            closeQuietly(source);
            if (archiveTemp != null) deleteRecursivelySafely(archiveTemp);
            if (staging != null) deleteRecursivelySafely(staging);
            if (backupCreated && absoluteTarget.exists() && backup != null) {
                deleteRecursivelySafely(backup);
            }
        }
    }

    private static void copyArchiveBounded(InputStream source, File destination, long maxBytes)
        throws IOException {
        long copied = 0L;
        try (
            InputStream input = new BufferedInputStream(source);
            FileOutputStream output = new FileOutputStream(destination)
        ) {
            byte[] buffer = new byte[32 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                if (copied > maxBytes - count) {
                    throw new IOException("Compressed rootfs archive exceeded the archive-size limit.");
                }
                output.write(buffer, 0, count);
                copied += count;
            }
            output.flush();
            output.getFD().sync();
        }
    }

    private static void validateEntryName(String name, Limits limits) throws IOException {
        if (name == null || name.isEmpty() || name.length() > limits.maxPathLength) {
            throw new IOException("Archive contains an empty or overlong entry path.");
        }
        if (name.startsWith("/") || name.startsWith("\\")
            || name.indexOf('\\') >= 0 || name.indexOf('\0') >= 0) {
            throw new IOException("Archive contains an absolute or invalid entry path.");
        }

        String normalized = name.endsWith("/") ? name.substring(0, name.length() - 1) : name;
        if (normalized.isEmpty()) {
            throw new IOException("Archive contains an empty entry path.");
        }
        String[] segments = normalized.split("/", -1);
        if (segments.length > limits.maxPathDepth) {
            throw new IOException("Archive entry exceeds the maximum directory depth.");
        }
        for (String segment : segments) {
            if (segment.isEmpty() || ".".equals(segment) || "..".equals(segment)) {
                throw new IOException("Archive contains a traversal or ambiguous path segment.");
            }
        }
    }

    private static void writeReadyMarker(File staging, int entries, long bytes) throws IOException {
        File marker = new File(staging, ".ready");
        try (FileOutputStream output = new FileOutputStream(marker)) {
            String content = "format=1\nentries=" + entries + "\nbytes=" + bytes + "\n";
            output.write(content.getBytes(StandardCharsets.UTF_8));
            output.flush();
            output.getFD().sync();
        }
    }

    private static void ensureNewDirectory(File directory) throws IOException {
        if (directory.exists()) {
            throw new IOException("A unique staging directory unexpectedly already exists.");
        }
        ensureDirectory(directory);
    }

    private static void ensureDirectory(File directory) throws IOException {
        if (directory == null) throw new IOException("Directory path is invalid.");
        if (directory.isDirectory()) return;
        if (directory.exists() || !directory.mkdirs() || !directory.isDirectory()) {
            throw new IOException("Could not create directory: " + directory.getAbsolutePath());
        }
    }

    private static boolean deleteRecursivelySafely(File file) {
        if (file == null || !file.exists()) return true;
        try {
            // Do not recurse through a symbolic link in an older, possibly partial rootfs.
            if (!file.getCanonicalFile().equals(file.getAbsoluteFile())) {
                return file.delete();
            }
        } catch (IOException ignored) {
            return false;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children == null) return false;
            for (File child : children) {
                if (!deleteRecursivelySafely(child)) return false;
            }
        }
        return file.delete();
    }

    private static void closeQuietly(InputStream input) {
        if (input == null) return;
        try {
            input.close();
        } catch (IOException ignored) {}
    }

    private static Result failure(int entries, long bytes, String message) {
        return new Result(false, entries, bytes, message == null ? "Rootfs extraction failed." : message);
    }

    private static String readableMessage(Exception error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
            ? error.getClass().getSimpleName()
            : message;
    }
}
