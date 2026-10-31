package com.bikash.mobileide;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Bounded process execution shared by the IDE's trusted diagnostic and
 * terminal paths. Output is drained concurrently from both pipes while the
 * process deadline is active.
 */
public final class CommandRunner {
    public static final long DEFAULT_TIMEOUT_MS = 60_000L;
    public static final int DEFAULT_MAX_OUTPUT_CHARS = 64 * 1024;

    private static final long TERMINATION_GRACE_MS = 500L;
    private static final long READER_JOIN_MS = 1_500L;

    private CommandRunner() {}

    public static final class Result {
        public final String stdout;
        public final String stderr;
        public final int exitCode;
        public final boolean timedOut;
        public final boolean cancelled;
        public final boolean outputTruncated;
        public final boolean outputReadIncomplete;
        public final long durationMillis;
        public final String launchError;

        private Result(
            String stdout,
            String stderr,
            int exitCode,
            boolean timedOut,
            boolean cancelled,
            boolean outputTruncated,
            boolean outputReadIncomplete,
            long durationMillis,
            String launchError
        ) {
            this.stdout = stdout;
            this.stderr = stderr;
            this.exitCode = exitCode;
            this.timedOut = timedOut;
            this.cancelled = cancelled;
            this.outputTruncated = outputTruncated;
            this.outputReadIncomplete = outputReadIncomplete;
            this.durationMillis = durationMillis;
            this.launchError = launchError;
        }

        public String toDisplayString() {
            StringBuilder display = new StringBuilder();
            appendSection(display, stdout, null);
            appendSection(display, stderr, "[stderr]");
            if (launchError != null && !launchError.isEmpty()) {
                appendSection(display, launchError, "[launch error]");
            }
            if (timedOut) {
                appendSection(display, "Command timed out. Process termination was requested.", "[timeout]");
            } else if (cancelled) {
                appendSection(display, "Command cancelled because its calling thread was interrupted.", "[cancelled]");
            } else if (exitCode != 0 && exitCode != Integer.MIN_VALUE) {
                appendSection(display, "Process exited with code " + exitCode + ".", "[exit status]");
            }
            if (outputTruncated) {
                appendSection(display, "Output was truncated; the most recent characters were retained.", "[output limit]");
            }
            if (outputReadIncomplete) {
                appendSection(display, "An output stream did not close before the reader deadline.", "[reader warning]");
            }
            return display.toString().trim();
        }

        private static void appendSection(StringBuilder target, String value, String heading) {
            if (value == null || value.isEmpty()) return;
            if (target.length() > 0) target.append('\n');
            if (heading != null) target.append(heading).append('\n');
            target.append(value);
        }
    }

    private static final class StreamCapture {
        private final int limit;
        private final StringBuilder text = new StringBuilder();
        private boolean truncated;

        StreamCapture(int limit) {
            this.limit = limit;
        }

        synchronized void append(char[] buffer, int offset, int count) {
            if (count <= 0) return;
            if (count >= limit) {
                text.setLength(0);
                text.append(buffer, offset + count - limit, limit);
                truncated = true;
                return;
            }
            int overflow = text.length() + count - limit;
            if (overflow > 0) {
                text.delete(0, overflow);
                truncated = true;
            }
            text.append(buffer, offset, count);
        }

        synchronized String value() {
            return text.toString();
        }

        synchronized boolean wasTruncated() {
            return truncated;
        }
    }

    public static Result run(ProcessBuilder builder, long timeoutMillis, int maxOutputChars) {
        final long startedAt = System.nanoTime();
        final int outputLimit = maxOutputChars > 0
            ? Math.max(1024, maxOutputChars)
            : DEFAULT_MAX_OUTPUT_CHARS;
        final long safeTimeout = timeoutMillis > 0 ? timeoutMillis : DEFAULT_TIMEOUT_MS;

        final AtomicBoolean interrupted = new AtomicBoolean(false);
        final StreamCapture stdout = new StreamCapture(outputLimit);
        final StreamCapture stderr = new StreamCapture(outputLimit);

        Process process = null;
        InputStream stdoutStream = null;
        InputStream stderrStream = null;
        Thread stdoutReader = null;
        Thread stderrReader = null;
        boolean timedOut = false;
        boolean cancelled = false;
        boolean outputReadIncomplete = false;
        String launchError = null;

        try {
            if (builder == null) {
                throw new IllegalArgumentException("ProcessBuilder must not be null.");
            }

            builder.redirectErrorStream(false);
            process = builder.start();
            stdoutStream = process.getInputStream();
            stderrStream = process.getErrorStream();

            stdoutReader = startReader(stdoutStream, stdout, "oxdroid-process-stdout");
            stderrReader = startReader(stderrStream, stderr, "oxdroid-process-stderr");

            if (!waitForWithinDeadline(process, safeTimeout)) {
                timedOut = true;
                terminateProcess(process, interrupted);
            }
        } catch (InterruptedException e) {
            cancelled = true;
            interrupted.set(true);
            if (process != null) terminateProcess(process, interrupted);
        } catch (Exception e) {
            launchError = readableMessage(e);
            if (process != null) terminateProcess(process, interrupted);
        } finally {
            if (process != null && isRunning(process)) {
                terminateProcess(process, interrupted);
            }

            boolean stdoutStillRunning = joinReader(stdoutReader, READER_JOIN_MS, interrupted);
            boolean stderrStillRunning = joinReader(stderrReader, READER_JOIN_MS, interrupted);

            if (stdoutStillRunning) {
                outputReadIncomplete = true;
                closeQuietly(stdoutStream);
                if (stdoutReader != null) stdoutReader.interrupt();
                joinReader(stdoutReader, 250L, interrupted);
            }
            if (stderrStillRunning) {
                outputReadIncomplete = true;
                closeQuietly(stderrStream);
                if (stderrReader != null) stderrReader.interrupt();
                joinReader(stderrReader, 250L, interrupted);
            }
        }

        int exitCode = Integer.MIN_VALUE;
        if (process != null) {
            try {
                exitCode = process.exitValue();
            } catch (IllegalThreadStateException ignored) {
                // A platform process that resisted termination has no exit code yet.
            }
        }

        long durationMillis = Math.max(0L, (System.nanoTime() - startedAt) / 1_000_000L);
        Result result = new Result(
            stdout.value(),
            stderr.value(),
            exitCode,
            timedOut,
            cancelled,
            stdout.wasTruncated() || stderr.wasTruncated(),
            outputReadIncomplete,
            durationMillis,
            launchError
        );

        if (interrupted.get()) {
            Thread.currentThread().interrupt();
        }
        return result;
    }

    private static Thread startReader(
        final InputStream stream,
        final StreamCapture capture,
        String threadName
    ) {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                drain(stream, capture);
            }
        }, threadName);
        thread.setDaemon(true);
        thread.start();
        return thread;
    }

    private static void drain(InputStream stream, StreamCapture capture) {
        try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(stream, StandardCharsets.UTF_8)
        )) {
            char[] buffer = new char[4096];
            int count;
            while ((count = reader.read(buffer, 0, buffer.length)) != -1) {
                capture.append(buffer, 0, count);
            }
        } catch (IOException ignored) {
            // Closing a timed-out/cancelled process stream is an expected exit path.
        }
    }

    /**
     * Uses System.nanoTime and short polling intervals instead of the timed
     * Process.waitFor overload, which is unavailable on some supported Android
     * releases. The process output readers continue draining during this wait.
     */
    private static boolean waitForWithinDeadline(Process process, long timeoutMillis)
        throws InterruptedException {
        long budgetNanos = TimeUnit.MILLISECONDS.toNanos(Math.max(0L, timeoutMillis));
        long startedAt = System.nanoTime();
        while (isRunning(process)) {
            long remainingNanos = budgetNanos - (System.nanoTime() - startedAt);
            if (remainingNanos <= 0L) return !isRunning(process);
            long sleepMillis = Math.max(
                1L,
                Math.min(20L, TimeUnit.NANOSECONDS.toMillis(remainingNanos))
            );
            Thread.sleep(sleepMillis);
        }
        return true;
    }

    private static void terminateProcess(Process process, AtomicBoolean interrupted) {
        if (process == null || !isRunning(process)) return;

        destroyDescendantsIfSupported(process);
        process.destroy();
        try {
            if (!waitForWithinDeadline(process, TERMINATION_GRACE_MS)) {
                forceDestroy(process);
                waitForWithinDeadline(process, TERMINATION_GRACE_MS);
            }
        } catch (InterruptedException e) {
            interrupted.set(true);
            forceDestroy(process);
        }
    }

    /**
     * Uses reflection so older Android releases simply skip descendant cleanup.
     */
    private static void destroyDescendantsIfSupported(Process process) {
        Object stream = null;
        try {
            Class<?> processHandleClass = Class.forName("java.lang.ProcessHandle");
            Method toHandle = Process.class.getMethod("toHandle");
            Object handle = toHandle.invoke(process);
            Method descendantsMethod = processHandleClass.getMethod("descendants");
            stream = descendantsMethod.invoke(handle);

            Class<?> streamClass = Class.forName("java.util.stream.Stream");
            Iterator<?> children = (Iterator<?>) streamClass.getMethod("iterator").invoke(stream);
            Method destroyForcibly = processHandleClass.getMethod("destroyForcibly");
            while (children.hasNext()) {
                destroyForcibly.invoke(children.next());
            }
        } catch (Exception ignored) {
            // ProcessHandle is not available on every Android API level.
        } finally {
            if (stream != null) {
                try {
                    Class.forName("java.util.stream.Stream").getMethod("close").invoke(stream);
                } catch (Exception ignored) {}
            }
        }
    }

    private static void forceDestroy(Process process) {
        try {
            Method destroyForcibly = Process.class.getMethod("destroyForcibly");
            destroyForcibly.invoke(process);
        } catch (Exception ignored) {
            process.destroy();
        }
    }

    private static boolean isRunning(Process process) {
        try {
            process.exitValue();
            return false;
        } catch (IllegalThreadStateException e) {
            return true;
        }
    }

    private static boolean join(
        Thread thread,
        long timeoutMillis,
        AtomicBoolean interrupted
    ) {
        if (thread == null || !thread.isAlive()) return false;
        try {
            thread.join(timeoutMillis);
        } catch (InterruptedException e) {
            interrupted.set(true);
        }
        return thread.isAlive();
    }

    private static void closeQuietly(InputStream stream) {
        if (stream == null) return;
        try {
            stream.close();
        } catch (IOException ignored) {}
    }

    private static String readableMessage(Exception error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
            ? error.getClass().getSimpleName()
            : message;
    }
}
