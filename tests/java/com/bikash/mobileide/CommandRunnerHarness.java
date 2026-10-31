package com.bikash.mobileide;

import java.util.concurrent.atomic.AtomicReference;

public final class CommandRunnerHarness {
    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static CommandRunner.Result run(String command, long timeoutMs, int limit) {
        return CommandRunner.run(
            new ProcessBuilder("/bin/sh", "-c", command),
            timeoutMs,
            limit
        );
    }

    public static void main(String[] args) throws Exception {
        testSuccessAndExitCode();
        testSeparateStderrAndNonZeroExit();
        testSilentCommandTimeout();
        testConcurrentLargeOutputAndTruncation();
        testThreadInterruptionCancelsCommand();
        testLaunchFailureIsReported();
        System.out.println("CommandRunner runtime harness: all checks passed.");
    }

    private static void testSuccessAndExitCode() {
        CommandRunner.Result result = run("printf 'runner-ok'", 2000L, 4096);
        check(result.exitCode == 0, "Successful command must exit with code zero.");
        check("runner-ok".equals(result.stdout), "Stdout must be captured exactly.");
        check(!result.timedOut, "Successful command must not time out.");
    }

    private static void testSeparateStderrAndNonZeroExit() {
        CommandRunner.Result result = run(
            "printf 'stdout-value'; printf 'stderr-value' >&2; exit 7",
            2000L,
            4096
        );
        check(result.exitCode == 7, "Non-zero process exit code must be preserved.");
        check("stdout-value".equals(result.stdout), "Stdout must remain separate.");
        check("stderr-value".equals(result.stderr), "Stderr must remain separate.");
        check(result.toDisplayString().contains("code 7"), "Display output must expose exit status.");
    }

    private static void testSilentCommandTimeout() {
        long startedAt = System.nanoTime();
        CommandRunner.Result result = run("sleep 5", 150L, 2048);
        long elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L;
        check(result.timedOut, "Silent command must time out.");
        check(elapsedMs < 2500L, "Timeout must return near its deadline plus termination grace.");
    }

    private static void testConcurrentLargeOutputAndTruncation() {
        String command =
            "i=0; while [ \"$i\" -lt 30000 ]; do " +
            "printf '0123456789'; printf 'abcdefghij' >&2; " +
            "i=$((i + 1)); done";
        CommandRunner.Result result = run(command, 8000L, 2048);
        check(!result.timedOut, "Concurrent stdout/stderr drain must avoid pipe deadlock.");
        check(result.exitCode == 0, "Large-output command must finish successfully.");
        check(result.outputTruncated, "Large output must set the truncation flag.");
        check(result.stdout.length() <= 2048, "Stdout buffer must remain bounded.");
        check(result.stderr.length() <= 2048, "Stderr buffer must remain bounded.");
    }

    private static void testThreadInterruptionCancelsCommand() throws Exception {
        final AtomicReference<CommandRunner.Result> resultRef =
            new AtomicReference<CommandRunner.Result>();
        final AtomicReference<Boolean> interruptRestored =
            new AtomicReference<Boolean>(Boolean.FALSE);

        Thread worker = new Thread(new Runnable() {
            @Override
            public void run() {
                resultRef.set(run("sleep 8", 10000L, 2048));
                interruptRestored.set(Thread.currentThread().isInterrupted());
            }
        }, "command-runner-cancellation-test");

        worker.start();
        Thread.sleep(150L);
        worker.interrupt();
        worker.join(2500L);

        check(!worker.isAlive(), "Interrupted command runner must terminate promptly.");
        check(resultRef.get() != null && resultRef.get().cancelled,
            "Interrupted caller must receive a cancelled result.");
        check(Boolean.TRUE.equals(interruptRestored.get()),
            "Caller interrupt status must be restored after cleanup.");
    }

    private static void testLaunchFailureIsReported() {
        CommandRunner.Result result = CommandRunner.run(
            new ProcessBuilder("/oxdroid-test-command-that-does-not-exist"),
            1000L,
            1024
        );
        check(result.launchError != null && !result.launchError.isEmpty(),
            "Process launch errors must be returned as structured diagnostics.");
        check(result.toDisplayString().contains("[launch error]"),
            "Launch error must be visible in formatted output.");
    }
}
