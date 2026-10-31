package com.bikash.mobileide;

public final class ProjectNameHarness {
    private static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    public static void main(String[] args) {
        check(ProjectName.validationError("My App") == null, "Ordinary display names must be accepted.");
        check(ProjectName.validationError("../escape") != null, "Forward slash must be rejected.");
        check(ProjectName.validationError("..\\\\escape") != null, "Backslash must be rejected.");
        check(ProjectName.validationError("Bad\\nName") != null, "Control characters must be rejected.");
        check(ProjectName.validationError(new String(new char[65]).replace('\\0', 'a')) != null,
            "Names longer than 64 characters must be rejected.");
        check("my-app".equals(ProjectName.fileStem("My App")), "File stem must be normalized.");
        check("app-9-demo".equals(ProjectName.fileStem("9 Demo")), "Numeric-leading names need a safe prefix.");
        check(ProjectName.javaClassName("my app").matches("[A-Za-z_$][A-Za-z0-9_$]*"),
            "Generated class name must be a Java identifier.");
        check("A\\\"B".equals(ProjectName.javaStringContent("A\\\"B")),
            "Quotes must be escaped for source string literals.");
        check("A\\\\B".equals(ProjectName.pythonStringContent("A\\\\B")),
            "Backslashes must be escaped for source string literals.");
        System.out.println("ProjectName runtime harness: all checks passed.");
    }
}
