package com.bikash.mobileide;

import java.util.Locale;

/** Validates user-facing project names and derives safe source/package identifiers. */
public final class ProjectName {
    private ProjectName() {}

    /** Returns null when valid, otherwise a user-readable validation error. */
    public static String validationError(String rawName) {
        if (rawName == null || rawName.trim().isEmpty()) return "Enter a project name.";
        String name = rawName.trim();
        if (name.length() > 64) return "Project name must be 64 characters or fewer.";
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isISOControl(c)) {
                return "Project name cannot contain control characters or line breaks.";
            }
            if (c == '/' || c == '\\') {
                return "Project name cannot contain path separators (/ or \\).";
            }
        }
        return null;
    }

    /** Stable, filesystem-safe ASCII stem; display names are kept separately. */
    public static String fileStem(String displayName) {
        String slug = displayName == null ? "" : displayName.trim().toLowerCase(Locale.US);
        slug = slug.replaceAll("[^a-z0-9]+", "-").replaceAll("^-+|-+$", "");
        if (slug.isEmpty()) slug = "project";
        if (Character.isDigit(slug.charAt(0))) slug = "app-" + slug;
        if (slug.length() > 48) slug = slug.substring(0, 48).replaceAll("-+$", "");
        return slug.isEmpty() ? "project" : slug;
    }

    /** Produces a valid Java class identifier from the safe stem. */
    public static String javaClassName(String displayName) {
        String[] parts = fileStem(displayName).split("-");
        StringBuilder result = new StringBuilder();
        for (String part : parts) {
            if (part.isEmpty()) continue;
            result.append(Character.toUpperCase(part.charAt(0)));
            if (part.length() > 1) result.append(part.substring(1));
        }
        if (result.length() == 0) return "OxProject";
        if (!Character.isJavaIdentifierStart(result.charAt(0))) result.insert(0, "App");
        return result.toString();
    }

    public static String javaStringContent(String value) {
        return escapeQuotedContent(value);
    }

    public static String pythonStringContent(String value) {
        return escapeQuotedContent(value);
    }

    private static String escapeQuotedContent(String value) {
        if (value == null) return "";
        StringBuilder escaped = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '\\': escaped.append("\\\\"); break;
                case '"': escaped.append("\\\""); break;
                case '\n': escaped.append("\\n"); break;
                case '\r': escaped.append("\\r"); break;
                case '\t': escaped.append("\\t"); break;
                default:
                    if (Character.isISOControl(c)) {
                        escaped.append(String.format(Locale.US, "\\u%04x", (int) c));
                    } else {
                        escaped.append(c);
                    }
            }
        }
        return escaped.toString();
    }
}
