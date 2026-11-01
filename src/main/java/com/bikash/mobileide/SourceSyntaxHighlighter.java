package com.bikash.mobileide;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Small, allocation-bounded lexer for editor coloring; tokens never overlap. */
public final class SourceSyntaxHighlighter {
    public enum Kind { KEYWORD, STRING, COMMENT }

    public static final class Token {
        public final int start;
        public final int end;
        public final Kind kind;

        private Token(int start, int end, Kind kind) {
            this.start = start;
            this.end = end;
            this.kind = kind;
        }
    }

    private static final Set<String> KEYWORDS = new HashSet<String>(Arrays.asList(
        "public", "private", "protected", "class", "static", "void", "int", "long",
        "double", "float", "String", "boolean", "fun", "val", "var", "def", "for",
        "in", "range", "while", "if", "else", "return", "import", "package",
        "include", "using", "namespace", "std", "cout", "endl", "print", "println",
        "const", "new", "this", "super", "extends", "implements", "try", "catch",
        "finally", "throw", "throws", "public", "async", "await", "let", "const"
    ));

    private SourceSyntaxHighlighter() {}

    public static List<Token> tokenize(CharSequence source, String language) {
        List<Token> tokens = new ArrayList<Token>();
        if (source == null || source.length() == 0) return tokens;

        boolean hashComments = "python".equalsIgnoreCase(language)
            || "shell".equalsIgnoreCase(language);
        int length = source.length();
        int i = 0;

        while (i < length) {
            char c = source.charAt(i);
            if (c == '"' || c == '\'') {
                int end = scanQuoted(source, i, c);
                tokens.add(new Token(i, end, Kind.STRING));
                i = end;
                continue;
            }

            if (c == '/' && i + 1 < length && source.charAt(i + 1) == '/') {
                int end = scanLine(source, i + 2);
                tokens.add(new Token(i, end, Kind.COMMENT));
                i = end;
                continue;
            }

            if (c == '/' && i + 1 < length && source.charAt(i + 1) == '*') {
                int end = scanBlockComment(source, i + 2);
                tokens.add(new Token(i, end, Kind.COMMENT));
                i = end;
                continue;
            }

            if (hashComments && c == '#') {
                int end = scanLine(source, i + 1);
                tokens.add(new Token(i, end, Kind.COMMENT));
                i = end;
                continue;
            }

            if (isIdentifierStart(c)) {
                int end = i + 1;
                while (end < length && isIdentifierPart(source.charAt(end))) end++;
                String word = source.subSequence(i, end).toString();
                if (KEYWORDS.contains(word)) tokens.add(new Token(i, end, Kind.KEYWORD));
                i = end;
                continue;
            }
            i++;
        }
        return tokens;
    }

    private static int scanQuoted(CharSequence source, int start, char quote) {
        int i = start + 1;
        while (i < source.length()) {
            char c = source.charAt(i++);
            if (c == '\\' && i < source.length()) {
                i++;
            } else if (c == quote) {
                break;
            } else if (c == '\n' || c == '\r') {
                break;
            }
        }
        return i;
    }

    private static int scanLine(CharSequence source, int start) {
        int i = start;
        while (i < source.length() && source.charAt(i) != '\n' && source.charAt(i) != '\r') i++;
        return i;
    }

    private static int scanBlockComment(CharSequence source, int start) {
        int i = start;
        while (i + 1 < source.length()) {
            if (source.charAt(i) == '*' && source.charAt(i + 1) == '/') return i + 2;
            i++;
        }
        return source.length();
    }

    private static boolean isIdentifierStart(char c) {
        return Character.isLetter(c) || c == '_' || c == '$';
    }

    private static boolean isIdentifierPart(char c) {
        return Character.isLetterOrDigit(c) || c == '_' || c == '$';
    }
}
