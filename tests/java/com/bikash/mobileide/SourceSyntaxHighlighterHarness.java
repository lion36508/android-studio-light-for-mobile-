package com.bikash.mobileide;

import java.util.List;

public final class SourceSyntaxHighlighterHarness {
    private static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    private static boolean containsToken(List<SourceSyntaxHighlighter.Token> tokens,
            CharSequence source, String value, SourceSyntaxHighlighter.Kind kind) {
        for (SourceSyntaxHighlighter.Token token : tokens) {
            if (token.kind == kind && source.subSequence(token.start, token.end).toString().equals(value)) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        String java = "String url = \"https://host/#frag\"; // real comment\n"
            + "String tag = \"#not-comment\";\nint count = 1; /* block comment */";
        List<SourceSyntaxHighlighter.Token> tokens =
            SourceSyntaxHighlighter.tokenize(java, "java");
        check(containsToken(tokens, java, "\"https://host/#frag\"", SourceSyntaxHighlighter.Kind.STRING),
            "URL comment markers inside a string must stay in the string token.");
        check(containsToken(tokens, java, "// real comment", SourceSyntaxHighlighter.Kind.COMMENT),
            "Real line comments must be recognized.");
        check(containsToken(tokens, java, "\"#not-comment\"", SourceSyntaxHighlighter.Kind.STRING),
            "Hash markers inside strings must stay in the string token.");
        check(containsToken(tokens, java, "/* block comment */", SourceSyntaxHighlighter.Kind.COMMENT),
            "Block comments must be recognized.");

        String cpp = "#include <iostream>\nconst char* url = \"https://host/path\";";
        List<SourceSyntaxHighlighter.Token> cppTokens =
            SourceSyntaxHighlighter.tokenize(cpp, "cpp");
        check(!containsToken(cppTokens, cpp, "#include <iostream>", SourceSyntaxHighlighter.Kind.COMMENT),
            "C/C++ preprocessor directives must not be treated as hash comments.");

        String python = "text = \"# not a comment\" # real comment";
        List<SourceSyntaxHighlighter.Token> pythonTokens =
            SourceSyntaxHighlighter.tokenize(python, "python");
        check(containsToken(pythonTokens, python, "# real comment", SourceSyntaxHighlighter.Kind.COMMENT),
            "Python hash comments must be recognized outside strings.");

        System.out.println("SourceSyntaxHighlighter runtime harness: all checks passed.");
    }
}
