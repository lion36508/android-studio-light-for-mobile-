(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

CodeMirror.defineMode("java", function() {

  // Keywords
  var keywords = {
    "abstract": "keyword",
    "assert": "keyword",
    "boolean": "keyword",
    "break": "keyword",
    "byte": "keyword",
    "case": "keyword",
    "catch": "keyword",
    "char": "keyword",
    "class": "keyword",
    "const": "keyword",
    "continue": "keyword",
    "default": "keyword",
    "do": "keyword",
    "double": "keyword",
    "else": "keyword",
    "enum": "keyword",
    "extends": "keyword",
    "final": "keyword",
    "finally": "keyword",
    "float": "keyword",
    "for": "keyword",
    "goto": "keyword",
    "if": "keyword",
    "implements": "keyword",
    "import": "keyword",
    "instanceof": "keyword",
    "int": "keyword",
    "interface": "keyword",
    "long": "keyword",
    "native": "keyword",
    "new": "keyword",
    "package": "keyword",
    "private": "keyword",
    "protected": "keyword",
    "public": "keyword",
    "return": "keyword",
    "short": "keyword",
    "static": "keyword",
    "strictfp": "keyword",
    "super": "keyword",
    "switch": "keyword",
    "synchronized": "keyword",
    "this": "keyword",
    "throw": "keyword",
    "throws": "keyword",
    "transient": "keyword",
    "try": "keyword",
    "void": "keyword",
    "volatile": "keyword",
    "while": "keyword"
  };

  // Operators
  var operators = {
    "+": "operator",
    "-": "operator",
    "*": "operator",
    "/": "operator",
    "%": "operator",
    "=": "operator",
    "+=": "operator",
    "-=": "operator",
    "*=": "operator",
    "/=": "operator",
    "%=": "operator",
    "==": "operator",
    "!=": "operator",
    ">": "operator",
    "<": "operator",
    ">=": "operator",
    "<=": "operator",
    "&&": "operator",
    "||": "operator",
    "!": "operator",
    "&": "operator",
    "|": "operator",
    "^": "operator",
    "~": "operator",
    "<<": "operator",
    ">>": "operator",
    ">>>": "operator",
    "++": "operator",
    "--": "operator"
  };

  // Punctuation
  var punctuation = {
    "(": "punctuation",
    ")": "punctuation",
    "{": "punctuation",
    "}": "punctuation",
    "[": "punctuation",
    "]": "punctuation",
    ";": "punctuation",
    ",": "punctuation",
    ".": "punctuation"
  };

  // Comments
  var comment = {
    "line": "//",
    "block": {
      "start": "/*",
      "end": "*/"
    }
  };

  // Strings
  var string = {
    "start": "\"",
    "end": "\"",
    "escape": "\\"
  };

  // Numbers
  var number = {
    "decimal": /\d+/,
    "hex": /0[xX][0-9a-fA-F]+/,
    "float": /\d+\.\d+/
  };

  // Annotations
  var annotation = {
    "start": "@",
    "end": /\s/
  };

  // Tokenizer
  function tokenize(stream, state) {
    // Check for comments
    if (stream.match(comment.line)) {
      stream.skipToEnd();
      return "comment";
    } else if (stream.match(comment.block.start)) {
      state.inBlockComment = true;
      return "comment";
    } else if (state.inBlockComment) {
      if (stream.match(comment.block.end)) {
        state.inBlockComment = false;
      } else {
        stream.skipToEnd();
      }
      return "comment";
    }

    // Check for strings
    if (stream.match(string.start)) {
      state.inString = true;
      return "string";
    } else if (state.inString) {
      if (stream.match(string.end)) {
        state.inString = false;
      } else if (stream.match(string.escape)) {
        stream.next();
      }
      return "string";
    }

    // Check for numbers
    if (stream.match(number.decimal) || stream.match(number.hex) || stream.match(number.float)) {
      return "number";
    }

    // Check for annotations
    if (stream.match(annotation.start)) {
      stream.match(annotation.end);
      return "annotation";
    }

    // Check for keywords and operators
    var word = stream.match(/[a-zA-Z_][a-zA-Z0-9_]*/);
    if (word) {
      if (keywords[word]) {
        return keywords[word];
      } else if (operators[word]) {
        return operators[word];
      }
    }

    // Check for punctuation
    var punct = stream.next();
    if (punctuation[punct]) {
      return punctuation[punct];
    }

    // Default
    stream.next();
    return null;
  }

  // Mode
  return {
    startState: function() {
      return {
        inBlockComment: false,
        inString: false
      };
    },
    token: function(stream, state) {
      return tokenize(stream, state);
    }
  };
});

CodeMirror.defineMIME("text/x-java", "java");

});