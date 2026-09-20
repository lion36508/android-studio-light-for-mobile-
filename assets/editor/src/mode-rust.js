// Rust mode for CodeMirror
(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

CodeMirror.defineMode("rust", function() {

  // Keywords
  var keywords = {
    "as": "keyword",
    "break": "keyword",
    "const": "keyword",
    "continue": "keyword",
    "crate": "keyword",
    "else": "keyword",
    "enum": "keyword",
    "extern": "keyword",
    "false": "keyword",
    "fn": "keyword",
    "for": "keyword",
    "if": "keyword",
    "impl": "keyword",
    "in": "keyword",
    "let": "keyword",
    "loop": "keyword",
    "match": "keyword",
    "mod": "keyword",
    "move": "keyword",
    "mut": "keyword",
    "pub": "keyword",
    "ref": "keyword",
    "return": "keyword",
    "self": "keyword",
    "Self": "keyword",
    "static": "keyword",
    "struct": "keyword",
    "super": "keyword",
    "trait": "keyword",
    "true": "keyword",
    "type": "keyword",
    "unsafe": "keyword",
    "use": "keyword",
    "where": "keyword",
    "while": "keyword",
    "dyn": "keyword",
    "abstract": "keyword",
    "become": "keyword",
    "box": "keyword",
    "do": "keyword",
    "final": "keyword",
    "macro": "keyword",
    "override": "keyword",
    "priv": "keyword",
    "typeof": "keyword",
    "unsized": "keyword",
    "virtual": "keyword",
    "yield": "keyword"
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
    "&": "operator",
    "|": "operator",
    "^": "operator",
    "!": "operator",
    "&&": "operator",
    "||": "operator",
    "<<": "operator",
    ">>": "operator",
    ">>>": "operator",
    "~": "operator",
    "..": "operator",
    "...": "operator",
    "..=": "operator",
    "=>": "operator",
    "@": "operator",
    "#": "operator",
    "$": "operator",
    "?": "operator"
  };

  // Delimiters
  var delimiters = {
    "(": "bracket",
    ")": "bracket",
    "[": "bracket",
    "]": "bracket",
    "{": "bracket",
    "}": "bracket",
    ",": "punctuation",
    ";": "punctuation",
    ":": "punctuation",
    "->": "punctuation"
  };

  // Comments
  function tokenComment(stream, state) {
    if (stream.match(/^\/\//)) {
      stream.skipToEnd();
      return "comment";
    } else if (stream.match(/^\/\*/)) {
      state.tokenize = tokenCommentBlock;
      return tokenCommentBlock(stream, state);
    }
    return null;
  }

  function tokenCommentBlock(stream, state) {
    if (stream.match(/^.*?\*\//)) {
      state.tokenize = null;
      return "comment";
    }
    stream.skipToEnd();
    return "comment";
  }

  // Strings
  function tokenString(stream, state) {
    if (stream.match(/^"/)) {
      state.tokenize = null;
      return "string";
    }
    if (stream.match(/^\\./)) {
      return "string";
    }
    stream.next();
    return "string";
  }

  // Numbers
  function tokenNumber(stream, state) {
    if (stream.match(/^[0-9]+/)) {
      if (stream.match(/^[uif]/)) {
        return "number";
      }
      return "number";
    }
    return null;
  }

  // Identifiers
  function tokenIdentifier(stream, state) {
    if (stream.match(/^[a-zA-Z_][a-zA-Z0-9_]*/)) {
      var word = stream.current();
      if (keywords.hasOwnProperty(word)) {
        return keywords[word];
      }
      return "variable";
    }
    return null;
  }

  // Main tokenizer
  function tokenBase(stream, state) {
    if (stream.eatSpace()) {
      return null;
    }

    var token = tokenComment(stream, state) ||
                tokenString(stream, state) ||
                tokenNumber(stream, state) ||
                tokenIdentifier(stream, state);

    if (token) {
      return token;
    }

    var ch = stream.next();
    if (operators.hasOwnProperty(ch)) {
      return operators[ch];
    } else if (delimiters.hasOwnProperty(ch)) {
      return delimiters[ch];
    }

    return null;
  }

  return {
    startState: function() {
      return {
        tokenize: null
      };
    },

    token: function(stream, state) {
      if (state.tokenize) {
        return state.tokenize(stream, state);
      }
      return tokenBase(stream, state);
    },

    blockCommentStart: "/*",
    blockCommentEnd: "*/",
    lineComment: "//"
  };
});

CodeMirror.defineMIME("text/x-rustsrc", "rust");
});