define("ace/ext/themelist",["require","exports","module"], function(require, exports, module){
/**
 * Generates a list of themes available when ace was built.
 * @fileOverview Generates a list of themes available when ace was built.
 * @author <a href="mailto:matthewkastor@gmail.com">
 *  Matthew Christopher Kastor-Inare III </a><br />
 *  ☭ Hial Atropa!! ☭
 */
"use strict";
var themeData = [
    ["Chrome"],
    ["Clouds"],
    ["Crimson Editor"],
    ["Dawn"],
    ["Dreamweaver"],
    ["Eclipse"],
    ["GitHub"],
    ["IPlastic"],
    ["Solarized Light"],
    ["TextMate"],
    ["Tomorrow"],
    ["XCode"],
    ["Kuroir"],
    ["KatzenMilch"],
    ["SQL Server", "sqlserver", "light"],
    ["CloudEditor", "cloud_editor", "light"],
    ["Ambiance", "ambiance", "dark"],
    ["Chaos", "chaos", "dark"],
    ["Clouds Midnight", "clouds_midnight", "dark"],
    ["Dracula", "", "dark"],
    ["Cobalt", "cobalt", "dark"],
    ["Gruvbox", "gruvbox", "dark"],
    ["Green on Black", "gob", "dark"],
    ["idle Fingers", "idle_fingers", "dark"],
    ["krTheme", "kr_theme", "dark"],
    ["Merbivore", "merbivore", "dark"],
    ["Merbivore Soft", "merbivore_soft", "dark"],
    ["Mono Industrial", "mono_industrial", "dark"],
    ["Monokai", "monokai", "dark"],
    ["Nord Dark", "nord_dark", "dark"],
    ["One Dark", "one_dark", "dark"],
    ["Pastel on dark", "pastel_on_dark", "dark"],
    ["Solarized Dark", "solarized_dark", "dark"],
    ["Terminal", "terminal", "dark"],
    ["Tomorrow Night", "tomorrow_night", "dark"],
    ["Tomorrow Night Blue", "tomorrow_night_blue", "dark"],
    ["Tomorrow Night Bright", "tomorrow_night_bright", "dark"],
    ["Tomorrow Night 80s", "tomorrow_night_eighties", "dark"],
    ["Twilight", "twilight", "dark"],
    ["Vibrant Ink", "vibrant_ink", "dark"],
    ["GitHub Dark", "github_dark", "dark"],
    ["CloudEditor Dark", "cloud_editor_dark", "dark"]
];
exports.themesByName = {};
exports.themes = themeData.map(function (data) {
    var name = data[1] || data[0].replace(/ /g, "_").toLowerCase();
    var theme = {
        caption: data[0],
        theme: "ace/theme/" + name,
        isDark: data[2] == "dark",
        name: name
    };
    exports.themesByName[name] = theme;
    return theme;
});

exports.previewTheme = function(editor, themeName) {
    var theme = exports.themesByName[themeName];
    if (theme) {
        editor.container.style.transition = "background-color 0.5s ease, color 0.5s ease";
        editor.setTheme(theme.theme);
        editor.container.style.backgroundColor = theme.isDark ? "#272822" : "#f8f8f8";
        editor.container.style.color = theme.isDark ? "#f8f8f8" : "#272822";
        localStorage.setItem('selectedTheme', themeName);
    }
};

exports.createThemePreview = function(themeName) {
    var theme = exports.themesByName[themeName];
    if (!theme) return null;

    var preview = document.createElement('div');
    preview.className = 'theme-preview';
    preview.style.transition = "background-color 0.5s ease, color 0.5s ease";
    preview.style.backgroundColor = theme.isDark ? '#272822' : '#f8f8f8';
    preview.style.color = theme.isDark ? '#f8f8f8' : '#272822';
    preview.style.padding = '10px';
    preview.style.borderRadius = '5px';
    preview.style.marginBottom = '10px';

    var title = document.createElement('h3');
    title.textContent = theme.caption;
    preview.appendChild(title);

    var code = document.createElement('pre');
    code.textContent = 'function example() {\n    // This is an example function\n    var exampleVar = "Hello, World!";\n    console.log(exampleVar);\n}';
    preview.appendChild(code);

    return preview;
};

exports.createLivePreview = function(editor, themeName) {
    var theme = exports.themesByName[themeName];
    if (!theme) return null;

    var livePreview = document.createElement('div');
    livePreview.className = 'live-preview';
    livePreview.style.transition = "background-color 0.5s ease, color 0.5s ease";
    livePreview.style.backgroundColor = theme.isDark ? '#272822' : '#f8f8f8';
    livePreview.style.color = theme.isDark ? '#f8f8f8' : '#272822';
    livePreview.style.padding = '10px';
    livePreview.style.borderRadius = '5px';
    livePreview.style.marginBottom = '10px';
    livePreview.style.height = '200px';
    livePreview.style.overflow = 'auto';

    var title = document.createElement('h3');
    title.textContent = 'Live Preview: ' + theme.caption;
    livePreview.appendChild(title);

    var previewEditor = ace.edit(livePreview);
    previewEditor.setTheme(theme.theme);
    previewEditor.setValue(editor.getValue(), -1);
    previewEditor.setReadOnly(true);
    previewEditor.setOptions({
        maxLines: Infinity,
        minLines: 10,
        showLineNumbers: true,
        showGutter: true,
        highlightActiveLine: false,
        highlightGutterLine: false
    });

    return livePreview;
};

exports.loadTheme = function(editor) {
    var themeName = localStorage.getItem('selectedTheme');
    if (themeName && exports.themesByName[themeName]) {
        exports.previewTheme(editor, themeName);
    }
};

});                (function() {
                    window.require(["ace/ext/themelist"], function(m) {
                        if (typeof module == "object" && typeof exports == "object" && module) {
                            module.exports = m;
                        }
                    });
                })();