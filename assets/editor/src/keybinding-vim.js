define("ace/ext/hardwrap",["require","exports","module","ace/range","ace/editor","ace/config"], function(require, exports, module){"use strict";
var Range = require("../range").Range;
function hardWrap(editor, options) {
    var max = options.column || editor.getOption("printMarginColumn");
    var allowMerge = options.allowMerge != false;
    var row = Math.min(options.startRow, options.endRow);
    var endRow = Math.max(options.startRow, options.endRow);
    var session = editor.session;
    while (row <= endRow) {
        var line = session.getLine(row);
        if (line.length > max) {
            var space = findSpace(line, max, 5);
            if (space) {
                var indentation = /^\s*/.exec(line)[0];
                session.replace(new Range(row, space.start, row, space.end), "\n" + indentation);
            }
            endRow++;
        }
        else if (allowMerge && /\S/.test(line) && row != endRow) {
            var nextLine = session.getLine(row + 1);
            if (nextLine && /\S/.test(nextLine)) {
                var trimmedLine = line.replace(/\s+$/, "");
                var trimmedNextLine = nextLine.replace(/^\s+/, "");
                var mergedLine = trimmedLine + " " + trimmedNextLine;
                var space = findSpace(mergedLine, max, 5);
                if (space && space.start > trimmedLine.length || mergedLine.length < max) {
                    var replaceRange = new Range(row, trimmedLine.length, row + 1, nextLine.length - trimmedNextLine.length);
                    session.replace(replaceRange, " ");
                    row--;
                    endRow--;
                }
                else if (trimmedLine.length < line.length) {
                    session.remove(new Range(row, trimmedLine.length, row, line.length));
                }
            }
        }
        row++;
    }
    function findSpace(line, max, min) {
        if (line.length < max)
            return;
        var before = line.slice(0, max);
        var after = line.slice(max);
        var spaceAfter = /^(?:(\s+)|(\S+)(\s+))/.exec(after);
        var spaceBefore = /(?:(\s+)|(\s+)(\S+))$/.exec(before);
        var start = 0;
        var end = 0;
        if (spaceBefore && !spaceBefore[2]) {
            start = max - spaceBefore[1].length;
            end = max;
        }
        if (spaceAfter && !spaceAfter[2]) {
            if (!start)
                start = max;
            end = max + spaceAfter[1].length;
        }
        if (start) {
            return {
                start: start,
                end: end
            };
        }
        if (spaceBefore && spaceBefore[2] && spaceBefore.index > min) {
            return {
                start: spaceBefore.index,
                end: spaceBefore.index + spaceBefore[2].length
            };
        }
        if (spaceAfter && spaceAfter[2]) {
            start = max + spaceAfter[2].length;
            return {
                start: start,
                end: start + spaceAfter[3].length
            };
        }
    }
}
function wrapAfterInput(e) {
    if (e.command.name == "insertstring" && /\S/.test(e.args)) {
        var editor = e.editor;
        var cursor = editor.selection.cursor;
        if (cursor.column <= editor.renderer.$printMarginColumn)
            return;
        var lastDelta = editor.session.$undoManager.$lastDelta;
        hardWrap(editor, {
            startRow: cursor.row, endRow: cursor.row,
            allowMerge: false
        });
        if (lastDelta != editor.session.$undoManager.$lastDelta)
            editor.session.markUndoGroup();
    }
}
var Editor = require("../editor").Editor;
require("../config").defineOptions(Editor.prototype, "editor", {
    hardWrap: {
        set: function (val) {
            if (val) {
                this.commands.on("afterExec", wrapAfterInput);
            }
            else {
                this.commands.off("afterExec", wrapAfterInput);
            }
        },
        value: false
    }
});
exports.hardWrap = hardWrap;

});

define("ace/keyboard/vim",["require","exports","module","ace/range","ace/lib/event_emitter","ace/lib/dom","ace/lib/oop","ace/lib/keys","ace/lib/event","ace/search","ace/lib/useragent","ace/search_highlight","ace/commands/multi_select_commands","ace/mode/text","ace/ext/hardwrap","ace/multi_select"], function(require, exports, module){// CodeMirror, copyright (c) by Marijn Haverbeke and others
'use strict';
function log() {
    var d = "";
    function format(p) {
        if (typeof p != "object")
            return p + "";
        if ("line" in p) {
            return p.line + ":" + p.ch;
        }
        if ("anchor" in p) {
            return format(p.anchor) + "->" + format(p.head);
        }
        if (Array.isArray(p))
            return "[" + p.map(function (x) {
                return format(x);
            }) + "]";
        return JSON.stringify(p);
    }
    for (var i = 0; i < arguments.length; i++) {
        var p = arguments[i];
        var f = format(p);
        d += f + "  ";
    }
    console.log(d);
}
var Range = require("../range").Range;
var EventEmitter = require("../lib/event_emitter").EventEmitter;
var domLib = require("../lib/dom");
var oop = require("../lib/oop");
var KEYS = require("../lib/keys");
var event = require("../lib/event");
var Search = require("../search").Search;
var useragent = require("../lib/useragent");
var SearchHighlight = require("../search_highlight").SearchHighlight;
var multiSelectCommands = require("../commands/multi_select_commands");
var TextModeTokenRe = require("../mode/text").Mode.prototype.tokenRe;
var hardWrap = require("../ext/hardwrap").hardWrap;
require("../multi_select");
var CodeMirror = function (ace) {
    this.ace = ace;
    this.state = {};
    this.marks = {};
    this.options = {};
    this.$uid = 0;
    this.onChange = this.onChange.bind(this);
    this.onSelectionChange = this.onSelectionChange.bind(this);
    this.onBeforeEndOperation = this.onBeforeEndOperation.bind(this);
    this.ace.on('change', this.onChange);
    this.ace.on('changeSelection', this.onSelectionChange);
    this.ace.on('beforeEndOperation', this.onBeforeEndOperation);
};
CodeMirror.Pos = function (line, ch) {
    if (!(this instanceof Pos))
        return new Pos(line, ch);
    this.line = line;
    this.ch = ch;
};
CodeMirror.defineOption = function (name, val, setter) { };
CodeMirror.commands = {
    redo: function (cm) { cm.ace.redo(); },
    undo: function (cm) { cm.ace.undo(); },
    newlineAndIndent: function (cm) { cm.ace.insert("\n"); },
    goLineLeft: function (cm) { cm.ace.selection.moveCursorLineStart(); },
    goLineRight: function (cm) { cm.ace.selection.moveCursorLineEnd(); }
};
CodeMirror.keyMap = {};
CodeMirror.addClass = CodeMirror.rmClass = function () { };
CodeMirror.e_stop = CodeMirror.e_preventDefault = event.stopEvent;
CodeMirror.keyName = function (e) {
    var key = (KEYS[e.keyCode] || e.key || "");
    if (key.length == 1)
        key = key.toUpperCase();
    key = event.getModifierString(e).replace(/(^|-)\w/g, function (m) {
        return m.toUpperCase();
    }) + key;
    return key;
};
CodeMirror.keyMap['default'] = function (key) {
    return function (cm) {
        var cmd = cm.ace.commands.commandKeyBinding[key.toLowerCase()];
        return cmd && cm.ace.execCommand(cmd) !== false;
    };
};
CodeMirror.lookupKey = function lookupKey(key, map, handle) {
    if (!map)
        map = "default";
    if (typeof map == "string")
        map = CodeMirror.keyMap[map];
    var found = typeof map == "function" ? map(key) : map[key];
    if (found === false)
        return "nothing";
    if (found === "...")
        return "multi";
    if (found != null && handle(found))
        return "handled";
    if (map.fallthrough) {
        if (!Array.isArray(map.fallthrough))
            return lookupKey(key, map.fallthrough, handle);
        for (var i = 0; i < map.fallthrough.length; i++) {
            var result = lookupKey(key, map.fallthrough[i], handle);
            if (result)
                return result;
        }
    }
};
CodeMirror.findMatchingTag = function (cm, head) {
    return cm.findMatchingTag(head);
};
CodeMirror.findEnclosingTag = function (cm, head) {
};
CodeMirror.signal = function (o, name, e) { return o._signal(name, e); };
CodeMirror.on = event.addListener;
CodeMirror.off = event.removeListener;
CodeMirror.isWordChar = function (ch) {
    if (ch < "\x7f")
        return /^\w$/.test(ch);
    TextModeTokenRe.lastIndex = 0;
    return TextModeTokenRe.test(ch);
};
(function () {
    oop.implement(CodeMirror.prototype, EventEmitter);
    this.destroy = function () {
        this.ace.off('change', this.onChange);
        this.ace.off('changeSelection', this.onSelectionChange);
        this.ace.off('beforeEndOperation', this.onBeforeEndOperation);
        this.removeOverlay();
    };
    this.virtualSelectionMode = function () {
        return this.ace.inVirtualSelectionMode && this.ace.selection.index;
    };
    this.onChange = function (delta) {
        var change = { text: delta.action[0] == 'i' ? delta.lines : [] };
        var curOp = this.curOp = this.curOp || {};
        if (!curOp.changeHandlers)
            curOp.changeHandlers = this._eventRegistry["change"] && this._eventRegistry["change"].slice();
        if (!curOp.lastChange) {
            curOp.lastChange = curOp.change = change;
        }
        else {
            curOp.lastChange.next = curOp.lastChange = change;
        }
        this.$updateMarkers(delta);
    };
    this.onSelectionChange = function () {
        var curOp = this.curOp = this.curOp || {};
        if (!curOp.cursorActivityHandlers)
            curOp.cursorActivityHandlers = this._eventRegistry["cursorActivity"] && this._eventRegistry["cursorActivity"].slice();
        this.curOp.cursorActivity = true;
        if (this.ace.inMultiSelectMode) {
            this.ace.keyBinding.removeKeyboardHandler(multiSelectCommands.keyboardHandler);
        }
    };
    this.operation = function (fn, force) {
        if (!force && this.curOp || force && this.curOp && this.curOp.force) {
            return fn();
        }
        if (force || !this.ace.curOp) {
            if (this.curOp)
                this.onBeforeEndOperation();
        }
        if (!this.ace.curOp) {
            var prevOp = this.ace.prevOp;
            this.ace.startOperation({
                command: { name: "vim", scrollIntoView: "cursor" }
            });
        }
        var curOp = this.curOp = this.curOp || {};
        this.curOp.force = force;
        var result = fn();
        if (this.ace.curOp && this.ace.curOp.command.name == "vim") {
            if (this.state.dialog)
                this.ace.curOp.command.scrollIntoView = this.ace.curOp.vimDialogScroll;
            this.ace.endOperation();
            if (!curOp.cursorActivity && !curOp.lastChange && prevOp)
                this.ace.prevOp = prevOp;
        }
        if (force || !this.ace.curOp) {
            if (this.curOp)
                this.onBeforeEndOperation();
        }
        return result;
    };
    this.onBeforeEndOperation = function () {
        var op = this.curOp;
        if (op) {
            if (op.change) {
                this.signal("change", op.change, op);
            }
            if (op && op.cursorActivity) {
                this.signal("cursorActivity", null, op);
            }
            this.curOp = null;
        }
    };
    this.signal = function (eventName, e, handlers) {
        var listeners = handlers ? handlers[eventName + "Handlers"]
            : (this._eventRegistry || {})[eventName];
        if (!listeners)
            return;
        listeners = listeners.slice();
        for (var i = 0; i < listeners.length; i++)
            listeners[i](this, e);
    };
    this.firstLine = function () { return 0; };
    this.lastLine = function () { return this.ace.session.getLength() - 1; };
    this.lineCount = function () { return this.ace.session.getLength(); };
    this.setCursor = function (line, ch) {
        if (typeof line === 'object') {
            ch = line.ch;
            line = line.line;
        }
        var shouldScroll = !this.curOp && !this.ace.inVirtualSelectionMode;
        if (!this.ace.inVirtualSelectionMode)
            this.ace.exitMultiSelectMode();
        this.ace.session.unfold({ row: line, column: ch });
        this.ace.selection.moveTo(line, ch);
        if (shouldScroll) {
            this.ace.renderer.scrollCursorIntoView();
            this.ace.endOperation();
        }
    };
    this.getCursor = function (p) {
        var sel = this.ace.selection;
        var pos = p == 'anchor' ? (sel.isEmpty() ? sel.lead : sel.anchor) :
            p == 'head' || !p ? sel.lead : sel.getRange()[p];
        return toCmPos(pos);
    };
    this.listSelections = function (p) {
        var ranges = this.ace.multiSelect.rangeList.ranges;
        if (!ranges.length || this.ace.inVirtualSelectionMode)
            return [{ anchor: this.getCursor('anchor'), head: this.getCursor('head') }];
        return ranges.map(function (r) {
            return {
                anchor: this.clipPos(toCmPos(r.cursor == r.end ? r.start : r.end)),
                head: this.clipPos(toCmPos(r.cursor))
            };
        }, this);
    };
    this.setSelections = function (p, primIndex) {
        var sel = this.ace.multiSelect;
        var ranges = p.map(function (x) {
            var anchor = toAcePos(x.anchor);
            var head = toAcePos(x.head);
            var r = Range.comparePoints(anchor, head) < 0
                ? new Range.fromPoints(anchor, head)
                : new Range.fromPoints(head, anchor);
            r.cursor = Range.comparePoints(r.start, head) ? r.end : r.start;
            return r;
        });
        if (this.ace.inVirtualSelectionMode) {
            this.ace.selection.fromOrientedRange(ranges[0].clone());
            return;
        }
        if (!primIndex) {
            ranges = ranges.reverse();
        }
        else if (ranges[primIndex]) {
            ranges.push(ranges.splice(primIndex, 1)[0]);
        }
        sel.toSingleRange(ranges[0].clone());
        var session = this.ace.session;
        for (var i = 0; i < ranges.length; i++) {
            var range = session.$clipRangeToDocument(ranges[i]); // todo why ace doesn't do this?
            sel.addRange(range);
        }
    };
    this.setSelection = function (a, h, options) {
        var sel = this.ace.selection;
        sel.moveTo(a.line, a.ch);
        sel.selectTo(h.line, h.ch);
        if (options && options.origin == '*mouse') {
            this.onBeforeEndOperation();
        }
    };
    this.somethingSelected = function (p) {
        return !this.ace.selection.isEmpty();
    };
    this.clipPos = function (p) {
        var pos = this.ace.session.$clipPositionToDocument(p.line, p.ch);
        return toCmPos(pos);
    };
    this.foldCode = function (pos) {
        this.ace.session.$toggleFoldWidget(pos.line, {});
    };
    this.markText = function (cursor) {
        return { clear: function () { }, find: function () { } };
    };
    this.$updateMarkers = function (delta) {
        var isInsert = delta.action == "insert";
        var start = delta.start;
        var end = delta.end;
        var rowShift = (end.row - start.row) * (isInsert ? 1 : -1);
        var colShift = (end.column - start.column) * (isInsert ? 1 : -1);
        if (isInsert)
            end = start;
        for (var i in this.marks) {
            var point = this.marks[i];
            var cmp = Range.comparePoints(point, start);
            if (cmp < 0) {
                continue; // delta starts after the range
            }
            if (cmp === 0) {
                if (isInsert) {
                    if (point.bias == 1) {
                        cmp = 1;
                    }
                    else {
                        point.bias = -1;
                        continue;
                    }
                }
            }
            var cmp2 = isInsert ? cmp : Range.comparePoints(point, end);
            if (cmp2 > 0) {
                point.row += rowShift;
                point.column += point.row == end.row ? colShift : 0;
                continue;
            }
            if (!isInsert && cmp2 <= 0) {
                point.row = start.row;
                point.column = start.column;
                if (cmp2 === 0)
                    point.bias = 1;
            }
        }
    };
    var Marker = function (cm, id, row, column) {
        this.cm = cm;
        this.id = id;
        this.row = row;
        this.column = column;
        cm.marks[this.id] = this;
    };
    Marker.prototype.clear = function () { delete this.cm.marks[this.id]; };
    Marker.prototype.find = function () { return toCmPos(this); };
    this.setBookmark = function (cursor, options) {
        var bm = new Marker(this, this.$uid++, cursor.line, cursor.ch);
        if (!options || !options.insertLeft)
            bm.$insertRight = true;
        this.marks[bm.id] = bm;
        return bm;
    };
    this.moveH = function (increment, unit) {
        if (unit == 'char') {
            var sel = this.ace.selection;
            sel.clearSelection();
            sel.moveCursorBy(0, increment);
        }
    };
    this.findPosV = function (start, amount, unit, goalColumn) {
        if (unit == 'page') {
            var renderer = this.ace.renderer;
            var config = renderer.layerConfig;
            amount = amount * Math.floor(config.height / config.lineHeight);
            unit = 'line';
        }
        if (unit == 'line') {
            var screenPos = this.ace.session.documentToScreenPosition(start.line, start.ch);
            if (goalColumn != null)
                screenPos.column = goalColumn;
            screenPos.row += amount;
            screenPos.row = Math.min(Math.max(0, screenPos.row), this.ace.session.getScreenLength() - 1);
            var pos = this.ace.session.screenToDocumentPosition(screenPos.row, screenPos.column);
            return toCmPos(pos);
        }
        else {
            debugger;
        }
    };
    this.charCoords = function (pos, mode) {
        if (mode == 'div' || !mode) {
            var sc = this.ace.session.documentToScreenPosition(pos.line, pos.ch);
            return { left: sc.column, top: sc.row };
        }
        if (mode == 'local') {
            var renderer = this.ace.renderer;
            var sc = this.ace.session.documentToScreenPosition(pos.line, pos.ch);
            var lh = renderer.layerConfig.lineHeight;
            var cw = renderer.layerConfig.characterWidth;
            var top = lh * sc.row;
            return { left: sc.column * cw, top: top, bottom: top + lh };
        }
    };
    this.coordsChar = function (pos, mode) {
        var renderer = this.ace.renderer;
        if (mode == 'local') {
            var row = Math.max(0, Math.floor(pos.top / renderer.lineHeight));
            var col = Math.max(0, Math.floor(pos.left / renderer.characterWidth));
            var ch = renderer.session.screenToDocumentPosition(row, col);
            return toCmPos(ch);
        }
        else if (mode == 'div') {
            throw "not implemented";
        }
    };
    this.getSearchCursor = function (query, pos, caseFold) {
        var caseSensitive = false;
        var isRegexp = false;
        if (query instanceof RegExp && !query.global) {
            caseSensitive = !query.ignoreCase;
            query = query.source;
            isRegexp = true;
        }
        if (query == "\\n") {
            query = "\n";
            isRegexp = false;
        }
        var search = new Search();
        if (pos.ch == undefined)
            pos.ch = Number.MAX_VALUE;
        var acePos = { row: pos.line, column: pos.ch };
        var cm = this;
        var last = null;
        return {
            findNext: function () { return this.find(false); },
            findPrevious: function () { return this.find(true); },
            find: function (back) {
                search.setOptions({
                    needle: query,
                    caseSensitive: caseSensitive,
                    wrap: false,
                    backwards: back,
                    regExp: isRegexp,
                    start: last || acePos
                });
                var range = search.find(cm.ace.session);
                last = range;
                return last && [!last.isEmpty()];
            },
            from: function () { return last && toCmPos(last.start); },
            to: function () { return last && toCmPos(last.end); },
            replace: function (text) {
                if (last) {
                    last.end = cm.ace.session.doc.replace(last, text);
                }
            }
        };
    };
    this.scrollTo = function (x, y) {
        var renderer = this.ace.renderer;
        var config = renderer.layerConfig;
        var maxHeight = config.maxHeight;
        maxHeight -= (renderer.$size.scrollerHeight - renderer.lineHeight) * renderer.$scrollPastEnd;
        if (y != null)
            this.ace.session.setScrollTop(Math.max(0, Math.min(y, maxHeight)));
        if (x != null)
            this.ace.session.setScrollLeft(Math.max(0, Math.min(x, config.width)));
    };
    this.scrollInfo = function () { return 0; };
    this.scrollIntoView = function (pos, margin) {
        if (pos) {
            var renderer = this.ace.renderer;
            var viewMargin = { "top": 0, "bottom": margin };
            renderer.scrollCursorIntoView(toAcePos(pos), (renderer.lineHeight * 2) / renderer.$size.scrollerHeight, viewMargin);
        }
    };
    this.getLine = function (row) { return this.ace.session.getLine(row); };
    this.getRange = function (s, e) {
        return this.ace.session.getTextRange(new Range(s.line, s.ch, e.line, e.ch));
    };
    this.replaceRange = function (text, s, e) {
        if (!e)
            e = s;
        var range = new Range(s.line, s.ch, e.line, e.ch);
        this.ace.session.$clipRangeToDocument(range);
        return this.ace.session.replace(range, text);
    };
    this.replaceSelection =
        this.replaceSelections = function (p) {
            var sel = this.ace.selection;
            if (this.ace.inVirtualSelectionMode) {
                this.ace.session.replace(sel.getRange(), p[0] || "");
                return;
            }
            sel.inVirtualSelectionMode = true;
            var ranges = sel.rangeList.ranges;
            if (!ranges.length)
                ranges = [this.ace.multiSelect.getRange()];
            for (var i = ranges.length; i--;)
                this.ace.session.replace(ranges[i], p[i] || "");
            sel.inVirtualSelectionMode = false;
        };
    this.getSelection = function () {
        return this.ace.getSelectedText();
    };
    this.getSelections = function () {
        return this.listSelections().map(function (x) {
            return this.getRange(x.anchor, x.head);
        }, this);
    };
    this.getInputField = function () {
        return this.ace.textInput.getElement();
    };
    this.getWrapperElement = function () {
        return this.ace.container;
    };
    var optMap = {
        indentWithTabs: "useSoftTabs",
        indentUnit: "tabSize",
        tabSize: "tabSize",
        firstLineNumber: "firstLineNumber",
        readOnly: "readOnly"
    };
    this.setOption = function (name, val) {
        this.state[name] = val;
        switch (name) {
            case 'indentWithTabs':
                name = optMap[name];
                val = !val;
                break;
            case 'keyMap':
                this.state.$keyMap = val;
                return;
                break;
            default:
                name = optMap[name];
        }
        if (name)
            this.ace.setOption(name, val);
    };
    this.getOption = function (name) {
        var val;
        var aceOpt = optMap[name];
        if (aceOpt)
            val = this.ace.getOption(aceOpt);
        switch (name) {
            case 'indentWithTabs':
                name = optMap[name];
                return !val;
            case 'keyMap':
                return this.state.$keyMap || 'vim';
        }
        return aceOpt ? val : this.state[name];
    };
    this.toggleOverwrite = function (on) {
        this.state.overwrite = on;
        return this.ace.setOverwrite(on);
    };
    this.addOverlay = function (o) {
        if (!this.$searchHighlight || !this.$searchHighlight.session) {
            var highlight = new SearchHighlight(null, "ace_highlight-marker", "text");
            var marker = this.ace.session.addDynamicMarker(highlight);
            highlight.id = marker.id;
            highlight.session = this.ace.session;
            highlight.destroy = function (o) {
                highlight.session.off("change", highlight.updateOnChange);
                highlight.session.off("changeEditor", highlight.destroy);
                highlight.session.removeMarker(highlight.id);
                highlight.session = null;
            };
            highlight.updateOnChange = function (delta) {
                var row = delta.start.row;
                if (row == delta.end.row)
                    highlight.cache[row] = undefined;
                else
                    highlight.cache.splice(row, highlight.cache.length);
            };
            highlight.session.on("changeEditor", highlight.destroy);
            highlight.session.on("change", highlight.updateOnChange);
        }
        var re = new RegExp(o.query.source, "gmi");
        this.$searchHighlight = o.highlight = highlight;
        this.$searchHighlight.setRegexp(re);
        this.ace.renderer.updateBackMarkers();
    };
    this.removeOverlay = function (o) {
        if (this.$searchHighlight && this.$searchHighlight.session) {
            this.$searchHighlight.destroy();
        }
    };
    this.getScrollInfo = function () {
        var renderer = this.ace.renderer;
        var config = renderer.layerConfig;
        return {
            left: renderer.scrollLeft,
            top: renderer.scrollTop,
            height: config.maxHeight,
            width: config.width,
            clientHeight: config.height,
            clientWidth: config.width
        };
    };
    this.getValue = function () {
        return this.ace.getValue();
    };
    this.setValue = function (v) {
        return this.ace.setValue(v, -1);
    };
    this.getTokenTypeAt = function (pos) {
        var token = this.ace.session.getTokenAt(pos.line, pos.ch);
        return token && /comment|string/.test(token.type) ? "string" : "";
    };
    this.findMatchingBracket = function (pos) {
        var m = this.ace.session.findMatchingBracket(toAcePos(pos));
        return { to: m && toCmPos(m) };
    };
    this.findMatchingTag = function (pos) {
        var m = this.ace.session.getMatchingTags(toAcePos(pos));
        if (!m)
            return;
        return {
            open: {
                from: toCmPos(m.openTag.start),
                to: toCmPos(m.openTag.end)
            },
            close: {
                from: toCmPos(m.closeTag.start),
                to: toCmPos(m.closeTag.end)
            }
        };
    };
    this.indentLine = function (line, method) {
        if (method === true)
            this.ace.session.indentRows(line, line, "\t");
        else if (method === false)
            this.ace.session.outdentRows(new Range(line, 0, line, 0));
    };
    this.indexFromPos = function (pos) {
        return this.ace.session.doc.positionToIndex(toAcePos(pos));
    };
    this.posFromIndex = function (index) {
        return toCmPos(this.ace.session.doc.indexToPosition(index));
    };
    this.focus = function (index) {
        return this.ace.textInput.focus();
    };
    this.blur = function (index) {
        return this.ace.blur();
    };
    this.defaultTextHeight = function (index) {
        return this.ace.renderer.layerConfig.lineHeight;
    };
    this.scanForBracket = function (pos, dir, _, options) {
        var re = options.bracketRegex.source;
        var tokenRe = /paren|text|operator|tag/;
        if (dir == 1) {
            var m = this.ace.session.$findClosingBracket(re.slice(1, 2), toAcePos(pos), tokenRe);
        }
        else {
            var m = this.ace.session.$findOpeningBracket(re.slice(-2, -1), { row: pos.line, column: pos.ch + 1 }, tokenRe);
        }
        return m && { pos: toCmPos(m) };
    };
    this.refresh = function () {
        return this.ace.resize(true);
    };
    this.getMode = function () {
        return { name: this.getOption("mode") };
    };
    this.execCommand = function (name) {
        if (CodeMirror.commands.hasOwnProperty(name))
            return CodeMirror.commands[name](this);
        if (name == "indentAuto")
            return this.ace.execCommand("autoindent");
        console.log(name + " is not implemented");
    };
    this.getLineNumber = function (handle) {
        return handle.row;
    };
    this.getLineHandle = function (row) {
        return { text: this.ace.session.getLine(row), row: row };
    };
}).call(CodeMirror.prototype);
function toAcePos(cmPos) {
    return { row: cmPos.line, column: cmPos.ch };
}
function toCmPos(acePos) {
    return new Pos(acePos.row, acePos.column);
}
var StringStream = CodeMirror.StringStream = function (string, tabSize) {
    this.pos = this.start = 0;
    this.string = string;
    this.tabSize = tabSize || 8;
    this.lastColumnPos = this.lastColumnValue = 0;
    this.lineStart = 0;
};
StringStream.prototype = {
    eol: function () { return this.pos >= this.string.length; },
    sol: function () { return this.pos == this.lineStart; },
    peek: function () { return this.string.charAt(this.pos) || undefined; },
    next: function () {
        if (this.pos < this.string.length)
            return this.string.charAt(this.pos++);
    },
    eat: function (match) {
        var ch = this.string.charAt(this.pos);
        if (typeof match == "string")
            var ok = ch == match;
        else
            var ok = ch && (match.test ? match.test(ch) : match(ch));
        if (ok) {
            ++this.pos;
            return ch;
        }
    },
    eatWhile: function (match) {
        var start = this.pos;
        while (this.eat(match)) { }
        return this.pos > start;
    },
    eatSpace: function () {
        var start = this.pos;
        while (/[\s\u00a0]/.test(this.string.charAt(this.pos)))
            ++this.pos;
        return this.pos > start;
    },
    skipToEnd: function () { this.pos = this.string.length; },
    skipTo: function (ch) {
        var found = this.string.indexOf(ch, this.pos);
        if (found > -1) {
            this.pos = found;
            return true;
        }
    },
    backUp: function (n) { this.pos -= n; },
    column: function () {
        throw "not implemented";
    },
    indentation: function () {
        throw "not implemented";
    },
    match: function (pattern, consume, caseInsensitive) {
        if (typeof pattern == "string") {
            var cased = function (str) { return caseInsensitive ? str.toLowerCase() : str; };
            var substr = this.string.substr(this.pos, pattern.length);
            if (cased(substr) == cased(pattern)) {
                if (consume !== false)
                    this.pos += pattern.length;
                return true;
            }
        }
        else {
            var match = this.string.slice(this.pos).match(pattern);
            if (match && match.index > 0)
                return null;
            if (match && consume !== false)
                this.pos += match[0].length;
            return match;
        }
    },
    current: function () { return this.string.slice(this.start, this.pos); },
    hideFirstChars: function (n, inner) {
        this.lineStart += n;
        try {
            return inner();
        }
        finally {
            this.lineStart -= n;
        }
    }
};
CodeMirror.defineExtension = function (name, fn) {
    CodeMirror.prototype[name] = fn;
};
domLib.importCssString(".normal-mode .ace_cursor{\n    border: none;\n    background-color: rgba(255,0,0,0.5);\n}\n.normal-mode .ace_hidden-cursors .ace_cursor{\n  background-color: transparent;\n  border: 1px solid red;\n  opacity: 0.7\n}\n.ace_dialog {\n  position: absolute;\n  left: 0; right: 0;\n  background: inherit;\n  z-index: 15;\n  padding: .1em .8em;\n  overflow: hidden;\n  color: inherit;\n}\n.ace_dialog-top {\n  border-bottom: 1px solid #444;\n  top: 0;\n}\n.ace_dialog-bottom {\n  border-top: 1px solid #444;\n  bottom: 0;\n}\n.ace_dialog input {\n  border: none;\n  outline: none;\n  background: transparent;\n  width: 20em;\n  color: inherit;\n  font-family: monospace;\n}", "vimMode", false);
(function () {
    function dialogDiv(cm, template, bottom) {
        var wrap = cm.ace.container;
        var dialog;
        dialog = wrap.appendChild(document.createElement("div"));
        if (bottom)
            dialog.className = "ace_dialog ace_dialog-bottom";
        else
            dialog.className = "ace_dialog ace_dialog-top";
        if (typeof template == "string") {
            dialog.innerHTML = template;
        }
        else { // Assuming it's a detached DOM element.
            dialog.appendChild(template);
        }
        return dialog;
    }
    function closeNotification(cm, newVal) {
        if (cm.state.currentNotificationClose)
            cm.state.currentNotificationClose();
        cm.state.currentNotificationClose = newVal;
    }
    CodeMirror.defineExtension("openDialog", function (template, callback, options) {
        if (this.virtualSelectionMode())
            return;
        if (!options)
            options = {};
        closeNotification(this, null);
        var dialog = dialogDiv(this, template, options.bottom);
        var closed = false, me = this;
        this.state.dialog = dialog;
        function close(newVal) {
            if (typeof newVal == 'string') {
                inp.value = newVal;
            }
            else {
                if (closed)
                    return;
                if (newVal && newVal.type == "blur") {
                    if (document.activeElement === inp)
                        return;
                }
                if (me.state.dialog == dialog) {
                    me.state.dialog = null;
                    me.focus();
                }
                closed = true;
                dialog.remove();
                if (options.onClose)
                    options.onClose(dialog);
                var cm = me;
                if (cm.state.vim) {
                    cm.state.vim.status = null;
                    cm.ace._signal("changeStatus");
                    cm.ace.renderer.$loop.schedule(cm.ace.renderer.CHANGE_CURSOR);
                }
            }
        }
        var inp = dialog.getElementsByTagName("input")[0], button;
        if (inp) {
            if (options.value) {
                inp.value = options.value;
                if (options.selectValueOnOpen !== false)
                    inp.select();
            }
            if (options.onInput)
                CodeMirror.on(inp, "input", function (e) { options.onInput(e, inp.value, close); });
            if (options.onKeyUp)
                CodeMirror.on(inp, "keyup", function (e) { options.onKeyUp(e, inp.value, close); });
            CodeMirror.on(inp, "keydown", function (e) {
                if (options && options.onKeyDown && options.onKeyDown(e, inp.value, close)) {
                    return;
                }
                if (e.keyCode == 13)
                    callback(inp.value);
                if (e.keyCode == 27 || (options.closeOnEnter !== false && e.keyCode == 13)) {
                    CodeMirror.e_stop(e);
                    close();
                }
            });
            if (options.closeOnBlur !== false)
                CodeMirror.on(inp, "blur", close);
            inp.focus();
        }
        else if (button = dialog.getElementsByTagName("button")[0]) {
            CodeMirror.on(button, "click", function () {
                close();
                me.focus();
            });
            if (options.closeOnBlur !== false)
                CodeMirror.on(button, "blur", close);
            button.focus();
        }
        return close;
    });
    CodeMirror.defineExtension("openNotification", function (template, options) {
        if (this.virtualSelectionMode())
            return;
        closeNotification(this, close);
        var dialog = dialogDiv(this, template, options && options.bottom);
        var closed = false, doneTimer;
        var duration = options && typeof options.duration !== "undefined" ? options.duration : 5000;
        function close() {
            if (closed)
                return;
            closed = true;
            clearTimeout(doneTimer);
            dialog.remove();
        }
        CodeMirror.on(dialog, 'click', function (e) {
            CodeMirror.e_preventDefault(e);
            close();
        });
        if (duration)
            doneTimer = setTimeout(close, duration);
        return close;
    });
})();
var Pos = CodeMirror.Pos;
function transformCursor(cm, range) {
    var vim = cm.state.vim;
    if (!vim || vim.insertMode)
        return range.head;
    var head = vim.sel.head;
    if (!head)
        return range.head;
    if (vim.visualBlock) {
        if (range.head.line != head.line) {
            return;
        }
    }
    if (range.from() == range.anchor && !range.empty()) {
        if (range.head.line == head.line && range.head.ch != head.ch)
            return new Pos(range.head.line, range.head.ch - 1);
    }
    return range.head;
}
function updateSelectionForSurrogateCharacters(cm, curStart, curEnd) {
    if (curStart.line === curEnd.line && curStart.ch >= curEnd.ch - 1) {
        var text = cm.getLine(curStart.line);
        var charCode = text.charCodeAt(curStart.ch);
        if (0xD800 <= charCode && charCode <= 0xD8FF) {
            curEnd.ch += 1;
        }
    }
    return { start: curStart, end: curEnd };
}
var defaultKeymap = [
    { keys: '<Left>', type: 'keyToKey', toKeys: 'h' },
    { keys: '<Right>', type: 'keyToKey', toKeys: 'l' },
    { keys: '<Up>', type: 'keyToKey', toKeys: 'k' },
    { keys: '<Down>', type: 'keyToKey', toKeys: 'j' },
    { keys: 'g<Up>', type: 'keyToKey', toKeys: 'gk' },
    { keys: 'g<Down>', type: 'keyToKey', toKeys: 'gj' },
    { keys: '<Space>', type: 'keyToKey', toKeys: 'l' },
    { keys: '<BS>', type: 'keyToKey', toKeys: 'h', context: 'normal' },
    { keys: '<Del>', type: 'keyToKey', toKeys: 'x', context: 'normal' },
    { keys: '<C-Space>', type: 'keyToKey', toKeys: 'W' },
    { keys: '<C-BS>', type: 'keyToKey', toKeys: 'B', context: 'normal' },
    { keys: '<S-Space>', type: 'keyToKey', toKeys: 'w' },
    { keys: '<S-BS>', type: 'keyToKey', toKeys: 'b', context: 'normal' },
    { keys: '<C-n>', type: 'keyToKey', toKeys: 'j' },
    { keys: '<C-p>', type: 'keyToKey', toKeys: 'k' },
    { keys: '<C-[>', type: 'keyToKey', toKeys: '<Esc>' },
    { keys: '<C-c>', type: 'keyToKey', toKeys: '<Esc>' },
    { keys: '<C-[>', type: 'keyToKey', toKeys: '<Esc>', context: 'insert' },
    { keys: '<C-c>', type: 'keyToKey', toKeys: '<Esc>', context: 'insert' },
    { keys: '<C-Esc>', type: 'keyToKey', toKeys: '<Esc>' },
    { keys: '<C-Esc>', type: 'keyToKey', toKeys: '<Esc>', context: 'insert' },
    { keys: 's', type: 'keyToKey', toKeys: 'cl', context: 'normal' },
    { keys: 's', type: 'keyToKey', toKeys: 'c', context: 'visual' },
    { keys: 'S', type: 'keyToKey', toKeys: 'cc', context: 'normal' },
    { keys: 'S', type: 'keyToKey', toKeys: 'VdO', context: 'visual' },
    { keys: '<Home>', type: 'keyToKey', toKeys: '0' },
    { keys: '<End>', type: 'keyToKey', toKeys: '$' },
    { keys: '<PageUp>', type: 'keyToKey', toKeys: '<C-b>' },
    { keys: '<PageDown>', type: 'keyToKey', toKeys: '<C-f>' },
    { keys: '<CR>', type: 'keyToKey', toKeys: 'j^', context: 'normal' },
    { keys: '<Ins>', type: 'keyToKey', toKeys: 'i', context: 'normal' },
    { keys: '<Ins>', type: 'action', action: 'toggleOverwrite', context: 'insert' },
    { keys: 'H', type: 'motion', motion: 'moveToTopLine', motionArgs: { linewise: true, toJumplist: true } },
    { keys: 'M', type: 'motion', motion: 'moveToMiddleLine', motionArgs: { linewise: true, toJumplist: true } },
    { keys: 'L', type: 'motion', motion: 'moveToBottomLine', motionArgs: { linewise: true, toJumplist: true } },
    { keys: 'h', type: 'motion', motion: 'moveByCharacters', motionArgs: { forward: false } },
    { keys: 'l', type: 'motion', motion: 'moveByCharacters', motionArgs: { forward: true } },
    { keys: 'j', type: 'motion', motion: 'moveByLines', motionArgs: { forward: true, linewise: true } },
    { keys: 'k', type: 'motion', motion: 'moveByLines', motionArgs: { forward: false, linewise: true } },
    { keys: 'gj', type: 'motion', motion: 'moveByDisplayLines', motionArgs: { forward: true } },
    { keys: 'gk', type: 'motion', motion: 'moveByDisplayLines', motionArgs: { forward: false } },
    { keys: 'w', type: 'motion', motion: 'moveByWords', motionArgs: { forward: true, wordEnd: false } },
    { keys: 'W', type: 'motion', motion: 'moveByWords', motionArgs: { forward: true, wordEnd: false, bigWord: true } },
    { keys: 'e', type: 'motion', motion: 'moveByWords', motionArgs: { forward: true, wordEnd: true, inclusive: true } },
    { keys: 'E', type: 'motion', motion: 'moveByWords', motionArgs: { forward: true, wordEnd: true, bigWord: true, inclusive: true } },
    { keys: 'b', type: 'motion', motion: 'moveByWords', motionArgs: { forward: false, wordEnd: false } },
    { keys: 'B', type: 'motion', motion: 'moveByWords', motionArgs: { forward: false, wordEnd: false, bigWord: true } },
    { keys: 'ge', type: 'motion', motion: 'moveByWords', motionArgs: { forward: false, wordEnd: true, inclusive: true } },
    { keys: 'gE', type: 'motion', motion: 'moveByWords', motionArgs: { forward: false, wordEnd: true, bigWord: true, inclusive: true } },
    { keys: '{', type: 'motion', motion: 'moveByParagraph', motionArgs: { forward: false, toJumplist: true } },
    { keys: '}', type: 'motion', motion: 'moveByParagraph', motionArgs: { forward: true, toJumplist: true } },
    { keys: '(', type: 'motion', motion: 'moveBySentence', motionArgs: { forward: false } },
    { keys: ')', type: 'motion', motion: 'moveBySentence', motionArgs: { forward: true } },
    { keys: '<C-f>', type: 'motion', motion: 'moveByPage', motionArgs: { forward: true } },
    { keys: '<C-b>', type: 'motion', motion: 'moveByPage', motionArgs: { forward: false } },
    { keys: '<C-d>', type: 'motion', motion: 'moveByScroll', motionArgs: { forward: true, explicitRepeat: true } },
    { keys: '<C-u>', type: 'motion', motion: 'moveByScroll', motionArgs: { forward: false, explicitRepeat: true } },
    { keys: 'gg', type: 'motion', motion: 'moveToLineOrEdgeOfDocument', motionArgs: { forward: false, explicitRepeat: true, linewise: true, toJumplist: true } },
    { keys: 'G', type: 'motion', motion: 'moveToLineOrEdgeOfDocument', motionArgs: { forward: true, explicitRepeat: true, linewise: true, toJumplist: true } },
    { keys: "g$", type: "motion", motion: "moveToEndOfDisplayLine" },
    { keys: "g^", type: "motion", motion: "moveToStartOfDisplayLine" },
    { keys: "g0", type: "motion", motion: "moveToStartOfDisplayLine" },
    { keys: '0', type: 'motion', motion: 'moveToStartOfLine' },
    { keys: '^', type: 'motion', motion: 'moveToFirstNonWhiteSpaceCharacter' },
    { keys: '+', type: 'motion', motion: 'moveByLines', motionArgs: { forward: true, toFirstChar: true } },
    { keys: '-', type: 'motion', motion: 'moveByLines', motionArgs: { forward: false, toFirstChar: true } },
    { keys: '_', type: 'motion', motion: 'moveByLines', motionArgs: { forward: true, toFirstChar: true, repeatOffset: -1 } },
    { keys: '$', type: 'motion', motion: 'moveToEol', motionArgs: { inclusive: true } },
    { keys: '%', type: 'motion', motion: 'moveToMatchedSymbol', motionArgs: { inclusive: true, toJumplist: true } },
    { keys: 'f<character>', type: 'motion', motion: 'moveToCharacter', motionArgs: { forward: true, inclusive: true } },
    { keys: 'F<character>', type: 'motion', motion: 'moveToCharacter', motionArgs: { forward: false } },
    { keys: 't<character>', type: 'motion', motion: 'moveTillCharacter', motionArgs: { forward: true, inclusive: true } },
    { keys: 'T<character>', type: 'motion', motion: 'moveTillCharacter', motionArgs: { forward: false } },
    { keys: ';', type: 'motion', motion: 'repeatLastCharacterSearch', motionArgs: { forward: true } },
    { keys: ',', type: 'motion', motion: 'repeatLastCharacterSearch', motionArgs: { forward: false } },
    { keys: '\'<character>', type: 'motion', motion: 'goToMark', motionArgs: { toJumplist: true, linewise: true } },
    { keys: '`<character>', type: 'motion', motion: 'goToMark', motionArgs: { toJumplist: true } },
    { keys: ']`', type: 'motion', motion: 'jumpToMark', motionArgs: { forward: true } },
    { keys: '[`', type: 'motion', motion: 'jumpToMark', motionArgs: { forward: false } },
    { keys: ']\'', type: 'motion', motion: 'jumpToMark', motionArgs: { forward: true, linewise: true } },
    { keys: '[\'', type: 'motion', motion: 'jumpToMark', motionArgs: { forward: false, linewise: true } },
    { keys: ']p', type: 'action', action: 'paste', isEdit: true, actionArgs: { after: true, isEdit: true, matchIndent: true } },
    { keys: '[p', type: 'action', action: 'paste', isEdit: true, actionArgs: { after: false, isEdit: true, matchIndent: true } },
    { keys: ']<character>', type: 'motion', motion: 'moveToSymbol', motionArgs: { forward: true, toJumplist: true } },
    { keys: '[<character>', type: 'motion', motion: 'moveToSymbol', motionArgs: { forward: false, toJumplist: true } },
    { keys: '|', type: 'motion', motion: 'moveToColumn' },
    { keys: 'o', type: 'motion', motion: 'moveToOtherHighlightedEnd', context: 'visual' },
    { keys: 'O', type: 'motion', motion: 'moveToOtherHighlightedEnd', motionArgs: { sameLine: true }, context: 'visual' },
    { keys: 'd', type: 'operator', operator: 'delete' },
    { keys: 'y', type: 'operator', operator: 'yank' },
    { keys: 'c', type: 'operator', operator: 'change' },
    { keys: '=', type: 'operator', operator: 'indentAuto' },
    { keys: '>', type: 'operator', operator: 'indent', operatorArgs: { indentRight: true } },
    { keys: '<', type: 'operator', operator: 'indent', operatorArgs: { indentRight: false } },
    { keys: 'g~', type: 'operator', operator: 'changeCase' },
    { keys: 'gu', type: 'operator', operator: 'changeCase', operatorArgs: { toLower: true }, isEdit: true },
    { keys: 'gU', type: 'operator', operator: 'changeCase', operatorArgs: { toLower: false }, isEdit: true },
    { keys: 'n', type: 'motion', motion: 'findNext', motionArgs: { forward: true, toJumplist: true } },
    { keys: 'N', type: 'motion', motion: 'findNext', motionArgs: { forward: false, toJumplist: true } },
    { keys: 'gn', type: 'motion', motion: 'findAndSelectNextInclusive', motionArgs: { forward: true } },
    { keys: 'gN', type: 'motion', motion: 'findAndSelectNextInclusive', motionArgs: { forward: false } },
    { keys: 'x', type: 'operatorMotion', operator: 'delete', motion: 'moveByCharacters', motionArgs: { forward: true }, operatorMotionArgs: { visualLine: false } },
    { keys: 'X', type: 'operatorMotion', operator: 'delete', motion: 'moveByCharacters', motionArgs: { forward: false }, operatorMotionArgs: { visualLine: true } },
    { keys: 'D', type: 'operatorMotion', operator: 'delete', motion: 'moveToEol', motionArgs: { inclusive: true }, context: 'normal' },
    { keys: 'D', type: 'operator', operator: 'delete', operatorArgs: { linewise: true }, context: 'visual' },
    { keys: 'Y', type: 'operatorMotion', operator: 'yank', motion: 'expandToLine', motionArgs: { linewise: true }, context: 'normal' },
    { keys: 'Y', type: 'operator', operator: 'yank', operatorArgs: { linewise: true }, context: 'visual' },
    { keys: 'C', type: 'operatorMotion', operator: 'change', motion: 'moveToEol', motionArgs: { inclusive: true }, context: 'normal' },
    { keys: 'C', type: 'operator', operator: 'change', operatorArgs: { linewise: true }, context: 'visual' },
    { keys: '~', type: 'operatorMotion', operator: 'changeCase', motion: 'moveByCharacters', motionArgs: { forward: true }, operatorArgs: { shouldMoveCursor: true }, context: 'normal' },
    { keys: '~', type: 'operator', operator: 'changeCase', context: 'visual' },
    { keys: '<C-u>', type: 'operatorMotion', operator: 'delete', motion: 'moveToStartOfLine', context: 'insert' },
    { keys: '<C-w>', type: 'operatorMotion', operator: 'delete', motion: 'moveByWords', motionArgs: { forward: false, wordEnd: false }, context: 'insert' },
    { keys: '<C-w>', type: 'idle', context: 'normal' },
    { keys: '<C-i>', type: 'action', action: 'jumpListWalk', actionArgs: { forward: true } },
    { keys: '<C-o>', type: 'action', action: 'jumpListWalk', actionArgs: { forward: false } },
    { keys: '<C-e>', type: 'action', action: 'scroll', actionArgs: { forward: true, linewise: true } },
    { keys: '<C-y>', type: 'action', action: 'scroll', actionArgs: { forward: false, linewise: true } },
    { keys: 'a', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'charAfter' }, context: 'normal' },
    { keys: 'A', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'eol' }, context: 'normal' },
    { keys: 'A', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'endOfSelectedArea' }, context: 'visual' },
    { keys: 'i', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'inplace' }, context: 'normal' },
    { keys: 'gi', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'lastEdit' }, context: 'normal' },
    { keys: 'I', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'firstNonBlank' }, context: 'normal' },
    { keys: 'gI', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'bol' }, context: 'normal' },
    { keys: 'I', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { insertAt: 'startOfSelectedArea' }, context: 'visual' },
    { keys: 'o', type: 'action', action: 'newLineAndEnterInsertMode', isEdit: true, interlaceInsertRepeat: true, actionArgs: { after: true }, context: 'normal' },
    { keys: 'O', type: 'action', action: 'newLineAndEnterInsertMode', isEdit: true, interlaceInsertRepeat: true, actionArgs: { after: false }, context: 'normal' },
    { keys: 'v', type: 'action', action: 'toggleVisualMode' },
    { keys: 'V', type: 'action', action: 'toggleVisualMode', actionArgs: { linewise: true } },
    { keys: '<C-v>', type: 'action', action: 'toggleVisualMode', actionArgs: { blockwise: true } },
    { keys: '<C-q>', type: 'action', action: 'toggleVisualMode', actionArgs: { blockwise: true } },
    { keys: 'gv', type: 'action', action: 'reselectLastSelection' },
    { keys: 'J', type: 'action', action: 'joinLines', isEdit: true },
    { keys: 'gJ', type: 'action', action: 'joinLines', actionArgs: { keepSpaces: true }, isEdit: true },
    { keys: 'p', type: 'action', action: 'paste', isEdit: true, actionArgs: { after: true, isEdit: true } },
    { keys: 'P', type: 'action', action: 'paste', isEdit: true, actionArgs: { after: false, isEdit: true } },
    { keys: 'r<character>', type: 'action', action: 'replace', isEdit: true },
    { keys: '@<character>', type: 'action', action: 'replayMacro' },
    { keys: 'q<character>', type: 'action', action: 'enterMacroRecordMode' },
    { keys: 'R', type: 'action', action: 'enterInsertMode', isEdit: true, actionArgs: { replace: true }, context: 'normal' },
    { keys: 'R', type: 'operator', operator: 'change', operatorArgs: { linewise: true, fullLine: true }, context: 'visual', exitVisualBlock: true },
    { keys: 'u', type: 'action', action: 'undo', context: 'normal' },
    { keys: 'u', type: 'operator', operator: 'changeCase', operatorArgs: { toLower: true }, context: 'visual', isEdit: true },
    { keys: 'U', type: 'operator', operator: 'changeCase', operatorArgs: { toLower: false }, context: 'visual', isEdit: true },
    { keys: '<C-r>', type: 'action', action: 'redo' },
    { keys: 'm<character>', type: 'action', action: 'setMark' },
    { keys: '"<character>', type: 'action', action: 'setRegister' },
    { keys: 'zz', type: 'action', action: 'scrollToCursor', actionArgs: { position: 'center' } },
    { keys: 'z.', type: 'action', action: 'scrollToCursor', actionArgs: { position: 'center' }, motion: 'moveToFirstNonWhiteSpaceCharacter' },
    { keys: 'zt', type: 'action', action: 'scrollToCursor', actionArgs: { position: 'top' } },
    { keys: 'z<CR>', type: 'action', action: 'scrollToCursor', actionArgs: { position: 'top' }, motion: 'moveToFirstNonWhiteSpaceCharacter' },
    { keys: 'zb', type: 'action', action: 'scrollToCursor', actionArgs: { position: 'bottom' } },
    { keys: 'z-', type: 'action', action: 'scrollToCursor', actionArgs: { position: 'bottom' }, motion: 'moveToFirstNonWhiteSpaceCharacter' },
    { keys: '.', type: 'action', action: 'repeatLastEdit' },
    { keys: '<C-a>', type: 'action', action: 'incrementNumberToken', isEdit: true, actionArgs: { increase: true, backtrack: false } },
    { keys: '<C-x>', type: 'action', action: 'incrementNumberToken', isEdit: true, actionArgs: { increase: false, backtrack: false } },
    { keys: '<C-t>', type: 'action', action: 'indent', actionArgs: { indentRight: true }, context: 'insert' },
    { keys: '<C-d>', type: 'action', action: 'indent', actionArgs: { indentRight: false }, context: 'insert' },
    { keys: 'a<character>', type: 'motion', motion: 'textObjectManipulation' },
    { keys: 'i<character>', type: 'motion', motion: 'textObjectManipulation', motionArgs: { textObjectInner: true } },
    { keys: '/', type: 'search', searchArgs: { forward: true, querySrc: 'prompt', toJumplist: true } },
    { keys: '?', type: 'search', searchArgs: { forward: false, querySrc: 'prompt', toJumplist: true } },
    { keys: '*', type: 'search', searchArgs: { forward: true, querySrc: 'wordUnderCursor', wholeWordOnly: true, toJumplist: true } },
    { keys: '#', type: 'search', searchArgs: { forward: false, querySrc: 'wordUnderCursor', wholeWordOnly: true, toJumplist: true } },
    { keys: 'g*', type: 'search', searchArgs: { forward: true, querySrc: 'wordUnderCursor', toJumplist: true } },
    { keys: 'g#', type: 'search', searchArgs: { forward: false, querySrc: 'wordUnderCursor', toJumplist: true } },
    { keys: ':', type: 'ex' }
];
var defaultKeymapLength = defaultKeymap.length;
var defaultExCommandMap = [
    { name: 'colorscheme', shortName: 'colo' },
    { name: 'map' },
    { name: 'imap', shortName: 'im' },
    { name: 'nmap', shortName: 'nm' },
    { name: 'vmap', shortName: 'vm' },
    { name: 'unmap' },
    { name: 'write', shortName: 'w' },
    { name: 'undo', shortName: 'u' },
    { name: 'redo', shortName: 'red' },
    { name: 'set', shortName: 'se' },
    { name: 'setlocal', shortName: 'setl' },
    { name: 'setglobal', shortName: 'setg' },
    { name: 'sort', shortName: 'sor' },
    { name: 'substitute', shortName: 's', possiblyAsync: true },
    { name: 'nohlsearch', shortName: 'noh' },
    { name: 'yank', shortName: 'y' },
    { name: 'delmarks', shortName: 'delm' },
    { name: 'registers', shortName: 'reg', excludeFromCommandHistory: true },
    { name: 'vglobal', shortName: 'v' },
    { name: 'global', shortName: 'g' }
];
function enterVimMode(cm) {
    cm.setOption('disableInput', true);
    cm.setOption('showCursorWhenSelecting', false);
    CodeMirror.signal(cm, "vim-mode-change", { mode: "normal" });
    cm.on('cursorActivity', onCursorActivity);
    maybeInitVimState(cm);
    CodeMirror.on(cm.getInputField(), 'paste', getOnPasteFn(cm));
}
function leaveVimMode(cm) {
    cm.setOption('disableInput', false);
    cm.off('cursorActivity', onCursorActivity);
    CodeMirror.off(cm.getInputField(), 'paste', getOnPasteFn(cm));
    cm.state.vim = null;
    if (highlightTimeout)
        clearTimeout(highlightTimeout);
}
function detachVimMap(cm, next) {
    if (this == CodeMirror.keyMap.vim) {
        cm.options.$customCursor = null;
        CodeMirror.rmClass(cm.getWrapperElement(), "cm-fat-cursor");
    }
    if (!next || next.attach != attachVimMap)
        leaveVimMode(cm);
}
function attachVimMap(cm, prev) {
    if (this == CodeMirror.keyMap.vim) {
        if (cm.curOp)
            cm.curOp.selectionChanged = true;
        cm.options.$customCursor = transformCursor;
        CodeMirror.addClass(cm.getWrapperElement(), "cm-fat-cursor");
    }
    if (!prev || prev.attach != attachVimMap)
        enterVimMode(cm);
}
CodeMirror.defineOption('vimMode', false, function (cm, val, prev) {
    if (val && cm.getOption("keyMap") != "vim")
        cm.setOption("keyMap", "vim");
    else if (!val && prev != CodeMirror.Init && /^vim/.test(cm.getOption("keyMap")))
        cm.setOption("keyMap", "default");
});
function cmKey(key, cm) {
    if (!cm) {
        return undefined;
    }
    if (this[key]) {
        return this[key];
    }
    var vimKey = cmKeyToVimKey(key);
    if (!vimKey) {
        return false;
    }
    var cmd = vimApi.findKey(cm, vimKey);
    if (typeof cmd == 'function') {
        CodeMirror.signal(cm, 'vim-keypress', vimKey);
    }
    return cmd;
}
var modifiers = { Shift: 'S', Ctrl: 'C', Alt: 'A', Cmd: 'D', Mod: 'A', CapsLock: '' };
var specialKeys = { Enter: 'CR', Backspace: 'BS', Delete: 'Del', Insert: 'Ins' };
function cmKeyToVimKey(key) {
    if (key.charAt(0) == '\'') {
        return key.charAt(1);
    }
    var pieces = key.split(/-(?!$)/);
    var lastPiece = pieces[pieces.length - 1];
    if (pieces.length == 1 && pieces[0].length == 1) {
        return false;
    }
    else if (pieces.length == 2 && pieces[0] == 'Shift' && lastPiece.length == 1) {
        return false;
    }
    var hasCharacter = false;
    for (var i = 0; i < pieces.length; i++) {
        var piece = pieces[i];
        if (piece in modifiers) {
            pieces[i] = modifiers[piece];
        }
        else {
            hasCharacter = true;
        }
        if (piece in specialKeys) {
            pieces[i] = specialKeys[piece];
        }
    }
    if (!hasCharacter) {
        return false;
    }
    if (isUpperCase(lastPiece)) {
        pieces[pieces.length - 1] = lastPiece.toLowerCase();
    }
    return '<' + pieces.join('-') + '>';
}
function getOnPasteFn(cm) {
    var vim = cm.state.vim;
    if (!vim.onPasteFn) {
        vim.onPasteFn = function () {
            if (!vim.insertMode) {
                cm.setCursor(offsetCursor(cm.getCursor(), 0, 1));
                actions.enterInsertMode(cm, {}, vim);
            }
        };
    }
    return vim.onPasteFn;
}
var numberRegex = /[\d]/;
var wordCharTest = [CodeMirror.isWordChar, function (ch) {
        return ch && !CodeMirror.isWordChar(ch) && !/\s/.test(ch);
    }], bigWordCharTest = [function (ch) {
        return /\S/.test(ch);
    }];
function makeKeyRange(start, size) {
    var keys = [];
    for (var i = start; i < start + size; i++) {
        keys.push(String.fromCharCode(i));
    }
    return keys;
}
var upperCaseAlphabet = makeKeyRange(65, 26);
var lowerCaseAlphabet = makeKeyRange(97, 26);
var numbers = makeKeyRange(48, 10);
var validMarks = [].concat(upperCaseAlphabet, lowerCaseAlphabet, numbers, ['<', '>']);
var validRegisters = [].concat(upperCaseAlphabet, lowerCaseAlphabet, numbers, ['-', '"', '.', ':', '_', '/', '+']);
var upperCaseChars;
try {
    upperCaseChars = new RegExp("^[\\p{Lu}]$", "u");
}
catch (_) {
    upperCaseChars = /^[A-Z]$/;
}
function isLine(cm, line) {
    return line >= cm.firstLine() && line <= cm.lastLine();
}
function isLowerCase(k) {
    return (/^[a-z]$/).test(k);
}
function isMatchableSymbol(k) {
    return '()[]{}'.indexOf(k) != -1;
}
function isNumber(k) {
    return numberRegex.test(k);
}
function isUpperCase(k) {
    return upperCaseChars.test(k);
}
function isWhiteSpaceString(k) {
    return (/^\s*$/).test(k);
}
function isEndOfSentenceSymbol(k) {
    return '.?!'.indexOf(k) != -1;
}
function inArray(val, arr) {
    for (var i = 0; i < arr.length; i++) {
        if (arr