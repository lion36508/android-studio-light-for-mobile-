(function() {
    window.require(["ace/ext/error_marker"], function(m) {
        if (typeof module == "object" && typeof exports == "object" && module) {
            module.exports = m;
        }
    });
})();

ace.define("ace/ext/error_marker", ["require", "exports", "module", "ace/range", "ace/lib/dom"], function(require, exports, module) {
    "use strict";

    var Range = require("../range").Range;
    var dom = require("../lib/dom");

    var ErrorMarker = function(session, range, clazz, type, inFront, message) {
        this.session = session;
        if (Array.isArray(range)) {
            range = Range.fromPoints(range[0], range[1]);
        }

        this.range = range;
        this.type = type;
        this.rendered = false;
        this.clazz = clazz;
        this.inFront = inFront;
        this.message = message || "";
        this.$markerId = dom.generateId();
    };

    (function() {
        this.update = function(range, clazz, type, message) {
            this.range = range;
            this.clazz = clazz;
            this.type = type;
            this.message = message || this.message;
            this.session._signal("changeFrontMarker");
        };

        this.attach = function() {
            this.session._addMarker(this);
        };

        this.detach = function() {
            this.session._removeMarker(this);
        };

        this.draw = function(html, markerLayer) {
            var range = this.range.clipRows(this.session.getScreenLastRow(), this.session.getScreenFirstRow());
            if (!range.isEmpty()) {
                var screenRange = this.session.addMarker(range, this.clazz, "text", this.inFront);
                if (screenRange) {
                    var height = markerLayer.lineHeight;
                    var width = (screenRange.end.column - screenRange.start.column) * markerLayer.characterWidth;
                    var top = markerLayer.$getTop(screenRange.start.row, true);
                    var left = markerLayer.$padding + screenRange.start.column * markerLayer.characterWidth;

                    var style = "position: absolute; top: " + top + "px; left: " + left + "px; width: " + width + "px; height: " + height + "px; background-color: rgba(255, 0, 0, 0.3); border-left: 2px solid red; z-index: 20;";

                    html.push(
                        "<div class='ace_error-marker' id='" + this.$markerId + "' style='" + style + "' title='" + this.message + "'></div>"
                    );
                    this.rendered = true;
                }
            }
        };

        this.redraw = function(markerLayer, bidiHandler) {
            var markerElement = document.getElementById(this.$markerId);
            if (!markerElement) {
                return;
            }

            var range = this.range.clipRows(this.session.getScreenLastRow(), this.session.getScreenFirstRow());
            if (range.isEmpty()) {
                this.rendered = false;
                markerElement.style.display = "none";
                return;
            }

            var screenRange = this.session.addMarker(range, this.clazz, "text", this.inFront);
            if (!screenRange) {
                this.rendered = false;
                markerElement.style.display = "none";
                return;
            }

            var height = markerLayer.lineHeight;
            var width = (screenRange.end.column - screenRange.start.column) * markerLayer.characterWidth;
            var top = markerLayer.$getTop(screenRange.start.row, true);
            var left = markerLayer.$padding + screenRange.start.column * markerLayer.characterWidth;

            markerElement.style.top = top + "px";
            markerElement.style.left = left + "px";
            markerElement.style.width = width + "px";
            markerElement.style.height = height + "px";
            markerElement.style.display = "block";
            markerElement.title = this.message;
            this.rendered = true;
        };

        this.destroy = function() {
            this.detach();
        };
    }).call(ErrorMarker.prototype);

    exports.ErrorMarker = ErrorMarker;
});

ace.define("ace/ext/bracket_matching", ["require", "exports", "module", "ace/range", "ace/lib/dom"], function(require, exports, module) {
    "use strict";

    var Range = require("../range").Range;
    var dom = require("../lib/dom");

    var BracketMatchingMarker = function(session, range, clazz, type, inFront, message) {
        this.session = session;
        if (Array.isArray(range)) {
            range = Range.fromPoints(range[0], range[1]);
        }

        this.range = range;
        this.type = type;
        this.rendered = false;
        this.clazz = clazz;
        this.inFront = inFront;
        this.message = message || "";
        this.$markerId = dom.generateId();
    };

    (function() {
        this.update = function(range, clazz, type, message) {
            this.range = range;
            this.clazz = clazz;
            this.type = type;
            this.message = message || this.message;
            this.session._signal("changeFrontMarker");
        };

        this.attach = function() {
            this.session._addMarker(this);
        };

        this.detach = function() {
            this.session._removeMarker(this);
        };

        this.draw = function(html, markerLayer) {
            var range = this.range.clipRows(this.session.getScreenLastRow(), this.session.getScreenFirstRow());
            if (!range.isEmpty()) {
                var screenRange = this.session.addMarker(range, this.clazz, "text", this.inFront);
                if (screenRange) {
                    var height = markerLayer.lineHeight;
                    var width = (screenRange.end.column - screenRange.start.column) * markerLayer.characterWidth;
                    var top = markerLayer.$getTop(screenRange.start.row, true);
                    var left = markerLayer.$padding + screenRange.start.column * markerLayer.characterWidth;

                    var style = "position: absolute; top: " + top + "px; left: " + left + "px; width: " + width + "px; height: " + height + "px; background-color: rgba(0, 255, 0, 0.3); border-left: 2px solid green; z-index: 20;";

                    html.push(
                        "<div class='ace_bracket-marker' id='" + this.$markerId + "' style='" + style + "' title='" + this.message + "'></div>"
                    );
                    this.rendered = true;
                }
            }
        };

        this.redraw = function(markerLayer, bidiHandler) {
            var markerElement = document.getElementById(this.$markerId);
            if (!markerElement) {
                return;
            }

            var range = this.range.clipRows(this.session.getScreenLastRow(), this.session.getScreenFirstRow());
            if (range.isEmpty()) {
                this.rendered = false;
                markerElement.style.display = "none";
                return;
            }

            var screenRange = this.session.addMarker(range, this.clazz, "text", this.inFront);
            if (!screenRange) {
                this.rendered = false;
                markerElement.style.display = "none";
                return;
            }

            var height = markerLayer.lineHeight;
            var width = (screenRange.end.column - screenRange.start.column) * markerLayer.characterWidth;
            var top = markerLayer.$getTop(screenRange.start.row, true);
            var left = markerLayer.$padding + screenRange.start.column * markerLayer.characterWidth;

            markerElement.style.top = top + "px";
            markerElement.style.left = left + "px";
            markerElement.style.width = width + "px";
            markerElement.style.height = height + "px";
            markerElement.style.display = "block";
            markerElement.title = this.message;
            this.rendered = true;
        };

        this.destroy = function() {
            this.detach();
        };
    }).call(BracketMatchingMarker.prototype);

    exports.BracketMatchingMarker = BracketMatchingMarker;
});