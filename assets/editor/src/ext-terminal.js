import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import { SearchAddon } from 'xterm-addon-search';
import { Unicode11Addon } from 'xterm-addon-unicode11';

class TerminalIntegration {
  constructor(editor) {
    this.editor = editor;
    this.terminal = null;
    this.fitAddon = new FitAddon();
    this.webLinksAddon = new WebLinksAddon();
    this.searchAddon = new SearchAddon();
    this.unicode11Addon = new Unicode11Addon();
    this.terminalElement = null;
    this.isTerminalVisible = false;
  }

  init() {
    this.createTerminalElement();
    this.createTerminal();
    this.setupTerminalAddons();
    this.setupTerminalEvents();
    this.setupEditorEvents();
  }

  createTerminalElement() {
    this.terminalElement = document.createElement('div');
    this.terminalElement.id = 'terminal';
    this.terminalElement.style.position = 'absolute';
    this.terminalElement.style.bottom = '0';
    this.terminalElement.style.left = '0';
    this.terminalElement.style.width = '100%';
    this.terminalElement.style.height = '30%';
    this.terminalElement.style.backgroundColor = '#1e1e1e';
    this.terminalElement.style.display = 'none';
    document.body.appendChild(this.terminalElement);
  }

  createTerminal() {
    this.terminal = new Terminal({
      cursorBlink: true,
      fontFamily: 'monospace',
      fontSize: 14,
      theme: {
        background: '#1e1e1e',
        foreground: '#cccccc',
        cursor: '#ffffff',
        selection: 'rgba(100, 100, 100, 0.3)',
      },
    });
  }

  setupTerminalAddons() {
    this.terminal.loadAddon(this.fitAddon);
    this.terminal.loadAddon(this.webLinksAddon);
    this.terminal.loadAddon(this.searchAddon);
    this.terminal.loadAddon(this.unicode11Addon);
  }

  setupTerminalEvents() {
    this.terminal.onData((data) => {
      // Handle terminal input
    });

    this.terminal.onResize((size) => {
      // Handle terminal resize
    });
  }

  setupEditorEvents() {
    this.editor.on('toggleTerminal', () => {
      this.toggleTerminal();
    });
  }

  toggleTerminal() {
    this.isTerminalVisible = !this.isTerminalVisible;
    this.terminalElement.style.display = this.isTerminalVisible ? 'block' : 'none';
    if (this.isTerminalVisible) {
      this.terminal.open(this.terminalElement);
      this.fitAddon.fit();
    } else {
      this.terminal.dispose();
    }
  }

  writeToTerminal(data) {
    if (this.terminal) {
      this.terminal.write(data);
    }
  }
}

export default TerminalIntegration;