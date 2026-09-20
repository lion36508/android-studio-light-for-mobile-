const vscode = require('vscode');
const { exec } = require('child_process');

class RefactorTools {
    constructor() {
        this.context = null;
    }

    activate(context) {
        this.context = context;

        // Register commands
        let extractMethodCommand = vscode.commands.registerCommand('extension.extractMethod', this.extractMethod, this);
        let renameSymbolCommand = vscode.commands.registerCommand('extension.renameSymbol', this.renameSymbol, this);
        let inlineVariableCommand = vscode.commands.registerCommand('extension.inlineVariable', this.inlineVariable, this);

        context.subscriptions.push(extractMethodCommand, renameSymbolCommand, inlineVariableCommand);
    }

    async extractMethod() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found.');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('No code selected for extraction.');
            return;
        }

        const selectedText = editor.document.getText(selection);
        const methodName = await vscode.window.showInputBox({ prompt: 'Enter method name' });

        if (!methodName) {
            return;
        }

        // Get the language ID to determine the appropriate refactoring tool
        const languageId = editor.document.languageId;
        let refactorCommand;

        switch (languageId) {
            case 'javascript':
                refactorCommand = `echo "${selectedText}" | node -e "const fs = require('fs'); const code = fs.readFileSync(0, 'utf-8'); console.log(code.replace(/\\b${selectedText}\\b/g, '${methodName}()'))"`;
                break;
            case 'python':
                refactorCommand = `echo "${selectedText}" | python3 -c "import sys; code = sys.stdin.read(); print(code.replace('${selectedText}', '${methodName}()'))"`;
                break;
            // Add more languages as needed
            default:
                vscode.window.showErrorMessage(`Refactoring not supported for ${languageId}`);
                return;
        }

        exec(refactorCommand, (error, stdout, stderr) => {
            if (error) {
                vscode.window.showErrorMessage(`Error: ${error.message}`);
                return;
            }
            if (stderr) {
                vscode.window.showErrorMessage(`Stderr: ${stderr}`);
                return;
            }

            editor.edit(editBuilder => {
                editBuilder.replace(selection, `${methodName}()`);
                // Insert the extracted method at the end of the file
                editBuilder.insert(new vscode.Position(editor.document.lineCount, 0), `\n${stdout}`);
            });
        });
    }

    async renameSymbol() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found.');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('No symbol selected for renaming.');
            return;
        }

        const selectedText = editor.document.getText(selection);
        const newName = await vscode.window.showInputBox({ prompt: 'Enter new name' });

        if (!newName) {
            return;
        }

        editor.edit(editBuilder => {
            editBuilder.replace(selection, newName);
        });
    }

    async inlineVariable() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found.');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('No variable selected for inlining.');
            return;
        }

        const selectedText = editor.document.getText(selection);
        const variableName = await vscode.window.showInputBox({ prompt: 'Enter variable name to inline' });

        if (!variableName) {
            return;
        }

        // Get the language ID to determine the appropriate refactoring tool
        const languageId = editor.document.languageId;
        let refactorCommand;

        switch (languageId) {
            case 'javascript':
                refactorCommand = `echo "${selectedText}" | node -e "const fs = require('fs'); const code = fs.readFileSync(0, 'utf-8'); console.log(code.replace(/\\b${variableName}\\b/g, '${selectedText}'))"`;
                break;
            case 'python':
                refactorCommand = `echo "${selectedText}" | python3 -c "import sys; code = sys.stdin.read(); print(code.replace('${variableName}', '${selectedText}'))"`;
                break;
            // Add more languages as needed
            default:
                vscode.window.showErrorMessage(`Refactoring not supported for ${languageId}`);
                return;
        }

        exec(refactorCommand, (error, stdout, stderr) => {
            if (error) {
                vscode.window.showErrorMessage(`Error: ${error.message}`);
                return;
            }
            if (stderr) {
                vscode.window.showErrorMessage(`Stderr: ${stderr}`);
                return;
            }

            editor.edit(editBuilder => {
                editBuilder.replace(selection, stdout);
            });
        });
    }
}

module.exports = new RefactorTools();