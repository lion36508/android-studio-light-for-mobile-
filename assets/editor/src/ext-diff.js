const vscode = require('vscode');
const path = require('path');
const fs = require('fs');

function activate(context) {
    let disposable = vscode.commands.registerCommand('extension.diffFiles', async () => {
        const file1 = await vscode.window.showOpenDialog({
            canSelectMany: false,
            openLabel: 'Select first file',
            filters: {
                'All files': ['*']
            }
        });

        if (!file1 || file1.length === 0) {
            return;
        }

        const file2 = await vscode.window.showOpenDialog({
            canSelectMany: false,
            openLabel: 'Select second file',
            filters: {
                'All files': ['*']
            }
        });

        if (!file2 || file2.length === 0) {
            return;
        }

        const file1Path = file1[0].fsPath;
        const file2Path = file2[0].fsPath;

        if (!fs.existsSync(file1Path) || !fs.existsSync(file2Path)) {
            vscode.window.showErrorMessage('One or both files do not exist.');
            return;
        }

        const title = `Diff: ${path.basename(file1Path)} ↔ ${path.basename(file2Path)}`;
        const diffView = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(255, 0, 0, 0.3)',
            isWholeLine: true
        });

        const diffEditor = await vscode.commands.executeCommand('vscode.diff', file1Path, file2Path, title);

        if (!diffEditor) {
            vscode.window.showErrorMessage('Failed to open diff editor.');
            return;
        }

        const diffEditorWebview = diffEditor.webview;
        diffEditorWebview.onDidReceiveMessage(message => {
            if (message.command === 'diff') {
                const diff = message.diff;
                const decorations = diff.map(line => {
                    return {
                        range: new vscode.Range(line, 0, line, 0),
                        hoverMessage: 'Difference'
                    };
                });
                diffEditor.setDecorations(diffView, decorations);
            }
        });

        context.subscriptions.push(disposable);
    });
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};