const vscode = require('vscode');
const { exec } = require('child_process');

function activate(context) {
    let disposable = vscode.commands.registerCommand('extension.runCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const document = editor.document;
        const selection = editor.selection;
        const code = selection.isEmpty ? document.getText() : document.getText(selection);

        if (!code.trim()) {
            vscode.window.showErrorMessage('No code selected');
            return;
        }

        const languageId = document.languageId;
        const terminal = vscode.window.createTerminal(`Code Execution: ${languageId}`);
        terminal.show();

        switch (languageId) {
            case 'python':
                terminal.sendText(`python -c "${code.replace(/"/g, '\\"')}"`);
                break;
            case 'javascript':
                terminal.sendText(`node -e "${code.replace(/"/g, '\\"')}"`);
                break;
            case 'typescript':
                terminal.sendText(`ts-node -e "${code.replace(/"/g, '\\"')}"`);
                break;
            case 'java':
                const javaFilePath = `${vscode.workspace.rootPath}/temp.java`;
                await vscode.workspace.fs.writeFile(vscode.Uri.file(javaFilePath), Buffer.from(code));
                terminal.sendText(`javac ${javaFilePath} && java -cp ${vscode.workspace.rootPath} temp`);
                break;
            case 'c':
                const cFilePath = `${vscode.workspace.rootPath}/temp.c`;
                await vscode.workspace.fs.writeFile(vscode.Uri.file(cFilePath), Buffer.from(code));
                terminal.sendText(`gcc ${cFilePath} -o ${vscode.workspace.rootPath}/temp && ${vscode.workspace.rootPath}/temp`);
                break;
            case 'cpp':
                const cppFilePath = `${vscode.workspace.rootPath}/temp.cpp`;
                await vscode.workspace.fs.writeFile(vscode.Uri.file(cppFilePath), Buffer.from(code));
                terminal.sendText(`g++ ${cppFilePath} -o ${vscode.workspace.rootPath}/temp && ${vscode.workspace.rootPath}/temp`);
                break;
            default:
                vscode.window.showErrorMessage(`Unsupported language: ${languageId}`);
                return;
        }
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};