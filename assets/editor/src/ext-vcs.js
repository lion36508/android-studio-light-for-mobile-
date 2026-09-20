// assets/editor/src/ext-vcs.js

const { exec } = require('child_process');
const path = require('path');

class GitIntegration {
  constructor(projectPath) {
    this.projectPath = projectPath;
  }

  init() {
    return new Promise((resolve, reject) => {
      exec('git init', { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error initializing Git repository: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }

  add(filePath) {
    const relativePath = path.relative(this.projectPath, filePath);
    return new Promise((resolve, reject) => {
      exec(`git add "${relativePath}"`, { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error adding file to Git: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }

  commit(message) {
    return new Promise((resolve, reject) => {
      exec(`git commit -m "${message}"`, { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error committing changes to Git: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }

  status() {
    return new Promise((resolve, reject) => {
      exec('git status', { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error getting Git status: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }

  log() {
    return new Promise((resolve, reject) => {
      exec('git log --oneline', { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error getting Git log: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }

  pull() {
    return new Promise((resolve, reject) => {
      exec('git pull', { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error pulling changes from Git: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }

  push() {
    return new Promise((resolve, reject) => {
      exec('git push', { cwd: this.projectPath }, (error, stdout, stderr) => {
        if (error) {
          reject(`Error pushing changes to Git: ${stderr}`);
        } else {
          resolve(stdout);
        }
      });
    });
  }
}

module.exports = GitIntegration;