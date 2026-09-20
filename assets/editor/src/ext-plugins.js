import { PluginManager } from './plugin-manager';

export class ExtPlugins {
  constructor() {
    this.pluginManager = new PluginManager();
  }

  async loadPlugins() {
    try {
      // Load plugins from the plugin manager
      await this.pluginManager.loadPlugins();
      console.log('Plugins loaded successfully');
    } catch (error) {
      console.error('Error loading plugins:', error);
    }
  }

  getPlugins() {
    return this.pluginManager.getPlugins();
  }

  getPlugin(name) {
    return this.pluginManager.getPlugin(name);
  }

  registerPlugin(plugin) {
    this.pluginManager.registerPlugin(plugin);
  }

  unregisterPlugin(name) {
    this.pluginManager.unregisterPlugin(name);
  }
}