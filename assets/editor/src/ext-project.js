import { Plugin } from 'ckeditor5/src/core';
import { ButtonView } from 'ckeditor5/src/ui';
import { Collection } from 'ckeditor5/src/utils';

export default class ProjectManagement extends Plugin {
    init() {
        const editor = this.editor;

        // Create a collection to store project items
        const projectItems = new Collection();

        // Add the project management button to the toolbar
        editor.ui.componentFactory.add('projectManagement', locale => {
            const view = new ButtonView(locale);

            view.set({
                label: 'Project Management',
                icon: '<svg>...</svg>',
                tooltip: true
            });

            // Bind the state of the button to the command
            view.bind('isOn', 'isEnabled').to(editor.commands.get('projectManagement'));

            // Execute the command when the button is clicked
            this.listenTo(view, 'execute', () => editor.execute('projectManagement'));

            return view;
        });

        // Add the project management command
        editor.commands.add('projectManagement', new ProjectManagementCommand(editor, projectItems));

        // Add the project management panel
        editor.ui.addPanel('projectManagement', {
            position: 'right',
            items: [
                {
                    type: 'list',
                    items: projectItems
                }
            ]
        });
    }
}

class ProjectManagementCommand extends Command {
    constructor(editor, projectItems) {
        super(editor);
        this.projectItems = projectItems;
    }

    execute() {
        // Toggle the project management panel
        this.editor.ui.getPanel('projectManagement').toggle();
    }

    refresh() {
        // Enable the command
        this.isEnabled = true;
    }
}