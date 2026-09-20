import { Extension } from '@tiptap/core'
import { Plugin, PluginKey } from 'prosemirror-state'
import { Decoration, DecorationSet } from 'prosemirror-view'

export const SnippetsExtension = Extension.create({
  name: 'snippets',

  addOptions() {
    return {
      snippets: [],
    }
  },

  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: new PluginKey('snippets'),
        state: {
          init: (_, { doc }) => {
            const decorations = []
            const snippets = this.options.snippets

            doc.descendants((node, pos) => {
              if (node.type.name === 'codeBlock') {
                const snippet = snippets.find(s => s.code === node.textContent)
                if (snippet) {
                  decorations.push(
                    Decoration.node(pos, pos + node.nodeSize, {
                      class: 'snippet',
                      'data-snippet-id': snippet.id,
                    })
                  )
                }
              }
            })

            return DecorationSet.create(doc, decorations)
          },
          apply: (tr, oldState) => {
            const newState = oldState.map(tr.mapping, tr.doc)
            const snippets = this.options.snippets

            if (tr.docChanged) {
              const decorations = []
              tr.doc.descendants((node, pos) => {
                if (node.type.name === 'codeBlock') {
                  const snippet = snippets.find(s => s.code === node.textContent)
                  if (snippet) {
                    decorations.push(
                      Decoration.node(pos, pos + node.nodeSize, {
                        class: 'snippet',
                        'data-snippet-id': snippet.id,
                      })
                    )
                  }
                }
              })
              return newState.add(tr.doc, decorations)
            }

            return newState
          },
        },
        props: {
          decorations(state) {
            return this.getState(state)
          },
        },
      }),
    ]
  },

  addCommands() {
    return {
      insertSnippet: (snippet) => ({ tr, dispatch }) => {
        const { from, to } = tr.selection
        tr.replaceWith(from, to, this.editor.schema.nodes.codeBlock.create(null, snippet.code))
        if (dispatch) dispatch(tr)
        return true
      },
    }
  },
})