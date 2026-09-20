import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FileTree = ({ onFileSelect }) => {
  const [treeData, setTreeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchFileTree = async () => {
      try {
        const response = await axios.get('/api/filetree');
        setTreeData(response.data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchFileTree();
  }, []);

  const renderTree = (nodes) => {
    if (!nodes) return null;

    return (
      <ul>
        {nodes.map((node) => (
          <li key={node.path}>
            {node.type === 'directory' ? (
              <>
                <span onClick={() => toggleNode(node.path)}>{node.name}</span>
                {node.children && renderTree(node.children)}
              </>
            ) : (
              <span onClick={() => onFileSelect(node.path)}>{node.name}</span>
            )}
          </li>
        ))}
      </ul>
    );
  };

  const toggleNode = (path) => {
    // Implement logic to toggle the visibility of child nodes
  };

  if (loading) return <div>Loading file tree...</div>;
  if (error) return <div>Error: {error}</div>;

  return <div className="file-tree">{renderTree(treeData)}</div>;
};

export default FileTree;