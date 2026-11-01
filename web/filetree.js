// The Files panel: corpus paths as a tree of folders, of which only the open ones are listed.

const folderNode = (name, path) => ({ name, path, count: 0, folders: new Map(), files: [] });

export function fileTree(paths) {
  const root = folderNode("", "");
  for (const path of paths) {
    const parts = path.split("/");
    const name = parts.pop();
    let node = root;
    node.count++;
    for (const part of parts) {
      if (!node.folders.has(part)) {
        node.folders.set(part, folderNode(part, node.path ? `${node.path}/${part}` : part));
      }
      node = node.folders.get(part);
      node.count++;
    }
    node.files.push({ name, path });
  }
  return root;
}

// The rows to show, folders before files, with at most `limit` files; `more` counts the rest.
export function visibleRows(tree, isOpen, limit = Infinity) {
  const rows = [];
  let shown = 0;
  let more = 0;
  const visit = (node, depth) => {
    for (const folder of node.folders.values()) {
      const open = isOpen(folder.path);
      rows.push({ kind: "folder", depth, open, name: folder.name, path: folder.path, count: folder.count });
      if (open) visit(folder, depth + 1);
    }
    for (const file of node.files) {
      if (shown < limit) {
        rows.push({ kind: "file", depth, name: file.name, path: file.path });
        shown++;
      } else {
        more++;
      }
    }
  };
  visit(tree, 0);
  return { rows, more };
}

// The folders above a file, outermost first.
export function ancestors(path) {
  const parts = path.split("/").slice(0, -1);
  return parts.map((_, i) => parts.slice(0, i + 1).join("/"));
}
