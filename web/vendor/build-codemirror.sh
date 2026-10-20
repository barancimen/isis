#!/bin/sh
# Rebuild web/vendor/codemirror.js, the one bundled file the page imports CodeMirror from.
# Needs Node.js. Run from the repository root: sh web/vendor/build-codemirror.sh
set -e
out="$(pwd)/web/vendor/codemirror.js"
work="$(mktemp -d)"
cd "$work"
npm init -y >/dev/null
npm install --silent --no-audit --no-fund \
  @codemirror/state@6.7.6 @codemirror/view@6.43.15 @codemirror/commands@6.11.1 \
  @codemirror/language@6.13.1 @codemirror/lang-markdown@6.5.2 @lezer/highlight@1.2.5 esbuild@0.28.2
cat > entry.js <<'JS'
export { EditorState, EditorSelection, StateField, StateEffect, Compartment } from "@codemirror/state";
export { EditorView, Decoration, WidgetType, keymap, placeholder, drawSelection, highlightActiveLine, lineNumbers } from "@codemirror/view";
export { history, defaultKeymap, historyKeymap, indentWithTab } from "@codemirror/commands";
export { syntaxHighlighting, HighlightStyle } from "@codemirror/language";
export { markdown } from "@codemirror/lang-markdown";
export { tags } from "@lezer/highlight";
JS
npx esbuild entry.js --bundle --format=esm --minify --legal-comments=none --outfile="$out"
cd / && rm -rf "$work"
echo "wrote $out"
