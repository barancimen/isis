// The searches made in this tab, newest first. The box searches while you type, so a query joins
// the list only once it is settled: on Enter, when a result is opened or added, or when the box
// is left.

export const HISTORY_LIMIT = 20;

// The list with `query` at the top: a search made again moves up instead of showing twice. Queries
// that differ only in spaces at their ends count as one.
export function remember(list, query, limit = HISTORY_LIMIT) {
  const q = query.trim();
  if (!q) return list;
  return [q, ...list.filter((old) => old !== q)].slice(0, limit);
}

export function forget(list, query) {
  return list.filter((old) => old !== query);
}

// A saved list, or an empty one when it is missing or not a list of strings.
export function cleanHistory(saved, limit = HISTORY_LIMIT) {
  if (!Array.isArray(saved)) return [];
  return [...new Set(saved.filter((q) => typeof q === "string" && q.trim()).map((q) => q.trim()))].slice(0, limit);
}
