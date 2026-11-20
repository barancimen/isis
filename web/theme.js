// Runs before the page is drawn, so that it opens in the chosen theme without a flash of the other.
// Light unless this browser has kept "dark" (see the theme menu in app.js).
try {
  if (localStorage.getItem("isis.theme") === "dark") document.documentElement.dataset.theme = "dark";
} catch {}
