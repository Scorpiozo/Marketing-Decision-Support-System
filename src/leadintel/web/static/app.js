// Shared hover tooltip for chart marks: any element with data-tip shows its text.
(() => {
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.setAttribute("role", "status");
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tip));
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) return;
    tip.textContent = el.getAttribute("data-tip");
    tip.classList.add("show");
  });
  document.addEventListener("pointermove", (e) => {
    if (!tip.classList.contains("show")) return;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.transform = `translate(${x}px, ${e.clientY + 14}px)`;
  });
  document.addEventListener("pointerout", (e) => {
    if (e.target.closest("[data-tip]")) tip.classList.remove("show");
  });
})();

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
}
