// The menu of every page: a button at the start of the page header that opens a drawer of the service's pages.
(() => {
  const PAGES = [
    { href: "/", label: "Generate", note: "Run jobs, review and download outputs" },
    { href: "/jobs", label: "Jobs", note: "Progress of every job: cancel, resume, re-generate" },
    { href: "/projects", label: "Projects", note: "Projects and the engines each may use" },
    { href: "/engines", label: "Engines", note: "Engines, their input kinds and outputs" },
    { href: "/prompts", label: "Prompts", note: "The prompt text of every engine, kind and case" },
    { href: "/references", label: "Object references", note: "One reference sheet per object" },
  ];
  const style = document.createElement("style");
  style.textContent = `
    .nav-btn { display: inline-flex; align-items: center; gap: 8px; align-self: center; padding: 7px 12px;
      border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--ink);
      font: 600 var(--step-0) var(--font-body); cursor: pointer; }
    .nav-btn:hover { border-color: var(--ink-soft); }
    .nav-btn svg { width: 16px; height: 16px; }
    .nav-drawer { position: fixed; inset: 0 auto 0 0; width: min(320px, 86vw); max-width: none; height: 100dvh;
      max-height: none; margin: 0; padding: 0; border: 0; border-right: 1px solid var(--rule);
      background: var(--surface); color: var(--ink); box-shadow: 8px 0 32px rgba(0,0,0,.18); }
    .nav-drawer::backdrop { background: rgba(8, 12, 18, .45); }
    .nav-drawer header { display: flex; align-items: center; justify-content: space-between; padding: 18px 18px 10px; }
    .nav-drawer header strong { font: 600 var(--step-3)/1.15 var(--font-display); }
    .nav-drawer header button { border: 0; background: none; color: var(--ink-soft); font-size: 22px; cursor: pointer;
      line-height: 1; padding: 4px 8px; border-radius: 6px; }
    .nav-drawer nav { display: grid; padding: 6px 10px 18px; gap: 2px; }
    .nav-drawer a { display: grid; gap: 1px; padding: 10px 12px; border-radius: 8px; color: var(--ink);
      text-decoration: none; border-left: 3px solid transparent; }
    .nav-drawer a span { font: 600 var(--step-2) var(--font-body); }
    .nav-drawer a small { color: var(--ink-soft); font-size: var(--step-0); }
    .nav-drawer a:hover { background: var(--sunken); }
    .nav-drawer a[aria-current="page"] { background: var(--sunken); border-left-color: var(--engine, var(--ink)); }`;
  document.head.appendChild(style);

  const here = location.pathname.replace(/\/+$/, "") || "/";
  const drawer = document.createElement("dialog");
  drawer.className = "nav-drawer";
  drawer.setAttribute("aria-label", "Pages");
  drawer.innerHTML = `<header><strong>Synthetic Data Generation</strong>
      <button type="button" aria-label="Close the menu">×</button></header>
    <nav>${PAGES.map((p) => `<a href="${p.href}"${p.href === here ? ' aria-current="page"' : ""}>
      <span>${p.label}</span><small>${p.note}</small></a>`).join("")}</nav>`;
  document.body.appendChild(drawer);
  drawer.querySelector("header button").addEventListener("click", () => drawer.close());
  drawer.addEventListener("click", (e) => { if (e.target === drawer) drawer.close(); });

  const button = document.createElement("button");
  button.type = "button";
  button.className = "nav-btn";
  button.setAttribute("aria-haspopup", "dialog");
  button.innerHTML = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor"
    stroke-width="1.6" stroke-linecap="round"/></svg>Menu`;
  button.addEventListener("click", () => drawer.showModal());
  const header = document.querySelector("header.top");
  if (header) header.prepend(button); else document.body.prepend(button);
})();
