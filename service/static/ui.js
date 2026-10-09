// Helpers of the management pages: element lookup, HTML escaping, JSON calls that throw the service's message.
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) { options.headers["content-type"] = "application/json"; options.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(url, options); } catch (_) { throw new Error("The service did not answer."); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `The service answered ${res.status}.`);
  }
  return data;
}

let toastTimer = null;
function toast(text, isError = false) {
  let el = $(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; el.setAttribute("role", "status"); document.body.appendChild(el); }
  el.textContent = text; el.classList.toggle("error", isError); el.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 2200);
}

// run an API change, toast its result; returns the answer or null when it failed
async function act(promise, done) {
  try { const data = await promise; if (done) toast(done); return data; }
  catch (err) { toast(err.message, true); return null; }
}

function param(name) { return new URLSearchParams(location.search).get(name); }
function setParam(name, value) {
  const url = new URL(location.href);
  if (value === null || value === undefined || value === "") url.searchParams.delete(name); else url.searchParams.set(name, value);
  history.replaceState(null, "", url);
}
