export async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json();
}
