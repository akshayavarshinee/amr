import { useEffect, useState } from "react";
import { fetchJson } from "./api.js";

export function useApi(path, pollMs = null) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    let timer;

    async function load() {
      try {
        const result = await fetchJson(path);
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    if (pollMs) {
      timer = setInterval(load, pollMs);
    }

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [path, pollMs]);

  return { data, error, loading };
}
