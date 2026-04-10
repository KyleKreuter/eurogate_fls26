import { useEffect, useState } from "react";

/**
 * Debounce a value by `delay` milliseconds.
 *
 * `useDebounce("hel", 300)` returns `""` for 300ms, then the latest typed
 * value. Used by the Container Inspector search input to avoid hammering
 * `/api/containers` on every keystroke.
 */
export function useDebounce<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(id);
  }, [value, delay]);

  return debounced;
}
