import { useCallback, useEffect, useState } from "react";

export function useAuth() {
  const [authed, setAuthed] = useState(null); // null = checking

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => setAuthed(r.ok))
      .catch(() => setAuthed(false));
  }, []);

  const login = useCallback(async (username, password) => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { Authorization: `Basic ${btoa(`${username}:${password}`)}` },
    });
    if (res.ok) setAuthed(true);
    return res.ok;
  }, []);

  const logout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    setAuthed(false);
  }, []);

  return { authed, login, logout };
}
