export const api = (url: string, init?: RequestInit) =>
fetch(url, { ...init, credentials: "include" });