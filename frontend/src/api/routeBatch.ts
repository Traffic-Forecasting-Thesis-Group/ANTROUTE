export type RouteRequestKey = 'normal' | 'antroute' | 'baseline';

/** Each result is delivered immediately; one rejection cannot discard another result. */
export function requestRoutesIndependently<T>(
  requests: Record<RouteRequestKey, (signal: AbortSignal) => Promise<T>>,
  callbacks: {
    onResult: (key: RouteRequestKey, result: T) => void;
    onError: (key: RouteRequestKey, error: unknown) => void;
    onSettled: (key: RouteRequestKey) => void;
  },
  signal: AbortSignal
): void {
  const keys: RouteRequestKey[] = ['normal', 'antroute', 'baseline'];
  for (const key of keys) {
    void Promise.resolve()
      .then(() => {
        if (signal.aborted) return undefined;
        return requests[key](signal);
      })
      .then((result) => {
        if (!signal.aborted && result !== undefined) callbacks.onResult(key, result);
      })
      .catch((error: unknown) => {
        if (!signal.aborted) callbacks.onError(key, error);
      })
      .finally(() => {
        if (!signal.aborted) callbacks.onSettled(key);
      });
  }
}
