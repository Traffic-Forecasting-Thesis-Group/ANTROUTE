/** Short HTTP requests inspect a long computation running in a worker. */
export interface RouteJobClient {
  post<T>(url: string, body: unknown, config: { signal?: AbortSignal; timeout: number }): Promise<{ data: T }>;
  get<T>(url: string, config: { signal?: AbortSignal; timeout: number }): Promise<{ data: T }>;
}

interface RouteJob<T> {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  result?: T | null;
  error?: string | null;
}

export interface RouteJobOptions {
  signal?: AbortSignal;
  pollIntervalMs?: number;
  maxWaitMs?: number;
}

function abortError(): Error {
  const error = new Error('Route request cancelled.');
  error.name = 'AbortError';
  return error;
}

function pause(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(abortError()); return; }
    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export async function waitForRouteJob<T>(
  client: RouteJobClient,
  payload: unknown,
  options: RouteJobOptions = {}
): Promise<T> {
  const { signal, pollIntervalMs = 1500, maxWaitMs = 65 * 60 * 1000 } = options;
  if (signal?.aborted) throw abortError();
  const requestConfig = { signal, timeout: 15000 };
  const { data: created } = await client.post<RouteJob<T>>('/routes/jobs', payload, requestConfig);
  const started = Date.now();
  let consecutiveErrors = 0;
  while (Date.now() - started < maxWaitMs) {
    if (signal?.aborted) throw abortError();
    let job: RouteJob<T>;
    try {
      const response = await client.get<RouteJob<T>>(
        `/routes/jobs/${encodeURIComponent(created.job_id)}`, requestConfig
      );
      job = response.data;
      consecutiveErrors = 0;
    } catch (error: any) {
      if (signal?.aborted) throw abortError();
      // GET is safe to retry. Never resubmit an accepted job after a tunnel hiccup.
      const status = error?.response?.status;
      if (status && status < 500) throw error;
      if (++consecutiveErrors >= 3) {
        throw new Error('Could not read route-job status. The calculation may still be running.');
      }
      await pause(pollIntervalMs, signal);
      continue;
    }
    if (job.status === 'completed') {
      if (job.result == null) throw new Error('The server returned a completed job without a route result.');
      return job.result;
    }
    if (job.status === 'failed') throw new Error(job.error || 'Route calculation failed.');
    if (job.status !== 'queued' && job.status !== 'running') {
      throw new Error('The server returned an unsupported route-job status.');
    }
    await pause(pollIntervalMs, signal);
  }
  throw new Error(`Route job ${created.job_id} is still unfinished. Check the worker logs before starting another calculation.`);
}
