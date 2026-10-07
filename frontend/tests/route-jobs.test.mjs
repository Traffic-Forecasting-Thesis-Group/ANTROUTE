import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as flush } from 'node:timers/promises';
import { waitForRouteJob } from '../src/api/routeJobs.ts';
import { requestRoutesIndependently } from '../src/api/routeBatch.ts';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function startBatch() {
  const jobs = { normal: deferred(), antroute: deferred(), baseline: deferred() };
  const events = [];
  const controller = new AbortController();
  requestRoutesIndependently(
    Object.fromEntries(Object.entries(jobs).map(([key, job]) => [key, () => job.promise])),
    {
      onResult: (key, result) => events.push(['result', key, result]),
      onError: (key, error) => events.push(['error', key, error.message]),
      onSettled: (key) => events.push(['settled', key]),
    }, controller.signal
  );
  return { jobs, events, controller };
}

test('ANTRoute is delivered while the baseline remains unresolved', async () => {
  const { jobs, events, controller } = startBatch();
  jobs.antroute.resolve({ routes: ['ready'] });
  await flush();
  assert.deepEqual(events, [['result', 'antroute', { routes: ['ready'] }], ['settled', 'antroute']]);
  controller.abort();
});

test('a baseline failure does not discard a successful ANTRoute result', async () => {
  const { jobs, events, controller } = startBatch();
  jobs.antroute.resolve('route');
  jobs.baseline.reject(new Error('execution limit'));
  await flush();
  assert.ok(events.some((event) => event[0] === 'result' && event[1] === 'antroute'));
  assert.ok(events.some((event) => event[0] === 'error' && event[1] === 'baseline'));
  controller.abort();
});

test('late results after editing the trip or unmounting are ignored', async () => {
  const { jobs, events, controller } = startBatch();
  await flush();
  controller.abort();
  jobs.baseline.resolve('old trip');
  jobs.antroute.reject(new Error('old error'));
  await flush();
  assert.deepEqual(events, []);
});

test('an aborted batch does not submit jobs', async () => {
  const controller = new AbortController();
  controller.abort();
  let calls = 0;
  const request = async () => { calls++; return 'unexpected'; };
  requestRoutesIndependently(
    { normal: request, antroute: request, baseline: request },
    { onResult: () => calls++, onError: () => calls++, onSettled: () => calls++ }, controller.signal
  );
  await flush();
  assert.equal(calls, 0);
});

function fakeClient(states) {
  const requests = [];
  return {
    requests,
    async post(url, payload, config) {
      requests.push(['POST', url, payload, config]);
      return { data: { job_id: 'job-1', status: 'queued' } };
    },
    async get(url, config) {
      requests.push(['GET', url, config]);
      const next = states.shift();
      if (next instanceof Error) throw next;
      assert.ok(next, 'unexpected extra status request');
      return { data: { job_id: 'job-1', ...next } };
    },
  };
}

test('queued and running jobs are polled with short request timeouts', async () => {
  const result = { routes: [{ algorithm: 'iaco' }], notice: null };
  const client = fakeClient([{ status: 'queued' }, { status: 'running' }, { status: 'completed', result }]);
  assert.deepEqual(await waitForRouteJob(client, { model: 'baseline' }, { pollIntervalMs: 0 }), result);
  assert.equal(client.requests[0][3].timeout, 15000);
  assert.equal(client.requests[1][2].timeout, 15000);
  assert.equal(client.requests.filter((item) => item[0] === 'POST').length, 1);
});

test('a temporary tunnel error retries status without resubmitting the calculation', async () => {
  const client = fakeClient([new Error('temporary tunnel failure'), { status: 'completed', result: { routes: [] } }]);
  await waitForRouteJob(client, {}, { pollIntervalMs: 0 });
  assert.equal(client.requests.filter((item) => item[0] === 'POST').length, 1);
  assert.equal(client.requests.filter((item) => item[0] === 'GET').length, 2);
});

test('repeated transport failures surface a status error without creating duplicate jobs', async () => {
  const client = fakeClient([new Error('offline'), new Error('offline'), new Error('offline')]);
  await assert.rejects(waitForRouteJob(client, {}, { pollIntervalMs: 0 }), /calculation may still be running/);
  assert.equal(client.requests.filter((item) => item[0] === 'POST').length, 1);
});

test('a failed worker job is surfaced rather than turned into an empty successful route', async () => {
  const client = fakeClient([{ status: 'failed', error: 'Worker execution limit' }]);
  await assert.rejects(waitForRouteJob(client, {}), /Worker execution limit/);
});

test('a legitimate empty baseline retains its explanatory notice', async () => {
  const result = { routes: [], notice: 'Outside scored network' };
  const client = fakeClient([{ status: 'completed', result }]);
  assert.deepEqual(await waitForRouteJob(client, {}), result);
});

test('unknown or expired jobs are not retried forever', async () => {
  const error = Object.assign(new Error('expired'), { response: { status: 404 } });
  const client = fakeClient([error]);
  await assert.rejects(waitForRouteJob(client, {}), /expired/);
  assert.equal(client.requests.length, 2);
});

test('cancellation stops waiting between status requests', async () => {
  const controller = new AbortController();
  const client = fakeClient([{ status: 'running' }]);
  const pending = waitForRouteJob(client, {}, { signal: controller.signal, pollIntervalMs: 10000 });
  await flush();
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(client.requests.length, 2);
});

test('completed jobs must actually contain a result', async () => {
  const client = fakeClient([{ status: 'completed', result: null }]);
  await assert.rejects(waitForRouteJob(client, {}), /without a route result/);
});

test('a polling budget expires without resubmitting the accepted job', async () => {
  const client = fakeClient([{ status: 'running' }]);
  await assert.rejects(
    waitForRouteJob(client, {}, { pollIntervalMs: 5, maxWaitMs: 1 }), /still unfinished/
  );
  assert.equal(client.requests.filter((item) => item[0] === 'POST').length, 1);
});
