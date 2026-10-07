# Route jobs: exact setup and verification

Based on `trial` commit `236e82b8b9cc78964d4bdfa34271cd59069f6464`. Proposed branch: `fix/ANT-19-baseline-route-jobs`; issue: [#19](https://github.com/Traffic-Forecasting-Thesis-Group/ANTROUTE/issues/19).

When IACO takes several minutes, HomeScreen must display successful ANTRoute routes immediately. This implementation submits jobs, polls short status requests, and gives the baseline its own process worker and queue. Baseline computation may still take minutes; this change removes the HTTP request's dependency on that runtime.

## 1. Get the complete implementation

In your existing ANTROUTE checkout, save your current tracked code changes before switching branches. `git status --short` shows whether you have edits to preserve. The branch switch will refuse to overwrite conflicting local edits; do not use a forced checkout. Keep your existing local traffic data and configuration.

```bash
git status --short
git fetch origin fix/ANT-19-baseline-route-jobs
git switch --create review-route-jobs --track origin/fix/ANT-19-baseline-route-jobs
```

If that local branch already exists, use `git switch review-route-jobs` instead. The complete changed code is in this branch; no snippet needs to be pasted over your local HomeScreen. Review any local-only UI additions separately.

## 2. Check the data and configuration

Use the same graph, scored risk snapshot, baseline traffic inputs, and configuration as your evaluation. Both workers mount `traffic_system` at `/app`, so use paths relative to that directory in `traffic_system/.env`. Merge these values into the existing file rather than replacing it:

```dotenv
RISK_EDGES_PATH=data/processed/risk_scores/risk_edges.csv
VEHICLE_COUNTS_DIR=data/processed/vehicle_counts
BASELINE_FALLBACK=false
EVENT_MU=0.0
ROUTE_JOB_TIME_LIMIT_SECONDS=3600
ROUTE_JOB_TTL_SECONDS=86400
```

The scored CSV is a local asset and is not supplied by this Git branch. Keep the existing spatial/traffic assets under `traffic_system/data`. Supply the vehicle counts used in the evaluation; otherwise the existing baseline implementation uses zero for its flow term.

Compose sets `CELERY_BROKER_URL=redis://redis:6379/1` and `CELERY_RESULT_BACKEND=redis://redis:6379/2` for the API and both workers. Redis database 0 continues serving the existing OSRM cache. Do not point one component at a different broker/result database.

The one-hour execution limit is a service limit, not an IACO iteration reduction. An interrupted calculation is a failed job and returns no completed route. Use the separate evaluation harness for runs that exceed this service limit.

## 3. Start the backend and both workers

Run from the repository root with Docker Desktop running, using Linux containers. The worker pool runs inside Docker on Windows too.

```bash
docker compose config
docker compose up -d --build db redis backend worker-primary worker-baseline
docker compose ps
docker compose logs --tail=80 backend worker-primary worker-baseline
```

The new dependency is `celery[redis]==5.6.3`. The two worker commands are:

```bash
celery -A app.task_queue:celery_app worker --loglevel=INFO --pool=prefork --concurrency=1 --queues=routes-primary --hostname=primary@%h
celery -A app.task_queue:celery_app worker --loglevel=INFO --pool=prefork --concurrency=1 --queues=routes-baseline --hostname=baseline@%h
```

These are already configured in Compose; do not start additional copies by hand. A baseline job cannot occupy the worker consuming `routes-primary`. Each worker loads its own data, so keep concurrency at one initially and check the machine's memory use.

Check API readiness:

```bash
curl http://127.0.0.1:8000/health
```

Expected body: `{"status":"ok"}`. Starting only Uvicorn is insufficient for the updated app: submitted jobs need their workers to execute.

## 4. Configure the phone's backend tunnel

The Expo tunnel serves the app bundle. Expose backend port 8000 separately using your existing backend tunnel. For an already configured ngrok installation:

```bash
ngrok http 8000
```

In `frontend/.env.local`, set the actual HTTPS forwarding URL, without `/routes` at the end:

```dotenv
EXPO_PUBLIC_API_URL=https://YOUR-ACTUAL-BACKEND-TUNNEL-DOMAIN
```

Open `https://YOUR-ACTUAL-BACKEND-TUNNEL-DOMAIN/health` on the phone and check for `{"status":"ok"}`. If your existing backend tunnel already works, keep its URL and running process.

Start the frontend:

```bash
cd frontend
npm install
npx expo start --tunnel --clear
```

Keep Docker, the backend tunnel, and Expo running. Update `.env.local` and reload Expo whenever the backend tunnel URL changes.

## 5. What the code now does

| File | Change |
| --- | --- |
| `traffic_system/app/task_queue.py` | Redis/Celery configuration, finite network timeouts, task tracking and result expiry. |
| `traffic_system/app/route_jobs.py` | Enqueue on the model's queue; read state without waiting; distinguish unknown/expired IDs. |
| `traffic_system/app/route_plan.py` | Reuse the existing engine and geometry handling; run legacy synchronous computation off the API event loop. |
| `traffic_system/app/route_tasks.py` | Calculate in a worker process and log job ID, model and elapsed seconds. |
| `traffic_system/app/routers/routes.py` | Add `POST /routes/jobs` (202) and `GET /routes/jobs/{job_id}`. Keep `/routes/plan` compatible. |
| `traffic_system/app/schemas_route.py`, `app/config.py` | Job response schema and configurable service limits. |
| `docker-compose.yml`, `requirements.txt` | Add separate process workers, Redis persistence, and Celery. |
| `frontend/src/api/routeJobs.ts` | Poll with 15-second HTTP timeouts, 1.5-second intervals, cancellation and up to three consecutive status transport failures. |
| `frontend/src/api/routeService.ts` | Preserve the public route function/result types while using job submission and polling. |
| `frontend/src/api/routeBatch.ts`, `screens/HomeScreen.tsx` | Publish each successful result independently; isolate baseline errors; stop stale polling when the trip changes or screen unmounts. |

Job status is `queued`, `running`, `completed`, or `failed`. A completed result may contain no baseline route with an explanatory `notice`; that is distinct from a worker failure. Completed routes retain their actual geometry and algorithm labels, including an explicitly enabled shortest-distance fallback.

The central frontend behavior is:

```typescript
requestRoutesIndependently(
  {
    normal: signal => planRoute(origin, destinations, false, coords, 'antroute', signal),
    antroute: signal => planRoute(origin, destinations, true, coords, 'antroute', signal),
    baseline: signal => planRoute(origin, destinations, true, coords, 'baseline', signal),
  },
  { onResult, onError, onSettled },
  controller.signal
);
```

Each `onResult` updates its model immediately. There is no aggregate promise whose rejection discards successful results. A temporary status-poll failure retries GET, not POST. Canceling a screen stops polling; it does not kill an already running worker calculation. New submissions still calculate fresh routes; there is no cross-trip route-result cache.

## 6. Run the checks

Backend contract tests, from the repository root:

```bash
docker compose exec backend python -m unittest discover -s tests -p test_route_jobs.py -v
```

Frontend tests and full type checking, from `frontend` (use Node 24 for the supplied test command):

```bash
npm run test:routes
npx --no-install tsc --noEmit
```

Then test real workers and the API with the trip that previously timed out. Copy `docs/route-job-trip.example.json` to a local file and replace the origin/destination coordinates with that trip's actual coordinates. The example coordinates only exercise the API; they are not benchmark evidence.

From the repository root:

```bash
docker compose cp docs/route-job-trip.example.json backend:/tmp/route-job-trip.json
docker compose exec backend python scripts/check_route_jobs.py --request-file /tmp/route-job-trip.json
```

Use the path to your edited trip file in the first command to reproduce the actual failure. The smoke test starts a baseline job, waits for its worker to start, submits ANTRoute, polls both jobs, and measures API health responses throughout. It rejects ANTRoute placeholder routes, so missing scored data cannot silently pass as a real graph computation. A baseline with no feasible route is reported with its notice.

In the app, verify:

1. ANTRoute appears and can be selected while the baseline is still calculating.
2. The Baseline tab says `Calculating baseline…` until its own result arrives.
3. A baseline failure leaves ANTRoute visible and shows the failure in the baseline area.
4. Changing the trip prevents a late result from the old trip appearing in the new comparison.
5. The backend `/health` remains reachable during baseline computation.

Watch timing logs with:

```bash
docker compose logs -f worker-primary worker-baseline
```

Verification performed while preparing this change: 10 Python contract tests and 13 frontend regression tests passed; Python compilation, JSON parsing and Compose YAML parsing passed. Full TypeScript checking, Docker startup, live Redis/Celery integration, phone/tunnel behavior and actual-road runtime were not executed in the preparation environment. Run those checks before merging.

## 7. Keep the thesis comparison valid

`baseline_iaco.py`, `baseline_router.py` and `BASELINE_CONFIG=IacoConfig(seed=0)` are unchanged. IACO still uses 20 ants and 60 iterations per leg. The ANTRoute configuration remains unchanged too. The app's existing `optimize_stop_order` flag is not implemented by the backend, so the former `Normal route (no traffic optimization)` card is now labeled `ANTRoute reference route`; it is not a separate no-traffic baseline.

Freeze the scored snapshot and baseline input files during a comparison. The existing in-process data caches do not notice file replacement automatically. After updating traffic assets, model settings or worker code, restart both workers together:

```bash
docker compose restart worker-primary worker-baseline
```

Report application wait time, queue wait, worker computation and algorithm runtime separately. The worker's `elapsed_seconds` includes data loading on a cold worker; it is not a pure IACO search timer. Report each algorithm's actual budget (currently ANTRoute 8 x 15, IACO 20 x 60), use paired origins/destinations and seeds in the evaluation harness, and treat any matched-budget experiment as a separate experiment. Job scheduling alone supplies no evidence that IACO converges faster or that ANTRoute's route quality improves.

Profile IACO on the actual road graph before changing its implementation. Measure dead ends, forward/backtracking steps, tours completed, time per tour, and selection-function cost. Any later pruning, graph reduction, compiled loop or parameter adjustment needs separate feasibility, route-quality and seeded-equivalence validation. Do not quietly reduce ants/iterations or label a shortest-distance replacement as IACO to make a demo finish sooner.

## References

- [FastAPI: heavy background computation and Celery](https://fastapi.tiangolo.com/tutorial/background-tasks/#caveat)
- [Celery: routing tasks to dedicated queues](https://docs.celeryq.dev/en/stable/userguide/routing.html)
- [Celery: Redis broker and result backend](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html)
- [Expo: environment variables](https://docs.expo.dev/guides/environment-variables/)
- [Expo CLI: tunneling](https://docs.expo.dev/more/expo-cli/#tunneling)
