# Two services, one codebase

The API runs as **two Render services** built from this same repository. They
share every line of source. They differ in which model bundles the image
carries and which routes `app/main.py` registers, selected by `SERVICE_ROLE`.

| | forecast service | lag service |
|---|---|---|
| Render name | `luxtransport-api` *(unchanged)* | `luxtransport-lag-api` *(new)*  |
| Dockerfile | `Dockerfile` | `Dockerfile.lag` |
| `SERVICE_ROLE` | `forecast` | `lag` |
| Models | `forecast_model_2024.pkl`, `forecast_model_2024_2025.pkl` | `forecast_model_24h.pkl`, `forecast_model_48h.pkl` |
| Needs history from the caller | no | **yes** |
| Routes | `/`, `/health`, `/models`, `/counters`, `/manifest`, `/actuals/{poste_id}`, `/forecast` | `/`, `/health`, `GET /forecast/{lead}h/spec`, `POST /forecast/{lead}h` |
| Worst-case RSS | 346 MB | 280 MB |

## Why

A Render free instance has 512 MB. Measured resident memory, `--workers 1`,
after forcing every bundle to load:

```
forecast   87 MB idle  ->  257 MB one bundle  ->  346 MB both
lag        82 MB idle  ->  256 MB one bundle  ->  280 MB both
```

One process holding all four is `346 + 280 - 85` (the interpreter and libraries
counted once) ≈ **540 MB**, over the cap *before* request overhead. Render does
not raise an error on that — it OOM-restarts the instance, which reads as
requests vanishing and an unexplained cold start. That is a failure mode people
misdiagnose for a week.

Split, each service has real headroom: 166 MB spare on the forecast service,
232 MB on the lag service. Both services keep lazy loading on top of that, so a
service whose callers only ever use one model never pays for the other.

Those numbers are macOS, so read them as the shape rather than exact Linux
figures. They agree with the 327 MB recorded for the two forecast bundles
measured on Render itself.

## Why not two repos

A fork drifts. `forecast/` is already vendored from the training repo precisely
so the API and the training pipeline cannot diverge, and this file's header
records what it cost the last time prediction logic lived in two places. One
codebase, two Dockerfiles, one `render.yaml`.

Each Dockerfile **bakes its role in as `ENV`** and copies **only its own
bundles**. Two consequences worth keeping:

- A role lost from a dashboard edit cannot start a service against the wrong
  models — the image does not contain them.
- `SERVICE_ROLE` is validated at import and an unknown value is **fatal**. A
  typo that fell back to a default would deploy a service serving the wrong
  half of the API and look healthy doing it.

## What changed for clients

Nothing, for anything calling `/forecast`, `/counters`, `/manifest`,
`/actuals`, `/models` or `/health`. Those stay on `luxtransport-api` at its
existing URL. The split moved the short-horizon routes **out**, not these.

**Two routes moved** to `luxtransport-lag-api`:

```
GET  /forecast/{lead}h/spec
POST /forecast/{lead}h            lead = 24 or 48
```

Hitting a moved route on the old service does not return a bare 404. It names
the route's new home, using `COMPANION_URL`, which `render.yaml` fills from the
other service automatically:

```json
{
  "error": "Unknown path: /forecast/48h/spec",
  "hint": "/forecast/48h/spec is a SHORT-HORIZON route and this instance is
           SERVICE_ROLE=forecast. It lives on https://luxtransport-lag-api....",
  "service_role": "forecast",
  "companion_service": "https://luxtransport-lag-api...."
}
```

The reverse holds: `/counters` on the lag service says where to find it. `GET /`
on either service reports its `service_role`, what it serves, its companion, and
only the endpoints it can actually answer.

## Running both locally

```sh
docker compose up --build          # both: forecast :8000, lag :8001
docker compose up --build api      # forecast only
docker compose up --build lag      # short-horizon only
```

Without Docker, `SERVICE_ROLE` picks the half:

```sh
SERVICE_ROLE=forecast uvicorn app.main:app --port 8000
SERVICE_ROLE=lag      uvicorn app.main:app --port 8001
SERVICE_ROLE=all      uvicorn app.main:app --port 8000   # everything, local only
```

`SERVICE_ROLE=all` is the pre-split behaviour and is the default when the
variable is unset, so a local checkout still behaves the way it always did. Do
not deploy it to the free plan — that is the 540 MB configuration.

## Deploying

`scripts/deploy_to_backend.sh` in the training repo is unchanged: it copies all
four bundles into this repository. Each Dockerfile then picks its own two. A
push deploys both services; Render rebuilds only the one whose inputs changed.
