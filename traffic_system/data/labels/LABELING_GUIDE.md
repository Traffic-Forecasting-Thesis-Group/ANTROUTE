# Tweet labelling guide (ANTROUTE gold set)

You are labelling tweets so the system can tell a **real Metro Manila road incident**
from everything else. Your labels become the test set the classifier is scored against,
so consistency between labelers matters more than speed.

Open your `tweet_gold_<yourname>.csv` in Excel or Google Sheets. Fill the five
`label_*` columns. Leave every other column alone — `tweet_id` is what joins your
work back to the corpus.

Budget about **2 hours for 400 rows**. If a row takes more than ~20 seconds, put your
best guess in and write why in `label_notes`. Do not stall on it.

---

## `label_relevant` — 1 or 0

**1** if the tweet reports a road condition at a **specific place in Metro Manila**:
a crash, a stalled vehicle, flooding, roadwork, a closure or reroute, a rally blocking
a road, or a named-location traffic report.

**0** for everything else. The common cases:

- Another city or country. The corpus is ~43% foreign — Bangalore, Hyderabad, Los
  Angeles, Delaware. If it names no Metro Manila place, it is 0.
- General news that merely mentions traffic ("DOH supports the proposed...", photo
  captions, opinion).
- Personal posts ("stuck in traffic again 😩", "kung nasa Manila pa rin ako, sasama
  ako sa rally") — no specific road condition being reported.
- Announcements about an event's start time being delayed *because of* traffic.
- Weather forecasts with no road named.

> Rule of thumb: **could a router act on this?** If it does not tell you *which road*
> is affected *now*, it is 0.

## `label_type` — only when `label_relevant` is 1, else leave blank

One of: `crash`, `stalled`, `flood`, `roadwork`, `closure`, `rally`, `heavy_traffic`,
`other`.

- `crash` — collision, road crash, sideswipe, hit-and-run, aksidente, banggaan.
- `stalled` — stalled/disabled vehicle, nasiraan, overheated.
- `flood` — flooding, baha, not passable to light vehicles.
- `roadwork` — reblocking, excavation, maintenance, construction.
- `closure` — road closed, rerouting, no entry (including for processions/traslación).
- `rally` — a rally, protest, motorcade, or procession actually occupying a road.
- `heavy_traffic` — a congestion report with no incident named
  (most `@MakatiTraffic` "TRAFFIC UPDATE ... (L)/(M)/(H)" posts).
- `other` — relevant but none of the above.

Pick the **cause**, not the effect. A crash that causes heavy traffic is `crash`.

## `label_location` — copy the location phrase verbatim

Copy the location **exactly as the tweet writes it**: `EDSA Ortigas NB`,
`Jupiter St. from Makati Ave. to N. Garcia`, `C5 Kalayaan`.

Do not normalise, translate, or guess coordinates — the geocoder does that. If several
locations are named, take the one the incident is **at**. If no location is given but
it is otherwise relevant, leave blank and set `label_relevant` to 0 (a router cannot
use it).

## `label_lanes` — 1, 2, or 3

Only when the tweet states how many lanes are occupied ("Two lanes occupied" → `2`).
Blank otherwise. Do not infer.

## `label_notes` — free text

Use it for anything you were unsure about, and for disagreements worth discussing. Rows
with notes are the ones we adjudicate together.

---

## Edge cases, decided once so we both do the same thing

| Situation | Label |
|---|---|
| Retweet/quote of a real MMDA alert | Same as the original — `1` |
| Incident reported as **cleared** ("now passable", "cleared as of...") | `1`, type as the original incident, note "cleared" |
| Forecast or warning ("expect flooding tonight") | `0` — not a current condition |
| Province outside NCR (Bulacan, Cavite, Laguna) | `0` — outside the routable area |
| NLEX/SLEX/Skyway segments **inside** NCR | `1` |
| Multiple incidents in one tweet (MMDA roundups) | `1`, location = the first one, note "multi" |
| Tagalog or mixed Taglish | Label normally — language is not relevance |
| Tweet is an image with no useful text | `0`, note "image only" |

## The overlap block

The first ~200 rows of your sheet are **also in your partner's sheet**. Do not compare
answers on those — they measure how much we agree (Cohen's kappa), which goes in the
paper. Label them independently, then we adjudicate the disagreements together.

## When you are done

Save as CSV (keep the filename) and tell me. Do not reorder or delete rows.
