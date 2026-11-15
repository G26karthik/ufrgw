# Feasibility of direct LLM labelling (measured 15 Nov 2026, 08:34-08:48 IST)

Prices are from the official pages `developers.openai.com/api/docs/pricing`, `.../models/gpt-6-astra`, `.../models/gpt-6-luna` and `.../guides/decisions.md`, read on 15 Nov 2026.

| model / endpoint | input $/1M | cached input | cache write (30 min) | output |
|---|---|---|---|---|
| `gpt-6-astra`, `/v1/responses` | 10.00 | 1.00 | 12.50 | 50.00 |
| `gpt-6-luna`, `/v1/responses` | 0.10 | 0.01 | 0.125 | 0.50 |
| `gpt-6-luna`, `/v1/decisions` | 0.10 | 0.01 | 0.125 | not charged |

## Rate limits (response headers on this key)

| endpoint / model | requests/min | tokens/min |
|---|---|---|
| `/v1/responses`, `gpt-6-astra` | 5,000 | 1,000,000 |
| `/v1/decisions`, `gpt-6-luna` | 5,000 | 2,000,000 |

The two endpoints count tokens differently in the header `x-ratelimit-remaining-tokens`:
- `/v1/decisions`: it drops by the total input tokens of the request, cached prefix included (probe 12: 72,600 input tokens, 5,405 of them cached, remaining 1,927,399). Input and cached tokens count against tokens/min.
- `/v1/responses` (`gpt-6-astra`): it drops by the output tokens only (pilot request 1: 36,100 input, 1,018 output, remaining 999,125). With 12 concurrent requests of about 37,000 input tokens each, the minimum was 998,311 of 1,000,000. No HTTP 429 occurred. A limit on input tokens that the header does not show cannot be excluded: nothing above 12 concurrent requests was sent.

## A. `gpt-6-astra`, full context (Responses API): the 525-row pilot

Parameters: `reasoning.effort = "high"`, `max_output_tokens = 8000`, strict `json_schema` output, 25 targets per request, 12 concurrent requests. `temperature` was not sent. Static prefix: 23,679 tokens (50 examples), cached with an explicit breakpoint.

| measure | value |
|---|---|
| requests / invalid | 21 / 0 |
| input tokens per row | 1,509 (562 target + neighbourhood, 947 prefix share) |
| cached input tokens per row | 947 |
| output tokens per row | 43, of which reasoning 27 |
| cost per row | $0.00871 (pilot total $4.574) |
| latency per request | median 17.1 s, max 34.0 s |
| speed | 525 rows in 64 s at 12 concurrent requests = 8.2 rows/s (1.5 rows/s per connection) |

Effort trade-off, one request of 25 targets each (different targets, so only tokens and time compare):

| effort | latency | reasoning tokens | output tokens | valid |
|---|---|---|---|---|
| low (lowest accepted by astra) | 4.9 s | 0 | 388 | yes |
| medium | 7.3 s | 58 | 443 | yes |
| high | 18.1 s | 805 | 1,194 | yes |

Projection for 260,582 rows at effort high (not measured at scale):
- Cost: about $2,270 at 25 targets per request.
- Cost at effort low: about $1,770 ($0.0068 per row, from the 25-target probe with a cached prefix).
- Time: the bound is latency, not tokens/min (see the header finding above). One connection does 1.5 rows/s at effort high. All rows in 45 minutes need about 66 concurrent requests (233 requests/min against a limit of 5,000). This concurrency was not tested.
- If input tokens did count against the 1,000,000 tokens/min limit, the 394M input tokens would take 6.6 hours. An earlier message to the main agent stated this as fact; that was wrong and was corrected at 08:53 IST.
- Not started: the cost is above the $1,000 ask-first threshold.

## B. `gpt-6-luna`, lean prompt (Decisions API): the shipped configuration

One `choice` question per target, 300 targets per request, static prefix 5,405 tokens (26 examples).

| measure | value |
|---|---|
| probes | 2 requests (20 and 200 targets), both valid; latency 1.1 s and 1.3 s |
| pilot | 525 rows, 2 requests, 0 invalid, $0.018 |
| input tokens per row (first 22,800 rows of the full run) | 339: about 173 target row + context rows, about 163 for the question (12 choices), 18 prefix share (cached) |
| output / reasoning tokens | 0 reported, not charged |
| cost per row | $0.0000323 |
| latency per request (300 rows) | median 0.77 s, 95th percentile 1.29 s |
| speed | 94-96 rows/s, set by the tokens/min limit (the client limiter runs at 1.93M tokens/min) |
| HTTP 429 / invalid / errors in the first 76 requests | 0 / 0 / 0 |

Projection for 260,582 rows: 88.4M tokens, about $8.4, about 46 minutes (start 08:44, end about 09:30 IST). Cap set to $12 (1.3 x, rounded up).

Why the context is lean: the budget to finish by 09:40 was about 2M x 50 min / 260,582 = about 380 tokens per row. The question alone costs 163 tokens. The full-context neighbourhood (562 tokens per row) would need about 190M tokens = 97 minutes.

## C. Not measured

- `gpt-6-luna` on `/v1/responses` with the lean prompt (expected about 250 tokens per row; the path was replaced by the Decisions API before any probe).
- `/v1/responses` above 12 concurrent requests. On that endpoint the lean limit was not needed: a `gpt-6-luna` run with full context looks possible on the header evidence, but it was not tried.
- Any accuracy number: Agent A computed no score. See `check/` (Agent B).

## Spend

Probes and pilots together: $6.06 (cap $25). Of that, astra pilot $4.57, astra probes $1.46, luna/Decisions probes and pilot $0.03.
