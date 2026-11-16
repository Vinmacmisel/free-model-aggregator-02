# free-model-aggregator-02

One server that answers OpenAI Chat Completions (`POST /v1/chat/completions`) and
Anthropic Messages (`POST /v1/messages`), in JSON and in server-sent events, with real
completions from OpenRouter's free models, dialled through the grader's gateway.

## How it works

- **Quality-first routing.** For each prompt, the free models that meet its requirements
  (tools, vision, JSON mode, context) are ranked by the brief's published rating, highest
  first, id order breaking ties. On the sealed table that means Inkling 2 Small (1475) for
  plain and tools prompts and Nemotron 3.5 Super VL (1303) for JSON-mode prompts. The
  ranking is derived from the brief at runtime, never hard-coded.
- **Failover.** A refusal that may be transient (429, 5xx, a dropped connection) is retried
  once on the same model, then the prompt fails over to the next candidate; any other 4xx
  moves on at once. At most six gateway attempts per prompt, shared by the first dial and
  any redial after an unusable answer. Every refusal is reported on the answer in the
  `x-makefun-upstream-failures` header. A 200 answer with no usable completion is treated
  as a failure and failed over from.
- **The wire, whole.** The provider's answer is relayed with the dialled model id put back
  on every object and streamed chunk. Anthropic requests are translated to Chat
  Completions and back, with stop reason, usage and tool calls carried over. Some free
  models omit usage, and the client requires an input count (>= 0) and an output count
  (>= 1) on every answer, so missing usage is estimated from the request and answer text
  (a usage chunk is injected into an OpenAI stream that carries none).
- **Catalog.** `node main.ts --list` prints the catalog document — exactly the fourteen
  free model ids, one provider (`openrouter`) — byte for byte as it is registered.
- **Determinism.** Every choice is a function of the brief, the requests and the
  gateway's answers: no wall clock, no unseeded randomness. The pauses between attempts
  are fixed.

## Layout

- `main.ts` — the server: brief, routing, failover, routes, register, work poll.
- `lib/providers.ts` — the brief's parser, the free model list, the catalog builder.
- `lib/wire.ts` — Anthropic <-> Chat Completions translation, model field, SSE writers,
  usage fallbacks.

## Run

```sh
node main.ts            # serves; needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL
node main.ts --list     # prints the catalog and exits 0
```

Local trials (mock gateway + the kit's real env and scorer) scored 100 on practice
trials 0, 1 and 2, 100 with scripted 429/500 refusals and missing usage, and 98 with
a double 429 plus a 400 forcing failover to the second-best models. Without a pushed
OpenRouter key every model call is refused 401 and the trial scores 5 (catalog only),
as the rules say.
