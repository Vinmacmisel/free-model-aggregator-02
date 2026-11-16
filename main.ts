// free-model-aggregator-02: one server, two client wires, the best free model behind every answer.
//
// The design, in short:
//   - routing is quality-first: for each arriving prompt, rank the brief's free models that meet the
//     prompt's requirements by the brief's published rating (highest first, id order breaking ties),
//     and dial the best one. The sealed ratings say the best plain/tools model is Inkling 2 Small
//     (1475) and the best JSON-capable model is Nemotron 3.5 Super VL (1303); this server derives
//     both from the brief at runtime instead of hard-coding them.
//   - failover is per-model and cheap: a refusal that may be transient (a 429, a 5xx, a dropped
//     connection) is retried once on the same model, and only then does the prompt fail over to the
//     next candidate. At most six gateway attempts per prompt, shared by the first dial and any
//     redial after an unusable answer, so a trial stays far under the gateway's 60-call cap, and
//     refused calls never count toward the tie-break anyway. A model that refused on one prompt is
//     tried fresh on the next: a refusal costs nothing but the call, and a limit that lifted is
//     worth the retry. Every refusal of a prompt is reported on its answer in the
//     `x-makefun-upstream-failures` header, the only way the failover column can see a refusal the
//     server recovered from.
//   - the wire is the client's, whole: OpenAI JSON and SSE pass the provider's answer through with
//     the dialled model id put back on every object and chunk; Anthropic JSON and SSE are translated
//     both ways with the stop reason, usage and tool calls carried over. Some free models omit usage,
//     and the env requires a whole input count (>= 0) and an output count (>= 1), so a missing usage
//     is filled from the request and answer text, never invented from nothing.
//   - determinism: every choice is a function of the brief, the requests and the gateway's answers.
//     No wall clock and no unseeded randomness anywhere in a request path; the pauses between two
//     attempts of one prompt are fixed, and the poll of GET /work only decides when to exit.
//
// The trial's shape (from the kit): the server listens on 0.0.0.0 at a free port, registers it with
// POST $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>}, serves
// GET /v1/models (OpenAI shape for Authorization: Bearer, Anthropic shape for x-api-key),
// POST /v1/chat/completions and POST /v1/messages, dials every prompt through the gateway's Chat
// Completions route at $OPENAI_BASE_URL, polls GET $MAKEFUN_ENV_URL/work and exits 0 when it says
// done. `node main.ts --list` prints the catalog and exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  ensureAnthropicUsage,
  ensureOpenAIUsage,
  estimateTokens,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway attempts one prompt may cost; six prompts stay far under the 60-call cap. */
const MAX_ATTEMPTS_PER_PROMPT = 6;
/** Fixed pause between two attempts of one prompt; never written into any request. */
const RETRY_DELAY_MS = 250;
/** How often POST /register is retried while the env is not answering yet. */
const REGISTER_ATTEMPTS = 20;
/** Fixed pause between two register attempts. */
const REGISTER_DELAY_MS = 500;
/** A request without an output cap gets one: the gateway refuses uncapped calls. */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** A fixed pause; waiting is fine, a time is never written into anything sent. */
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** Rank two candidates: the higher rating first, the lower id first on a tie. */
function byRank(a: ModelRow, b: ModelRow): number {
  if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
}

/**
 * The free models that can serve one prompt's requirements, best first: rating order, id order on a
 * tie. A model is only ever picked from this list, so the order is a pure function of the brief and
 * the models already attempted on this prompt.
 */
function rankedCandidates(requires: Requirement, excluded: ReadonlySet<string>): ModelRow[] {
  return freeModels(brief)
    .filter((model) => !excluded.has(model.id) && meetsRequirements(requires, capabilitiesOf(model)))
    .sort(byRank);
}

/** One refusal, in the shape the `x-makefun-upstream-failures` header carries. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

interface DialSuccess {
  ok: true;
  model: string;
  response: Response;
  failures: FailureRow[];
  /** Gateway attempts this dial spent, so a redial shares the prompt's budget. */
  attempts: number;
}

interface DialFailure {
  ok: false;
  failures: FailureRow[];
  lastStatus: number;
  lastCode: string;
  attempts: number;
}

type DialOutcome = DialSuccess | DialFailure;

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The refusal trail as a response header: at most 8 rows, in the order the calls refused. */
function failuresHeader(failures: readonly FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly FailureRow[] = []): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...failuresHeader(failures),
  });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  failures: readonly FailureRow[] = [],
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway error body's code, or '' when the body carries none. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = value.error.code;
    if (typeof code === 'string' && code !== '') return code.slice(0, 64);
    const type = value.error.type;
    if (typeof type === 'string' && type !== '') return type.slice(0, 64);
    const message = value.error.message;
    if (typeof message === 'string' && message !== '') return message.slice(0, 64);
  } catch {
    // Not JSON: no code to read.
  }
  return '';
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/**
 * An OpenAI-shaped client body as the gateway call: the dialled model, an output cap (the gateway
 * refuses uncapped calls) and usage on streams (the client checks usage on every answer).
 */
function prepareOpenAI(body: Json): Json {
  const out: Json = { ...body };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = DEFAULT_MAX_TOKENS;
  if (out.stream === true && !isRecord(out.stream_options)) out.stream_options = { include_usage: true };
  return out;
}

/**
 * Dial one prompt through the gateway with failover. The attempts walk the ranked candidates: a
 * refusal that may be transient (a 429, a 5xx, a dropped connection) is retried once on the same
 * model, and only then does the prompt fail over to the next candidate; any other 4xx moves on at
 * once. The outcome is decided only by the brief and the gateway's answers, never by the clock, so
 * a replay takes the same path. `extra.maxAttempts` caps this dial's attempts, so a redial after
 * an unusable answer shares the prompt's budget of six.
 */
async function dialWithFailover(
  upstreamBody: Json,
  requires: Requirement,
  extra?: { exclude?: ReadonlySet<string>; priorFailures?: readonly FailureRow[]; maxAttempts?: number },
): Promise<DialOutcome> {
  const budget = extra?.maxAttempts ?? MAX_ATTEMPTS_PER_PROMPT;
  const failures: FailureRow[] = [...(extra?.priorFailures ?? [])];
  const attempted = new Set<string>(extra?.exclude ?? []);
  const retried = new Set<string>();
  let retryModel: string | null = null;
  let attempts = 0;
  while (attempts < budget) {
    let model: ModelRow | null = null;
    if (retryModel !== null) {
      model = freeModels(brief).find((m) => m.id === retryModel) ?? null;
      retryModel = null;
    } else {
      const candidates = rankedCandidates(requires, attempted);
      model = candidates.length > 0 ? candidates[0] : null;
    }
    if (model === null) break;
    attempted.add(model.id);
    attempts += 1;
    let upstream: Response;
    try {
      upstream = await dial(model.id, { ...upstreamBody, model: model.id });
    } catch {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      if (!retried.has(model.id)) {
        retried.add(model.id);
        retryModel = model.id;
      }
      if (attempts < budget) await sleep(RETRY_DELAY_MS);
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      return { ok: true, model: model.id, response: upstream, failures, attempts };
    }
    const text = await upstream.text().catch(() => '');
    const code = errorCodeOf(text) !== '' ? errorCodeOf(text) : `http_${upstream.status}`;
    failures.push({ status: upstream.status, code, model: model.id });
    if ((upstream.status === 429 || upstream.status >= 500) && !retried.has(model.id)) {
      retried.add(model.id);
      retryModel = model.id;
    }
    if (attempts < budget) await sleep(RETRY_DELAY_MS);
  }
  const last = failures[failures.length - 1];
  return { ok: false, failures, lastStatus: last?.status ?? 502, lastCode: last?.code ?? 'upstream_error', attempts };
}

/** The text of an OpenAI answer, for the usage estimate when the provider omits usage. */
function openAIAnswerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
  }
  return '';
}

/** Whether a 200 OpenAI answer is usable: a message with text, or a tool call the client can read. */
function usableOpenAIAnswer(answer: Json): boolean {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return false;
  if (message.role !== 'assistant') return false;
  if (openAIAnswerText(answer).trim() !== '') return true;
  return Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
}

/** Whether a translated Anthropic answer is usable: text to read, or a tool call the client can read. */
function usableAnthropicAnswer(message: Json): boolean {
  const content = Array.isArray(message.content) ? message.content : [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text.trim() !== '') return true;
    if (block.type === 'tool_use') return true;
  }
  return false;
}

/** The text of a translated Anthropic answer, for the usage estimate. */
function anthropicAnswerText(message: Json): string {
  const content = Array.isArray(message.content) ? message.content : [];
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** A 200 answer that carries no usable completion is a failure: fail over once more, then give up. */
async function redialAfterUnusable(
  upstreamBody: Json,
  requires: Requirement,
  outcome: DialSuccess,
): Promise<DialOutcome> {
  return dialWithFailover(upstreamBody, requires, {
    exclude: new Set([outcome.model]),
    priorFailures: [...outcome.failures, { status: 200, code: 'empty_answer', model: outcome.model }],
    maxAttempts: Math.max(1, MAX_ATTEMPTS_PER_PROMPT - outcome.attempts),
  });
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const upstreamBody = prepareOpenAI(body);
  const inputEst = estimateTokens(JSON.stringify(body).length);
  let outcome = await dialWithFailover(upstreamBody, requires);
  if (!outcome.ok) {
    return sendError(res, 'openai', 502, outcome.lastCode, 'the gateway refused every free model that meets this request', outcome.failures);
  }
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...failuresHeader(outcome.failures),
    });
    try {
      for await (const frame of openAIStreamWithModel(outcome.response.body, outcome.model, inputEst)) res.write(frame);
    } catch {
      // The client went away mid-stream: nothing more to send it.
    }
    res.end();
    return;
  }
  let answer: unknown;
  try {
    answer = await outcome.response.json();
  } catch {
    const retry = await redialAfterUnusable(upstreamBody, requires, outcome);
    if (!retry.ok) {
      return sendError(res, 'openai', 502, retry.lastCode, 'the gateway refused every free model that meets this request', retry.failures);
    }
    outcome = retry;
    try {
      answer = await outcome.response.json();
    } catch {
      return sendError(res, 'openai', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', outcome.failures);
    }
  }
  if (!isRecord(answer) || !usableOpenAIAnswer(answer)) {
    const retry = await redialAfterUnusable(upstreamBody, requires, outcome);
    if (!retry.ok) {
      return sendError(res, 'openai', 502, retry.lastCode, 'the gateway refused every free model that meets this request', retry.failures);
    }
    try {
      const second: unknown = await retry.response.json();
      if (!isRecord(second) || !usableOpenAIAnswer(second)) {
        return sendError(res, 'openai', 502, 'empty_answer', 'the gateway answered no usable completion', retry.failures);
      }
      answer = second;
      outcome = retry;
    } catch {
      return sendError(res, 'openai', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', retry.failures);
    }
  }
  const outputEst = estimateTokens(openAIAnswerText(isRecord(answer) ? answer : {}).length);
  const fixed = ensureOpenAIUsage(isRecord(answer) ? answer : {}, inputEst, outputEst);
  sendJson(res, 200, openAIJsonWithModel(fixed, outcome.model), outcome.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const inputEst = estimateTokens(JSON.stringify(body).length);
  const translated = anthropicToOpenAI(body, '');
  let outcome = await dialWithFailover(translated, requires);
  if (!outcome.ok) {
    return sendError(res, 'anthropic', 502, outcome.lastCode, 'the gateway refused every free model that meets this request', outcome.failures);
  }
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...failuresHeader(outcome.failures),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(outcome.response.body, outcome.model)) res.write(event);
    } catch {
      // The client went away mid-stream: nothing more to send it.
    }
    res.end();
    return;
  }
  let answer: unknown;
  try {
    answer = await outcome.response.json();
  } catch {
    const retry = await redialAfterUnusable(translated, requires, outcome);
    if (!retry.ok) {
      return sendError(res, 'anthropic', 502, retry.lastCode, 'the gateway refused every free model that meets this request', retry.failures);
    }
    outcome = retry;
    try {
      answer = await outcome.response.json();
    } catch {
      return sendError(res, 'anthropic', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', outcome.failures);
    }
  }
  if (!isRecord(answer)) {
    const retry = await redialAfterUnusable(translated, requires, outcome);
    if (!retry.ok) {
      return sendError(res, 'anthropic', 502, retry.lastCode, 'the gateway refused every free model that meets this request', retry.failures);
    }
    try {
      const second: unknown = await retry.response.json();
      if (!isRecord(second)) {
        return sendError(res, 'anthropic', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', retry.failures);
      }
      answer = second;
      outcome = retry;
    } catch {
      return sendError(res, 'anthropic', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', retry.failures);
    }
  }
  const firstTranslated = openAIToAnthropic(answer, outcome.model);
  if (!usableAnthropicAnswer(firstTranslated)) {
    const retry = await redialAfterUnusable(translated, requires, outcome);
    if (!retry.ok) {
      return sendError(res, 'anthropic', 502, retry.lastCode, 'the gateway refused every free model that meets this request', retry.failures);
    }
    try {
      const second: unknown = await retry.response.json();
      if (!isRecord(second)) {
        return sendError(res, 'anthropic', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', retry.failures);
      }
      const secondTranslated = openAIToAnthropic(second, retry.model);
      if (!usableAnthropicAnswer(secondTranslated)) {
        return sendError(res, 'anthropic', 502, 'empty_answer', 'the gateway answered no usable completion', retry.failures);
      }
      sendJson(
        res,
        200,
        ensureAnthropicUsage(secondTranslated, inputEst, estimateTokens(anthropicAnswerText(secondTranslated).length)),
        retry.failures,
      );
      return;
    } catch {
      return sendError(res, 'anthropic', 502, 'bad_upstream_json', 'the gateway answered a body that is not JSON', retry.failures);
    }
  }
  sendJson(
    res,
    200,
    ensureAnthropicUsage(firstTranslated, inputEst, estimateTokens(anthropicAnswerText(firstTranslated).length)),
    outcome.failures,
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Register, retrying while the env is still starting; a register that fails for good ends the trial. */
async function registerWithRetry(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS; attempt += 1) {
    try {
      await register(port, report);
      return;
    } catch (err) {
      lastError = err;
      if (attempt + 1 < REGISTER_ATTEMPTS) await sleep(REGISTER_DELAY_MS);
    }
  }
  throw lastError;
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await registerWithRetry(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
