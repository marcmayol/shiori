# Shiori

*栞, "bookmark"; originally the bent branches that marked the path through the forest.*

**[Español](README.es.md)**

A local proxy in front of the LLM APIs (Anthropic, OpenAI, Gemini) to cut cost and latency without losing quality. One Rust binary, no runtime, for Linux, macOS and Windows.

It works with **Claude Code** and **Codex**, including their **subscriptions** (claude.ai Pro/Max, ChatGPT): Shiori forwards your credentials untouched, so whatever login the tool uses keeps working.

> **Status: 0.1, phase 1.** Shiori passes every request through unchanged, with streaming, and records tokens, cost and provider-cache savings per project. The optimisation layers (exact cache, prompt-caching optimisation, router, semantic cache, compaction) come in later versions.

## Quick start

```sh
cargo install --path .
shiori run claude        # Claude Code through Shiori, this folder as the project
shiori metrics           # tokens, cost and cache savings
```

`shiori run` starts the proxy in the background if it is not running, then runs the program with `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` pointing to it. It works with any program: `shiori run codex`, `shiori run -- python agent.py`.

Without `shiori run`, point the client at the proxy yourself:

| Client | Variable |
|---|---|
| Claude Code, Anthropic SDKs | `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` |
| OpenAI SDKs, Codex with an API key | `OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1` |
| Codex signed in with ChatGPT | `OPENAI_BASE_URL=http://127.0.0.1:8787/chatgpt/codex` |
| Gemini API | base URL `http://127.0.0.1:8787/gemini` |

## Commands

| Command | What it does |
|---|---|
| `shiori serve` | Run the proxy in the foreground |
| `shiori run <program>` | Run a program through Shiori (starts it in the background if needed) |
| `shiori status` / `shiori stop` | Is it running? Stop the background proxy |
| `shiori metrics [--project P] [--since 7d] [--json]` | Tokens, cost and provider-cache savings per project, model and login kind |
| `shiori config validate [file]` | Check a configuration file; errors give the exact line |
| `shiori config show` / `path` / `schema` | Configuration in effect, where the files are, JSON schema |
| `shiori bench` | Measure what Shiori adds against calling the API directly |
| `shiori fake` | Fake Anthropic/OpenAI/Gemini API, to test without spending tokens |

## Projects

Each request is counted under a project, decided in this order:

1. the `x-shiori-project` header;
2. `project:` in the project file (`shiori.yaml` in the folder);
3. the folder name;
4. `default`.

`shiori run` sends the current folder in the URL (`/_shiori/dir/<folder>/…`), so it works with clients that cannot add headers. The `x-shiori-dir` header does the same.

## Configuration

YAML, in two files. Both are optional.

- **Global**: `config.yaml` in the Shiori config folder (`shiori config path` shows it), or the file named by `SHIORI_CONFIG`.
- **Project**: `shiori.yaml` in the project folder. It only sets what belongs to the project (`project`, `layers`). It may come from someone else's repository, so it cannot change the server or the upstreams: that is an error, not ignored.

```yaml
# config.yaml (global)
server:
  host: 127.0.0.1      # only this computer
  port: 8787
  max_body_mb: 256
upstreams:             # where each API really is
  anthropic: https://api.anthropic.com
  openai: https://api.openai.com
  chatgpt: https://chatgpt.com/backend-api
  gemini: https://generativelanguage.googleapis.com
metrics:
  enabled: true
  # path: /somewhere/metrics.db
prices:                # USD per million tokens, by model id prefix
  gpt-5: { input: 1.25, output: 10 }
```

Priority, lowest to highest: defaults, global file, environment variables (`SHIORI_HOST`, `SHIORI_PORT`, `SHIORI_METRICS`, `SHIORI_METRICS_PATH`, `SHIORI_UPSTREAM_ANTHROPIC`/`OPENAI`/`CHATGPT`/`GEMINI`, `SHIORI_DATA_DIR`), then the project file and the request headers for what applies per request. Unknown keys are errors.

You can also configure Shiori with TOML: [see how](docs/toml.md).

### Outgoing proxy

Shiori reaches the APIs through the same proxy as other programs: `HTTPS_PROXY`, `HTTP_PROXY` and `ALL_PROXY` (or their lowercase forms) in the environment where the proxy starts (`shiori serve`, or the `shiori run` that starts it) and, if none is set, the system proxy (Windows Internet settings, macOS network settings). That is what you need behind a company proxy. Upstreams on this computer (`127.0.0.1`, `::1`, `localhost`) are always reached directly. If that proxy is down, every call fails with a 502 (`cannot reach anthropic: ... tcp connect error`). Over plain `http://`, the proxy sees the request headers, API keys included. List the hosts that must skip it in `NO_PROXY`, for example `NO_PROXY=api.anthropic.com`.

## Metrics and cost

Shiori reads the usage each API returns (input, output, cache read, cache write) without changing the response, and stores one row per call in SQLite. The cost uses the built-in table of Anthropic prices; add other models under `prices:`. With a subscription the cost is the API equivalent, not what you pay: the `auth` column separates `api_key` from `subscription`.

To read the usage, Shiori asks the upstream for an uncompressed response (`accept-encoding: identity`). Between your client and the proxy, on the same computer, compression saves nothing.

## Performance

`shiori bench` runs the fake API in-process and measures direct calls against calls through Shiori: p50/p99 latency, time to first byte when streaming, requests per second and memory. On an Apple M5, with metrics on, Shiori adds about 0.2–0.5 ms per JSON request and does not delay the first streamed byte.

## Development

```sh
cargo test                       # unit, proxy and CLI tests
cargo bench                      # Criterion micro-benchmarks
cargo run -- bench               # performance battery
```

Hexagonal architecture: `dominio` (providers, usage, prices, ports), `aplicacion` (the proxy), `adaptadores` (axum in, reqwest out, SQLite metrics), `configuracion`.

## License

MIT
