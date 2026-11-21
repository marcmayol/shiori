# Configuring Shiori with TOML

YAML is the main format, but every configuration file can also be TOML with the same keys. Use `config.toml` for the global file and `shiori.toml` for the project file. Keep only one per folder: having both `shiori.yaml` and `shiori.toml` is an error, so it is always clear which one applies.

```toml
# config.toml (global)
[server]
host = "127.0.0.1"
port = 8787
max_body_mb = 256

[upstreams]
anthropic = "https://api.anthropic.com"
openai = "https://api.openai.com"
chatgpt = "https://chatgpt.com/backend-api"
gemini = "https://generativelanguage.googleapis.com"

[metrics]
enabled = true

[prices.gpt-5]
input = 1.25
output = 10.0
```

```toml
# shiori.toml (project)
project = "my-project"

[layers.exact_cache]
enabled = false
```

`shiori config validate shiori.toml` checks it and gives the exact line of each error.
