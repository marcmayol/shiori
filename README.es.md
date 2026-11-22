# Shiori

*栞, «punto de lectura»; originalmente las ramas dobladas que marcaban el camino en el bosque.*

**[English](README.md)**

Proxy local delante de las APIs de LLM (Anthropic, OpenAI, Gemini) para reducir coste y latencia sin perder calidad. Un solo binario en Rust, sin runtime, para Linux, macOS y Windows.

Funciona con **Claude Code** y **Codex**, también con sus **suscripciones** (claude.ai Pro/Max, ChatGPT): Shiori reenvía las credenciales sin tocarlas, así que el inicio de sesión que use la herramienta sigue valiendo.

> **Estado: 0.1, fase 1.** Shiori pasa cada petición tal cual, con streaming, y apunta tokens, coste y ahorro de la caché del proveedor por proyecto. Las capas de optimización (caché exacta, optimización del prompt caching, router, caché semántica y compactación) llegan en las siguientes versiones.

## Para empezar

```sh
cargo install --path .
shiori run claude        # Claude Code a través de Shiori, con esta carpeta como proyecto
shiori metrics           # tokens, coste y ahorro por caché
```

`shiori run` arranca el proxy en segundo plano si no está en marcha y lanza el programa con `ANTHROPIC_BASE_URL` y `OPENAI_BASE_URL` apuntando a él. Sirve para cualquier programa: `shiori run codex`, `shiori run -- python agente.py`.

Sin `shiori run`, apunta tú el cliente al proxy:

| Cliente | Variable |
|---|---|
| Claude Code, SDK de Anthropic | `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` |
| SDK de OpenAI, Codex con API key | `OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1` |
| Codex con la cuenta de ChatGPT | `OPENAI_BASE_URL=http://127.0.0.1:8787/chatgpt/codex` |
| API de Gemini | URL base `http://127.0.0.1:8787/gemini` |

## Comandos

| Comando | Qué hace |
|---|---|
| `shiori serve` | El proxy en primer plano |
| `shiori run <programa>` | Lanza un programa a través de Shiori (y lo arranca en segundo plano si hace falta) |
| `shiori status` / `shiori stop` | ¿Está en marcha? Parar el de segundo plano |
| `shiori metrics [--project P] [--since 7d] [--json]` | Tokens, coste y ahorro por proyecto, modelo y tipo de acceso |
| `shiori config validate [archivo]` | Revisa un archivo de configuración; los errores dicen la línea exacta |
| `shiori config show` / `path` / `schema` | Configuración en uso, dónde están los archivos, esquema JSON |
| `shiori bench` | Mide lo que añade Shiori frente a llamar directo |
| `shiori fake` | API falsa de Anthropic, OpenAI y Gemini para probar sin gastar tokens |

## Proyectos

Cada petición cuenta para un proyecto, en este orden:

1. la cabecera `x-shiori-project`;
2. `project:` en el archivo del proyecto (`shiori.yaml` en la carpeta);
3. el nombre de la carpeta;
4. `default`.

`shiori run` manda la carpeta actual en la URL (`/_shiori/dir/<carpeta>/…`), así funciona con clientes que no pueden poner cabeceras. La cabecera `x-shiori-dir` hace lo mismo.

## Configuración

YAML, en dos archivos. Los dos son opcionales.

- **Global**: `config.yaml` en la carpeta de configuración de Shiori (la dice `shiori config path`), o el que indique `SHIORI_CONFIG`.
- **Proyecto**: `shiori.yaml` en la carpeta del proyecto. Solo fija lo del proyecto (`project`, `layers`). Puede venir en un repositorio ajeno, así que no puede cambiar el servidor ni los destinos: si lo intenta es error, no se ignora.

```yaml
# config.yaml (global)
server:
  host: 127.0.0.1      # solo este equipo
  port: 8787
  max_body_mb: 256
upstreams:             # dónde está de verdad cada API
  anthropic: https://api.anthropic.com
  openai: https://api.openai.com
  chatgpt: https://chatgpt.com/backend-api
  gemini: https://generativelanguage.googleapis.com
metrics:
  enabled: true
prices:                # dólares por millón de tokens, por prefijo del modelo
  gpt-5: { input: 1.25, output: 10 }
```

Prioridad, de menos a más: valores por defecto, archivo global, variables de entorno (`SHIORI_HOST`, `SHIORI_PORT`, `SHIORI_METRICS`, `SHIORI_METRICS_PATH`, `SHIORI_UPSTREAM_ANTHROPIC`/`OPENAI`/`CHATGPT`/`GEMINI`, `SHIORI_DATA_DIR`) y, para lo que se decide en cada petición, el archivo del proyecto y las cabeceras. Las claves desconocidas son error.

También se puede configurar con TOML: [mira cómo](docs/toml.md).

Para llegar a las APIs, Shiori hace caso a las variables del proxy del sistema (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`), que es lo que hace falta detrás de un proxy de empresa. Los destinos de este mismo equipo (`127.0.0.1`, `::1`, `localhost`) van siempre directos. Por `http://` sin cifrar, el proxy ve las cabeceras de la petición, claves de la API incluidas; los destinos que no deben pasar por él van en `NO_PROXY`.

## Métricas y coste

Shiori lee el uso que devuelve cada API (entrada, salida, lectura y escritura de caché) sin tocar la respuesta y guarda una fila por llamada en SQLite. El coste sale de la tabla integrada de precios de Anthropic; los demás modelos se añaden en `prices:`. Con suscripción, el coste es el equivalente de la API, no lo que pagas: la columna `auth` separa `api_key` de `subscription`.

Para leer el uso, Shiori pide la respuesta sin comprimir (`accept-encoding: identity`). Entre tu cliente y el proxy, en el mismo equipo, comprimir no ahorra nada.

## Rendimiento

`shiori bench` arranca la API falsa en el mismo proceso y compara llamar directo con llamar a través de Shiori: latencia p50/p99, tiempo hasta el primer byte en streaming, peticiones por segundo y memoria. En un Apple M5, con métricas, Shiori añade unos 0,2-0,5 ms por petición JSON y no retrasa el primer byte del streaming.

## Desarrollo

```sh
cargo test                       # tests unitarios, del proxy y de la línea de comandos
cargo bench                      # microbenchmarks con Criterion
cargo run -- bench               # batería de rendimiento
```

Arquitectura hexagonal: `dominio` (proveedores, uso, precios, puertos), `aplicacion` (el proxy), `adaptadores` (entrada axum, salida reqwest, métricas en SQLite) y `configuracion`.

## Licencia

MIT
