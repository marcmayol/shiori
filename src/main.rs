use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use tokio::net::TcpListener;

use shiori::adaptadores::metricas_sqlite::{self, Filtro};
use shiori::configuracion::{self, Config, entorno_real};
use shiori::{bench, falso, lanzar, servidor};

#[derive(Parser)]
#[command(
    name = "shiori",
    version,
    about = "Local proxy in front of the LLM APIs (Anthropic, OpenAI, Gemini): cuts cost and latency \
             without losing quality. Works with Claude Code, Codex and their subscriptions."
)]
struct Cli {
    #[command(subcommand)]
    comando: Comando,
}

#[derive(Subcommand)]
enum Comando {
    /// Run the proxy in the foreground
    Serve(OpcionesServidor),
    /// Run a program (claude, codex...) through Shiori, with this folder as the project. Starts the
    /// proxy in the background if it is not running
    Run {
        /// The program and its arguments: `shiori run claude`, `shiori run -- codex exec "hi"`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        comando: Vec<String>,
    },
    /// Is the proxy running? Where are its files?
    Status,
    /// Stop the proxy running in the background
    Stop,
    /// Tokens, cost and provider-cache savings per project, model and kind of login
    Metrics {
        /// Only this project
        #[arg(long)]
        project: Option<String>,
        /// Only since then: `7d`, `24h`, `30m` or a date `2026-11-01`
        #[arg(long)]
        since: Option<String>,
        /// As JSON
        #[arg(long)]
        json: bool,
    },
    /// Check, show or describe the configuration
    #[command(subcommand)]
    Config(ComandoConfig),
    /// Fake server that imitates the Anthropic, OpenAI and Gemini APIs, to test without spending tokens
    Fake {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8788)]
        port: u16,
        /// Output tokens per answer
        #[arg(long, default_value_t = 20)]
        tokens: u32,
        /// Milliseconds between tokens when streaming
        #[arg(long, default_value_t = 0)]
        token_delay_ms: u64,
    },
    /// Measure what Shiori adds against calling the API directly (on the fake server, no tokens spent)
    Bench {
        #[arg(long, default_value_t = 500)]
        requests: usize,
        #[arg(long, default_value_t = 16)]
        concurrency: usize,
        #[arg(long, default_value_t = 50)]
        tokens: u32,
        /// Milliseconds between tokens when streaming
        #[arg(long, default_value_t = 0)]
        token_delay_ms: u64,
        /// Milliseconds the fake API waits before answering
        #[arg(long, default_value_t = 0)]
        first_token_ms: u64,
        /// Measure without recording metrics
        #[arg(long)]
        no_metrics: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Args)]
struct OpcionesServidor {
    /// Overrides server.host
    #[arg(long)]
    host: Option<String>,
    /// Overrides server.port
    #[arg(long)]
    port: Option<u16>,
}

#[derive(Subcommand)]
enum ComandoConfig {
    /// Check a configuration file and say the exact line of each error. Without a path: the global
    /// file and the project file of this folder
    Validate { ruta: Option<PathBuf> },
    /// The configuration in effect (defaults, global file and environment)
    Show,
    /// Where the files are
    Path,
    /// JSON schema of the global file, or of the project file with --project
    Schema {
        #[arg(long)]
        project: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let filtro = tracing_subscriber::EnvFilter::try_from_env("SHIORI_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filtro)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("shiori: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(ejecutar(cli.comando)) {
        Ok(codigo) => ExitCode::from(codigo.clamp(0, 255) as u8),
        Err(e) => {
            eprintln!("shiori: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cargar_config() -> Result<Config, String> {
    configuracion::cargar(&entorno_real).map_err(|e| e.to_string())
}

async fn ejecutar(comando: Comando) -> Result<i32, String> {
    match comando {
        Comando::Serve(op) => {
            let mut config = cargar_config()?;
            if let Some(h) = op.host {
                config.host = h;
            }
            if let Some(p) = op.port {
                config.port = p;
            }
            servir(config).await?;
            Ok(0)
        }
        Comando::Run { comando } => lanzar::ejecutar(&cargar_config()?, &comando).await,
        Comando::Status => {
            let config = cargar_config()?;
            match lanzar::estado(&config).await {
                lanzar::Estado::EnMarcha { version, pid } => {
                    println!(
                        "Shiori {version} is running at {} (pid {pid})",
                        config.url_local()
                    )
                }
                lanzar::Estado::Parado => {
                    println!("Shiori is not running ({} is free)", config.url_local())
                }
                lanzar::Estado::Ocupado => println!(
                    "Another program is listening at {}; Shiori cannot use that port",
                    config.url_local()
                ),
            }
            imprimir_rutas(&config);
            Ok(0)
        }
        Comando::Stop => {
            let config = cargar_config()?;
            if lanzar::parar(&config).await? {
                println!("Shiori stopped");
            } else {
                println!("Shiori was not running at {}", config.url_local());
            }
            Ok(0)
        }
        Comando::Metrics {
            project,
            since,
            json,
        } => {
            let config = cargar_config()?;
            let desde_ms = since.as_deref().map(desde).transpose()?;
            let filas = metricas_sqlite::resumen(
                &config.ruta_metricas,
                &Filtro {
                    proyecto: project,
                    desde_ms,
                },
            )?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&filas).map_err(|e| e.to_string())?
                );
            } else {
                println!("{}", tabla_metricas(&filas));
            }
            Ok(0)
        }
        Comando::Config(c) => comando_config(c),
        Comando::Fake {
            host,
            port,
            tokens,
            token_delay_ms,
        } => {
            let s = falso::arrancar(
                &format!("{host}:{port}"),
                falso::OpcionesFalso {
                    tokens,
                    retardo_token_ms: token_delay_ms,
                    retardo_inicial_ms: 0,
                },
            )
            .await
            .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
            println!("Fake API listening at {} (Ctrl+C to stop)", s.url());
            let _ = tokio::signal::ctrl_c().await;
            Ok(0)
        }
        Comando::Bench {
            requests,
            concurrency,
            tokens,
            token_delay_ms,
            first_token_ms,
            no_metrics,
            json,
        } => {
            let informe = bench::ejecutar(bench::OpcionesBench {
                peticiones: requests,
                concurrencia: concurrency,
                tokens,
                retardo_token_ms: token_delay_ms,
                retardo_inicial_ms: first_token_ms,
                metricas: !no_metrics,
            })
            .await?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&informe).map_err(|e| e.to_string())?
                );
            } else {
                print!("{}", bench::mostrar(&informe));
            }
            let errores: usize = informe.results.iter().map(|r| r.errors).sum();
            Ok(if errores > 0 { 1 } else { 0 })
        }
    }
}

async fn servir(config: Config) -> Result<(), String> {
    // Las señales se registran antes de escuchar: una parada temprana no mata el proceso a medias.
    let parada = senal_de_parada();
    let direccion = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&direccion).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            format!("{direccion} is already in use (is Shiori already running? `shiori status`)")
        } else {
            format!("cannot listen on {direccion}: {e}")
        }
    })?;
    let clave = lanzar::nueva_clave(&config)?;
    let ruta_clave = lanzar::ruta_clave(&config);
    let url = config.url_local();
    let metricas = config.metricas.then(|| config.ruta_metricas.clone());
    let s = servidor::arrancar(config, listener, Some(clave)).await?;
    tracing::info!("Shiori {} listening at {url}", shiori::VERSION);
    match metricas {
        Some(r) => tracing::info!("metrics in {}", r.display()),
        None => tracing::info!("metrics off"),
    }
    let apagar = s.senal_apagado();
    tokio::spawn(async move {
        if parada.await {
            tracing::info!("stopping");
            apagar.notify_one();
        }
    });
    let r = s.esperar().await;
    let _ = std::fs::remove_file(ruta_clave);
    r.map_err(|e| e.to_string())
}

/// Lo que hace parar `serve` en orden: Ctrl+C y, en Unix, también SIGTERM (`kill`, systemd). En
/// Unix los manejadores quedan registrados al llamarla, no al esperar el futuro. Da `false` si no se
/// pudo escuchar ninguna señal.
#[cfg(unix)]
fn senal_de_parada() -> impl Future<Output = bool> + Send + 'static {
    use tokio::signal::unix::{SignalKind, signal};
    let int = signal(SignalKind::interrupt());
    let term = signal(SignalKind::terminate());
    async move {
        match (int, term) {
            (Ok(mut i), Ok(mut t)) => tokio::select! {
                Some(()) = i.recv() => true,
                Some(()) = t.recv() => true,
                else => false,
            },
            (Ok(mut i), Err(_)) => i.recv().await.is_some(),
            (Err(_), Ok(mut t)) => t.recv().await.is_some(),
            (Err(_), Err(_)) => false,
        }
    }
}

#[cfg(not(unix))]
async fn senal_de_parada() -> bool {
    tokio::signal::ctrl_c().await.is_ok()
}

fn imprimir_rutas(config: &Config) {
    match &config.archivo {
        Some(a) => println!("Global config: {}", a.display()),
        None => println!(
            "Global config: none (create {})",
            configuracion::carpeta_config()
                .map(|c| c.join("config.yaml").display().to_string())
                .unwrap_or_else(|| "?".into())
        ),
    }
    println!(
        "Metrics:       {}{}",
        config.ruta_metricas.display(),
        if config.metricas { "" } else { " (off)" }
    );
    println!("Log:           {}", lanzar::ruta_log().display());
}

fn comando_config(c: ComandoConfig) -> Result<i32, String> {
    match c {
        ComandoConfig::Validate { ruta } => {
            let mut revisados = Vec::new();
            let resultado: Result<(), configuracion::ErrorConfig> = (|| {
                match ruta {
                    Some(r) => {
                        if configuracion::es_de_proyecto(&r) {
                            configuracion::leer_proyecto(&r)?;
                        } else {
                            configuracion::leer_global(&r)?;
                        }
                        revisados.push(r);
                    }
                    None => {
                        if let Some(g) = configuracion::ruta_global(&entorno_real)? {
                            configuracion::leer_global(&g)?;
                            revisados.push(g);
                        }
                        let aqui = std::env::current_dir().unwrap_or_default();
                        if let Some(p) = configuracion::archivo_de_proyecto(&aqui)? {
                            configuracion::leer_proyecto(&p)?;
                            revisados.push(p);
                        }
                    }
                }
                Ok(())
            })();
            match resultado {
                Ok(()) if revisados.is_empty() => {
                    println!("No configuration files found: Shiori uses its defaults");
                    Ok(0)
                }
                Ok(()) => {
                    for r in revisados {
                        println!("{}: OK", r.display());
                    }
                    Ok(0)
                }
                Err(e) => {
                    eprintln!("{e}");
                    Ok(1)
                }
            }
        }
        ComandoConfig::Show => {
            let config = cargar_config()?;
            println!(
                "{}",
                serde_yaml_ng::to_string(&config).map_err(|e| e.to_string())?
            );
            Ok(0)
        }
        ComandoConfig::Path => {
            imprimir_rutas(&cargar_config()?);
            println!("Project file:  shiori.yaml (or shiori.toml) in the project folder");
            Ok(0)
        }
        ComandoConfig::Schema { project } => {
            println!("{}", configuracion::esquema(project));
            Ok(0)
        }
    }
}

/// `7d`, `24h`, `30m` o `AAAA-MM-DD` (UTC) → milisegundos desde 1970.
fn desde(texto: &str) -> Result<i64, String> {
    let t = texto.trim();
    let ahora = shiori::dominio::llamada::ahora_ms();
    for (sufijo, ms) in [("d", 86_400_000i64), ("h", 3_600_000), ("m", 60_000)] {
        if let Some(n) = t.strip_suffix(sufijo)
            && let Ok(n) = n.parse::<i64>()
        {
            return Ok(ahora - n * ms);
        }
    }
    let partes: Vec<&str> = t.split('-').collect();
    if let [a, m, d] = partes[..]
        && let (Ok(a), Ok(m), Ok(d)) = (a.parse::<i64>(), m.parse::<u32>(), d.parse::<u32>())
        && (1..=12).contains(&m)
        && (1..=31).contains(&d)
    {
        return Ok(dias_desde_1970(a, m, d) * 86_400_000);
    }
    Err(format!(
        "--since: `{texto}` is not 7d, 24h, 30m or a date like 2026-11-01"
    ))
}

/// Días desde 1970-01-01 (algoritmo de Howard Hinnant).
fn dias_desde_1970(a: i64, m: u32, d: u32) -> i64 {
    let a = if m <= 2 { a - 1 } else { a };
    let era = if a >= 0 { a } else { a - 399 } / 400;
    let ae = a - era * 400;
    let m = m as i64;
    let dia_ano = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
    let dia_era = ae * 365 + ae / 4 - ae / 100 + dia_ano;
    era * 146_097 + dia_era - 719_468
}

fn tabla_metricas(filas: &[metricas_sqlite::FilaResumen]) -> String {
    if filas.is_empty() {
        return "No calls recorded yet".into();
    }
    let dinero = |v: Option<f64>| {
        v.map(|v| format!("${v:.4}"))
            .unwrap_or_else(|| "n/a".into())
    };
    let mut s = format!(
        "{:<20} {:<9} {:<24} {:<12} {:>6} {:>5} {:>10} {:>9} {:>11} {:>10} {:>10} {:>10}\n",
        "project",
        "provider",
        "model",
        "auth",
        "calls",
        "errs",
        "input",
        "output",
        "cache read",
        "cache wr",
        "cost",
        "saved"
    );
    let mut coste = 0.0;
    let mut ahorro = 0.0;
    for f in filas {
        coste += f.cost_usd.unwrap_or(0.0);
        ahorro += f.cache_savings_usd.unwrap_or(0.0);
        s.push_str(&format!(
            "{:<20} {:<9} {:<24} {:<12} {:>6} {:>5} {:>10} {:>9} {:>11} {:>10} {:>10} {:>10}\n",
            recortar(&f.project, 20),
            f.provider,
            recortar(&f.model, 24),
            f.auth,
            f.calls,
            f.errors,
            f.input_tokens,
            f.output_tokens,
            f.cache_read_tokens,
            f.cache_write_tokens,
            dinero(f.cost_usd),
            dinero(f.cache_savings_usd),
        ));
    }
    s.push_str(&format!(
        "\nTotal: ${coste:.4}, saved by the provider cache: ${ahorro:.4}. With a subscription the cost is the API equivalent, not what you pay."
    ));
    s
}

fn recortar(t: &str, n: usize) -> String {
    if t.chars().count() <= n {
        t.to_string()
    } else {
        let mut r: String = t.chars().take(n - 1).collect();
        r.push('…');
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fechas() {
        assert_eq!(dias_desde_1970(1970, 1, 1), 0);
        assert_eq!(dias_desde_1970(2026, 11, 22), 20_779);
        assert_eq!(desde("2026-11-22").unwrap(), 20_779 * 86_400_000);
        assert!(desde("ayer").is_err());
        assert!(desde("7d").unwrap() < shiori::dominio::llamada::ahora_ms());
    }
}
