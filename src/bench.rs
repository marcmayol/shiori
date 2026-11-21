//! Batería de rendimiento integrada (`shiori bench`): mide lo que añade Shiori frente a llamar
//! directo, contra el servidor falso, sin gastar tokens. Latencia p50/p99, tiempo hasta el primer
//! byte en streaming, peticiones por segundo y memoria.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::json;
use tokio::net::TcpListener;

use crate::configuracion::Config;
use crate::dominio::proveedor::Proveedor;
use crate::falso::{self, OpcionesFalso};
use crate::servidor;

#[derive(Clone, Debug)]
pub struct OpcionesBench {
    pub peticiones: usize,
    pub concurrencia: usize,
    pub tokens: u32,
    pub retardo_token_ms: u64,
    pub retardo_inicial_ms: u64,
    pub metricas: bool,
}

impl Default for OpcionesBench {
    fn default() -> Self {
        OpcionesBench {
            peticiones: 500,
            concurrencia: 16,
            tokens: 50,
            retardo_token_ms: 0,
            retardo_inicial_ms: 0,
            metricas: true,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Medida {
    pub scenario: &'static str,
    pub via: &'static str,
    pub requests: usize,
    pub errors: usize,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub first_byte_p50_ms: f64,
    pub first_byte_p99_ms: f64,
    pub requests_per_second: f64,
}

#[derive(Debug, Serialize)]
pub struct Sobrecoste {
    pub scenario: &'static str,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub first_byte_p50_ms: f64,
    pub first_byte_p99_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct Informe {
    pub requests: usize,
    pub concurrency: usize,
    pub tokens: u32,
    pub token_delay_ms: u64,
    pub metrics: bool,
    pub results: Vec<Medida>,
    /// Lo que añade Shiori (vía Shiori menos directo).
    pub overhead: Vec<Sobrecoste>,
    /// Memoria física del proceso del bench (cliente, servidor falso y proxy juntos), antes y después.
    pub rss_before_mb: Option<f64>,
    pub rss_after_mb: Option<f64>,
}

fn rss_mb() -> Option<f64> {
    memory_stats::memory_stats().map(|m| m.physical_mem as f64 / (1024.0 * 1024.0))
}

fn percentil(ordenados: &[f64], p: f64) -> f64 {
    if ordenados.is_empty() {
        return 0.0;
    }
    let i = ((p / 100.0) * (ordenados.len() - 1) as f64).round() as usize;
    ordenados[i.min(ordenados.len() - 1)]
}

async fn una(cliente: &reqwest::Client, url: &str, stream: bool) -> Result<(f64, f64), String> {
    let cuerpo = json!({
        "model": "claude-sonnet-5-5",
        "max_tokens": 1024,
        "stream": stream,
        "messages": [{"role": "user", "content": "bench"}]
    });
    let inicio = Instant::now();
    let r = cliente
        .post(url)
        .header("x-api-key", "sk-ant-api-bench")
        .header("anthropic-version", "2023-06-01")
        .json(&cuerpo)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("status {}", r.status()));
    }
    let mut flujo = r.bytes_stream();
    let mut primer_byte = None;
    while let Some(trozo) = flujo.next().await {
        trozo.map_err(|e| e.to_string())?;
        if primer_byte.is_none() {
            primer_byte = Some(inicio.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let total = inicio.elapsed().as_secs_f64() * 1000.0;
    Ok((total, primer_byte.unwrap_or(total)))
}

async fn medir(
    cliente: &reqwest::Client,
    url: &str,
    stream: bool,
    op: &OpcionesBench,
    scenario: &'static str,
    via: &'static str,
) -> Medida {
    // Calentar conexiones.
    for _ in 0..op.concurrencia.min(20) {
        let _ = una(cliente, url, stream).await;
    }
    let inicio = Instant::now();
    let resultados: Vec<Result<(f64, f64), String>> = futures_util::stream::iter(0..op.peticiones)
        .map(|_| una(cliente, url, stream))
        .buffer_unordered(op.concurrencia.max(1))
        .collect()
        .await;
    let segundos = inicio.elapsed().as_secs_f64();
    let errores = resultados.iter().filter(|r| r.is_err()).count();
    let (mut lat, mut pb): (Vec<f64>, Vec<f64>) = resultados.into_iter().flatten().unzip();
    lat.sort_by(f64::total_cmp);
    pb.sort_by(f64::total_cmp);
    Medida {
        scenario,
        via,
        requests: op.peticiones,
        errors: errores,
        latency_p50_ms: percentil(&lat, 50.0),
        latency_p99_ms: percentil(&lat, 99.0),
        first_byte_p50_ms: percentil(&pb, 50.0),
        first_byte_p99_ms: percentil(&pb, 99.0),
        requests_per_second: if segundos > 0.0 {
            (op.peticiones - errores) as f64 / segundos
        } else {
            0.0
        },
    }
}

pub async fn ejecutar(op: OpcionesBench) -> Result<Informe, String> {
    let rss_before_mb = rss_mb();
    let falso = falso::arrancar(
        "127.0.0.1:0",
        OpcionesFalso {
            tokens: op.tokens,
            retardo_token_ms: op.retardo_token_ms,
            retardo_inicial_ms: op.retardo_inicial_ms,
        },
    )
    .await
    .map_err(|e| format!("cannot start the fake server: {e}"))?;

    let ruta_metricas =
        std::env::temp_dir().join(format!("shiori-bench-{}.db", std::process::id()));
    let mut config = Config {
        metricas: op.metricas,
        ruta_metricas: ruta_metricas.clone(),
        ..Config::default()
    };
    for p in Proveedor::TODOS {
        config.destinos.insert(p, falso.url());
    }
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| e.to_string())?;
    let proxy = servidor::arrancar(config, listener, None).await?;

    crate::instalar_tls();
    let cliente = reqwest::Client::builder()
        .no_proxy()
        .pool_max_idle_per_host(op.concurrencia.max(1) * 2)
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let directo = format!("{}/v1/messages", falso.url());
    let via_shiori = format!("{}/v1/messages", proxy.url());
    let mut results = Vec::new();
    let mut overhead = Vec::new();
    for (scenario, stream) in [("json", false), ("stream", true)] {
        let d = medir(&cliente, &directo, stream, &op, scenario, "direct").await;
        let s = medir(&cliente, &via_shiori, stream, &op, scenario, "shiori").await;
        overhead.push(Sobrecoste {
            scenario,
            latency_p50_ms: s.latency_p50_ms - d.latency_p50_ms,
            latency_p99_ms: s.latency_p99_ms - d.latency_p99_ms,
            first_byte_p50_ms: s.first_byte_p50_ms - d.first_byte_p50_ms,
            first_byte_p99_ms: s.first_byte_p99_ms - d.first_byte_p99_ms,
        });
        results.push(d);
        results.push(s);
    }
    let rss_after_mb = rss_mb();
    proxy.parar().await;
    for sufijo in ["", "-wal", "-shm"] {
        let mut r = ruta_metricas.clone().into_os_string();
        r.push(sufijo);
        let _ = std::fs::remove_file(r);
    }
    Ok(Informe {
        requests: op.peticiones,
        concurrency: op.concurrencia,
        tokens: op.tokens,
        token_delay_ms: op.retardo_token_ms,
        metrics: op.metricas,
        results,
        overhead,
        rss_before_mb,
        rss_after_mb,
    })
}

pub fn mostrar(i: &Informe) -> String {
    let mut s = format!(
        "Shiori bench: {} requests, concurrency {}, {} tokens per answer, {} ms between tokens, metrics {}\n\n",
        i.requests,
        i.concurrency,
        i.tokens,
        i.token_delay_ms,
        if i.metrics { "on" } else { "off" }
    );
    s.push_str(&format!(
        "{:<8} {:<7} {:>10} {:>10} {:>12} {:>12} {:>9} {:>7}\n",
        "scenario", "via", "p50 ms", "p99 ms", "1st byte p50", "1st byte p99", "req/s", "errors"
    ));
    for m in &i.results {
        s.push_str(&format!(
            "{:<8} {:<7} {:>10.2} {:>10.2} {:>12.2} {:>12.2} {:>9.0} {:>7}\n",
            m.scenario,
            m.via,
            m.latency_p50_ms,
            m.latency_p99_ms,
            m.first_byte_p50_ms,
            m.first_byte_p99_ms,
            m.requests_per_second,
            m.errors
        ));
    }
    s.push_str("\nAdded by Shiori:\n");
    for o in &i.overhead {
        s.push_str(&format!(
            "  {:<7} p50 {:+.2} ms, p99 {:+.2} ms, first byte p50 {:+.2} ms, p99 {:+.2} ms\n",
            o.scenario,
            o.latency_p50_ms,
            o.latency_p99_ms,
            o.first_byte_p50_ms,
            o.first_byte_p99_ms
        ));
    }
    if let (Some(a), Some(b)) = (i.rss_before_mb, i.rss_after_mb) {
        s.push_str(&format!(
            "\nMemory of the bench process (client, fake server and proxy): {a:.1} MB before, {b:.1} MB after\n"
        ));
    }
    s
}
