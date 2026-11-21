#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Duration;

use shiori::adaptadores::metricas_sqlite;
use shiori::configuracion::Config;
use shiori::dominio::llamada::Llamada;
use shiori::dominio::proveedor::Proveedor;
use shiori::falso::{self, OpcionesFalso, ServidorFalso};
use shiori::servidor::{self, ServidorEnMarcha};
use tokio::net::TcpListener;

pub struct Entorno {
    pub falso: ServidorFalso,
    pub proxy: ServidorEnMarcha,
    pub metricas: PathBuf,
    pub cliente: reqwest::Client,
    pub dir: tempfile::TempDir,
}

pub fn cliente() -> reqwest::Client {
    shiori::instalar_tls();
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

pub async fn entorno(opciones: OpcionesFalso) -> Entorno {
    entorno_con(opciones, |_| {}).await
}

pub async fn entorno_con(opciones: OpcionesFalso, ajustar: impl FnOnce(&mut Config)) -> Entorno {
    let falso = falso::arrancar("127.0.0.1:0", opciones).await.unwrap();
    let dir = tempfile::tempdir().unwrap();
    let metricas = dir.path().join("metrics.db");
    let mut config = Config {
        ruta_metricas: metricas.clone(),
        ..Config::default()
    };
    for p in Proveedor::TODOS {
        config.destinos.insert(p, falso.url());
    }
    ajustar(&mut config);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy = servidor::arrancar(config, listener, Some("clave-test".into()))
        .await
        .unwrap();
    Entorno {
        falso,
        proxy,
        metricas,
        cliente: cliente(),
        dir,
    }
}

impl Entorno {
    pub fn url(&self, ruta: &str) -> String {
        format!("{}{ruta}", self.proxy.url())
    }

    pub fn directo(&self, ruta: &str) -> String {
        format!("{}{ruta}", self.falso.url())
    }

    /// Espera a que haya al menos `n` llamadas registradas (se registran al terminar cada respuesta)
    /// y las devuelve de la más antigua a la más reciente.
    pub async fn llamadas(&self, n: usize) -> Vec<Llamada> {
        for _ in 0..200 {
            self.proxy.vaciar_metricas();
            if self.metricas.exists() {
                let mut v = metricas_sqlite::llamadas(&self.metricas, 10_000).unwrap();
                if v.len() >= n {
                    v.reverse();
                    return v;
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("no llegaron {n} llamadas a las métricas");
    }
}

pub fn cuerpo_anthropic(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "stream": stream,
        "messages": [{"role": "user", "content": "hola"}]
    })
}
