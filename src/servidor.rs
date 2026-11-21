//! Montar y parar el servidor: configuración + adaptadores + proxy.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::serve::ListenerExt;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::adaptadores::entrada_axum::{EstadoHttp, router};
use crate::adaptadores::metricas_sqlite::MetricasSqlite;
use crate::adaptadores::salida_reqwest::DestinoReqwest;
use crate::aplicacion::Proxy;
use crate::configuracion::Config;
use crate::dominio::puertos::{AlmacenMetricas, Destino, SinMetricas};

pub struct ServidorEnMarcha {
    pub direccion: SocketAddr,
    metricas: Arc<dyn AlmacenMetricas>,
    apagar: Arc<Notify>,
    tarea: JoinHandle<std::io::Result<()>>,
}

impl ServidorEnMarcha {
    pub fn url(&self) -> String {
        format!("http://{}", self.direccion)
    }

    /// Espera a que lo registrado en las métricas esté en disco.
    pub fn vaciar_metricas(&self) {
        self.metricas.vaciar();
    }

    /// Para de aceptar peticiones, deja terminar las que están en curso y guarda las métricas.
    pub async fn parar(self) {
        self.apagar.notify_one();
        let _ = self.tarea.await;
        let m = self.metricas.clone();
        let _ = tokio::task::spawn_blocking(move || m.vaciar()).await;
    }

    /// Espera a que el servidor termine (por `/_shiori/stop` o por `parar`).
    pub async fn esperar(self) -> std::io::Result<()> {
        let r = self
            .tarea
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(e)));
        let m = self.metricas.clone();
        let _ = tokio::task::spawn_blocking(move || m.vaciar()).await;
        r
    }

    pub fn senal_apagado(&self) -> Arc<Notify> {
        self.apagar.clone()
    }
}

/// Arranca con la salida real a las APIs.
pub async fn arrancar(
    config: Config,
    listener: TcpListener,
    clave_parada: Option<String>,
) -> Result<ServidorEnMarcha, String> {
    let destino = DestinoReqwest::nuevo()?;
    arrancar_con(config, destino, listener, clave_parada).await
}

pub async fn arrancar_con<D: Destino>(
    config: Config,
    destino: D,
    listener: TcpListener,
    clave_parada: Option<String>,
) -> Result<ServidorEnMarcha, String> {
    let metricas: Arc<dyn AlmacenMetricas> = if config.metricas {
        Arc::new(MetricasSqlite::nuevo(&config.ruta_metricas)?)
    } else {
        Arc::new(SinMetricas)
    };
    let direccion = listener.local_addr().map_err(|e| e.to_string())?;
    let apagar = Arc::new(Notify::new());
    let estado = Arc::new(EstadoHttp {
        proxy: Proxy::nuevo(Arc::new(config), destino, metricas.clone()),
        clave_parada,
        apagar: apagar.clone(),
    });
    let app = router(estado);
    let senal = apagar.clone();
    let tarea = tokio::spawn(async move {
        axum::serve(
            // Sin Nagle: cada evento SSE sale en cuanto llega, sin esperar al ACK del anterior.
            listener.tap_io(|tcp| {
                let _ = tcp.set_nodelay(true);
            }),
            app,
        )
        .with_graceful_shutdown(async move { senal.notified().await })
        .await
    });
    Ok(ServidorEnMarcha {
        direccion,
        metricas,
        apagar,
        tarea,
    })
}
