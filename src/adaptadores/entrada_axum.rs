//! Entrada HTTP con axum: todo lo que no es `/_shiori/health` ni `/_shiori/stop` va al proxy.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;
use tokio::sync::Notify;

use crate::aplicacion::{Entrada, Proxy, Salida, error};
use crate::dominio::proveedor::resolver_ruta;
use crate::dominio::puertos::Destino;

pub const CABECERA_CLAVE_PARADA: &str = "x-shiori-stop-key";

pub struct EstadoHttp<D: Destino> {
    pub proxy: Proxy<D>,
    /// Clave que pide `/_shiori/stop`; sin clave, no se puede parar por HTTP.
    pub clave_parada: Option<String>,
    pub apagar: Arc<Notify>,
}

pub fn router<D: Destino>(estado: Arc<EstadoHttp<D>>) -> Router {
    Router::new()
        .route("/_shiori/health", get(salud::<D>))
        .route("/_shiori/stop", post(parar::<D>))
        .fallback(atender::<D>)
        .with_state(estado)
}

async fn salud<D: Destino>(State(e): State<Arc<EstadoHttp<D>>>) -> Response {
    let c = e.proxy.config();
    axum::Json(json!({
        "ok": true,
        "name": "shiori",
        "version": crate::VERSION,
        "pid": std::process::id(),
        "metrics": c.metricas,
    }))
    .into_response()
}

async fn parar<D: Destino>(State(e): State<Arc<EstadoHttp<D>>>, cabeceras: HeaderMap) -> Response {
    let dada = cabeceras
        .get(CABECERA_CLAVE_PARADA)
        .and_then(|v| v.to_str().ok());
    match (&e.clave_parada, dada) {
        (Some(clave), Some(d)) if iguales(clave.as_bytes(), d.as_bytes()) => {
            e.apagar.notify_one();
            (StatusCode::OK, axum::Json(json!({"stopping": true}))).into_response()
        }
        _ => (
            StatusCode::FORBIDDEN,
            axum::Json(json!({"error": "wrong stop key"})),
        )
            .into_response(),
    }
}

/// Comparación en tiempo constante.
fn iguales(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn atender<D: Destino>(State(e): State<Arc<EstadoHttp<D>>>, peticion: Request) -> Response {
    let (partes, cuerpo) = peticion.into_parts();
    let ruta = partes
        .uri
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "/".into());
    let limite = e.proxy.config().max_cuerpo_bytes;
    let cuerpo = match axum::body::to_bytes(cuerpo, limite).await {
        Ok(b) => b,
        Err(err) => {
            let (proveedor, _) = resolver_ruta(&ruta);
            let mb = limite / (1024 * 1024);
            return respuesta(error(
                proveedor,
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!(
                    "shiori: cannot read the request body (limit {mb} MB, server.max_body_mb): {err}"
                ),
            ));
        }
    };
    let salida = e
        .proxy
        .atender(Entrada {
            metodo: partes.method,
            ruta,
            cabeceras: partes.headers,
            cuerpo,
        })
        .await;
    respuesta(salida)
}

fn respuesta(s: Salida) -> Response {
    let mut r = Response::new(Body::from_stream(s.cuerpo));
    *r.status_mut() = s.estado;
    *r.headers_mut() = s.cabeceras;
    r
}
