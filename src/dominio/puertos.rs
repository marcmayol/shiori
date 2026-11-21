//! Puertos: lo que la aplicación necesita del exterior, sin saber cómo se hace.

use std::future::Future;
use std::pin::Pin;

use axum::http::{HeaderMap, Method, StatusCode};
use bytes::Bytes;
use futures_util::Stream;

use super::llamada::Llamada;

pub type FlujoBytes = Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;

pub struct PeticionDestino {
    pub metodo: Method,
    pub url: String,
    pub cabeceras: HeaderMap,
    pub cuerpo: Bytes,
}

pub struct RespuestaDestino {
    pub estado: StatusCode,
    pub cabeceras: HeaderMap,
    pub cuerpo: FlujoBytes,
}

/// La API real (o el servidor falso): manda una petición y devuelve la respuesta en streaming.
pub trait Destino: Send + Sync + 'static {
    fn enviar(
        &self,
        peticion: PeticionDestino,
    ) -> impl Future<Output = Result<RespuestaDestino, String>> + Send;
}

/// Dónde se guardan las llamadas. `registrar` no puede bloquear: se llama desde el camino de la respuesta.
pub trait AlmacenMetricas: Send + Sync + 'static {
    fn registrar(&self, llamada: Llamada);
    /// Espera a que lo registrado esté guardado (al parar y en los tests).
    fn vaciar(&self);
}

/// Métricas desactivadas.
pub struct SinMetricas;

impl AlmacenMetricas for SinMetricas {
    fn registrar(&self, _: Llamada) {}
    fn vaciar(&self) {}
}
