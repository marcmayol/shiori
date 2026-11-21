use serde::{Deserialize, Serialize};

use super::autenticacion::Autenticacion;
use super::proveedor::Proveedor;
use super::uso::Uso;

/// Una llamada que pasó por el proxy, tal como se guarda en las métricas.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Llamada {
    /// Milisegundos desde 1970 (UTC).
    pub momento_ms: i64,
    pub proyecto: String,
    pub proveedor: Proveedor,
    pub metodo: String,
    pub ruta: String,
    pub estado: u16,
    pub stream: bool,
    pub autenticacion: Autenticacion,
    pub uso: Uso,
    pub coste_usd: Option<f64>,
    pub ahorro_usd: Option<f64>,
    /// Desde que llega la petición hasta el último byte de la respuesta.
    pub latencia_ms: u64,
    /// Desde que llega la petición hasta el primer byte del cuerpo de la respuesta.
    pub primer_byte_ms: Option<u64>,
    pub bytes: u64,
    /// El cliente cortó antes de recibir la respuesta entera.
    pub cancelada: bool,
}

pub fn ahora_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
