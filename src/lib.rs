//! Shiori: proxy local delante de las APIs de LLM (Anthropic, OpenAI, Gemini).
//!
//! Arquitectura hexagonal:
//! - `dominio`: tipos y reglas sin dependencias de red ni de disco (proveedores, uso, precios, llamadas).
//! - `aplicacion`: el caso de uso del proxy, que solo habla con puertos (`dominio::puertos`).
//! - `adaptadores`: la entrada HTTP (axum), la salida a las APIs (reqwest) y el almacén de métricas (SQLite).
//! - `configuracion`: archivo global, archivo de proyecto, variables de entorno y su validación.

pub mod adaptadores;
pub mod aplicacion;
pub mod bench;
pub mod configuracion;
pub mod dominio;
pub mod falso;
pub mod lanzar;
pub mod servidor;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Instala ring como proveedor criptográfico de rustls. Hay que llamarlo antes de crear un cliente HTTPS;
/// llamarlo más de una vez no hace nada.
pub fn instalar_tls() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
