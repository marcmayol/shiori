use std::time::Duration;

use futures_util::TryStreamExt;

use crate::dominio::puertos::{Destino, PeticionDestino, RespuestaDestino};

/// Salida hacia las APIs reales con reqwest (rustls con las raíces del sistema).
#[derive(Clone)]
pub struct DestinoReqwest {
    cliente: reqwest::Client,
}

impl DestinoReqwest {
    pub fn nuevo() -> Result<Self, String> {
        crate::instalar_tls();
        let cliente = reqwest::Client::builder()
            // Sin tiempo máximo total: una respuesta en streaming puede durar muchos minutos.
            .connect_timeout(Duration::from_secs(15))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(30))
            // Un proxy transparente no sigue redirecciones: se las devuelve al cliente.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| format!("cannot create the HTTP client: {e}"))?;
        Ok(DestinoReqwest { cliente })
    }
}

impl Destino for DestinoReqwest {
    async fn enviar(&self, p: PeticionDestino) -> Result<RespuestaDestino, String> {
        let respuesta = self
            .cliente
            .request(p.metodo, &p.url)
            .headers(p.cabeceras)
            .body(p.cuerpo)
            .send()
            .await
            .map_err(|e| describir(&e))?;
        let estado = respuesta.status();
        let cabeceras = respuesta.headers().clone();
        let cuerpo = respuesta.bytes_stream().map_err(std::io::Error::other);
        Ok(RespuestaDestino {
            estado,
            cabeceras,
            cuerpo: Box::pin(cuerpo),
        })
    }
}

/// El error con sus causas, que es donde viene lo útil (DNS, conexión rechazada, certificado...).
fn describir(e: &reqwest::Error) -> String {
    let mut texto = e.to_string();
    let mut causa = std::error::Error::source(e);
    while let Some(c) = causa {
        texto.push_str(": ");
        texto.push_str(&c.to_string());
        causa = c.source();
    }
    texto
}
