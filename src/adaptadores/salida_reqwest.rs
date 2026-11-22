use std::time::Duration;

use futures_util::TryStreamExt;

use crate::dominio::puertos::{Destino, PeticionDestino, RespuestaDestino};

/// Salida hacia las APIs reales con reqwest (rustls con las raíces del sistema).
///
/// Hacia fuera respeta el proxy del sistema (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY`),
/// que es lo que hace falta detrás de un proxy de empresa. Hacia un destino en este mismo equipo
/// (`127.0.0.1`, `::1`, `localhost`) va siempre directo: un proxy ahí no pinta nada, y por HTTP
/// plano vería las claves de la API.
#[derive(Clone)]
pub struct DestinoReqwest {
    cliente: reqwest::Client,
    cliente_local: reqwest::Client,
}

fn base() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        // Sin tiempo máximo total: una respuesta en streaming puede durar muchos minutos.
        .connect_timeout(Duration::from_secs(15))
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(30))
        // Un proxy transparente no sigue redirecciones: se las devuelve al cliente.
        .redirect(reqwest::redirect::Policy::none())
}

impl DestinoReqwest {
    pub fn nuevo() -> Result<Self, String> {
        crate::instalar_tls();
        let error = |e: reqwest::Error| format!("cannot create the HTTP client: {e}");
        Ok(DestinoReqwest {
            cliente: base().build().map_err(error)?,
            cliente_local: base().no_proxy().build().map_err(error)?,
        })
    }

    fn cliente_para(&self, url: &str) -> &reqwest::Client {
        if es_local(url) {
            &self.cliente_local
        } else {
            &self.cliente
        }
    }
}

/// ¿La URL apunta a este equipo?
fn es_local(url: &str) -> bool {
    let Ok(u) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = u.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => {
            let host = host.to_ascii_lowercase();
            host == "localhost" || host.ends_with(".localhost")
        }
    }
}

impl Destino for DestinoReqwest {
    async fn enviar(&self, p: PeticionDestino) -> Result<RespuestaDestino, String> {
        let respuesta = self
            .cliente_para(&p.url)
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

#[cfg(test)]
mod tests {
    use super::es_local;

    #[test]
    fn destinos_locales() {
        for u in [
            "http://127.0.0.1:8788",
            "http://127.0.0.5/v1",
            "http://[::1]:9000",
            "http://localhost:1",
            "http://LocalHost",
            "http://api.localhost",
        ] {
            assert!(es_local(u), "{u}");
        }
        for u in [
            "https://api.anthropic.com",
            "http://192.168.1.10:8080",
            "http://localhost.example.com",
            "no es una url",
        ] {
            assert!(!es_local(u), "{u}");
        }
    }
}
