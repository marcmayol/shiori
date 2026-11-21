use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

use super::proveedor::Proveedor;

/// Cómo paga quien llama. Shiori no toca las credenciales: solo mira de qué tipo son, para separar en
/// las métricas lo que va por API de pago por uso y lo que va por suscripción (claude.ai, ChatGPT), cuyo
/// coste en dólares es solo el equivalente de la API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Autenticacion {
    ApiKey,
    Subscription,
    Other,
    None,
}

impl Autenticacion {
    pub fn nombre(self) -> &'static str {
        match self {
            Autenticacion::ApiKey => "api_key",
            Autenticacion::Subscription => "subscription",
            Autenticacion::Other => "other",
            Autenticacion::None => "none",
        }
    }

    pub fn desde_nombre(nombre: &str) -> Self {
        match nombre {
            "api_key" => Autenticacion::ApiKey,
            "subscription" => Autenticacion::Subscription,
            "none" => Autenticacion::None,
            _ => Autenticacion::Other,
        }
    }
}

pub fn detectar(proveedor: Proveedor, cabeceras: &HeaderMap, query: Option<&str>) -> Autenticacion {
    let texto = |n: &str| cabeceras.get(n).and_then(|v| v.to_str().ok());
    let portador = texto("authorization")
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .map(str::trim);
    match proveedor {
        Proveedor::Anthropic => {
            if texto("x-api-key").is_some() {
                Autenticacion::ApiKey
            } else if let Some(t) = portador {
                // Los tokens OAuth de claude.ai (Claude Code con suscripción Pro/Max) empiezan así.
                if t.starts_with("sk-ant-oat") {
                    Autenticacion::Subscription
                } else if t.starts_with("sk-ant-api") {
                    Autenticacion::ApiKey
                } else {
                    Autenticacion::Other
                }
            } else {
                Autenticacion::None
            }
        }
        Proveedor::Openai | Proveedor::Chatgpt => match portador {
            Some(t) if t.starts_with("sk-") => Autenticacion::ApiKey,
            // Codex con la cuenta de ChatGPT manda un JWT y la cabecera chatgpt-account-id.
            Some(_) if proveedor == Proveedor::Chatgpt || texto("chatgpt-account-id").is_some() => {
                Autenticacion::Subscription
            }
            Some(_) => Autenticacion::Other,
            None => Autenticacion::None,
        },
        Proveedor::Gemini => {
            let clave_en_query = query
                .map(|q| q.split('&').any(|p| p.starts_with("key=")))
                .unwrap_or(false);
            if texto("x-goog-api-key").is_some() || clave_en_query {
                Autenticacion::ApiKey
            } else if portador.is_some() {
                Autenticacion::Other
            } else {
                Autenticacion::None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cab(pares: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pares {
            h.insert(*k, v.parse().unwrap());
        }
        h
    }

    #[test]
    fn anthropic() {
        let a = Proveedor::Anthropic;
        assert_eq!(
            detectar(a, &cab(&[("x-api-key", "sk-ant-api03-x")]), None),
            Autenticacion::ApiKey
        );
        assert_eq!(
            detectar(
                a,
                &cab(&[("authorization", "Bearer sk-ant-oat01-abc")]),
                None
            ),
            Autenticacion::Subscription
        );
        assert_eq!(
            detectar(a, &cab(&[("authorization", "Bearer otro")]), None),
            Autenticacion::Other
        );
        assert_eq!(detectar(a, &cab(&[]), None), Autenticacion::None);
    }

    #[test]
    fn openai_y_chatgpt() {
        assert_eq!(
            detectar(
                Proveedor::Openai,
                &cab(&[("authorization", "Bearer sk-proj-1")]),
                None
            ),
            Autenticacion::ApiKey
        );
        assert_eq!(
            detectar(
                Proveedor::Chatgpt,
                &cab(&[("authorization", "Bearer eyJhbGc")]),
                None
            ),
            Autenticacion::Subscription
        );
        assert_eq!(
            detectar(
                Proveedor::Openai,
                &cab(&[
                    ("authorization", "Bearer eyJhbGc"),
                    ("chatgpt-account-id", "x")
                ]),
                None
            ),
            Autenticacion::Subscription
        );
    }

    #[test]
    fn gemini() {
        assert_eq!(
            detectar(Proveedor::Gemini, &cab(&[]), Some("alt=sse&key=abc")),
            Autenticacion::ApiKey
        );
        assert_eq!(
            detectar(Proveedor::Gemini, &cab(&[("x-goog-api-key", "k")]), None),
            Autenticacion::ApiKey
        );
    }
}
