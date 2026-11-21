use serde::{Deserialize, Serialize};

/// A qué API va una petición. `Chatgpt` es la API de la suscripción de ChatGPT (la que usa Codex al
/// iniciar sesión con la cuenta), distinta de la API de pago por uso de OpenAI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Proveedor {
    Anthropic,
    Openai,
    Chatgpt,
    Gemini,
}

impl Proveedor {
    pub const TODOS: [Proveedor; 4] = [
        Proveedor::Anthropic,
        Proveedor::Openai,
        Proveedor::Chatgpt,
        Proveedor::Gemini,
    ];

    pub fn nombre(self) -> &'static str {
        match self {
            Proveedor::Anthropic => "anthropic",
            Proveedor::Openai => "openai",
            Proveedor::Chatgpt => "chatgpt",
            Proveedor::Gemini => "gemini",
        }
    }

    pub fn desde_nombre(nombre: &str) -> Option<Self> {
        Self::TODOS.into_iter().find(|p| p.nombre() == nombre)
    }

    /// Dirección de la API real por defecto.
    pub fn destino_por_defecto(self) -> &'static str {
        match self {
            Proveedor::Anthropic => "https://api.anthropic.com",
            Proveedor::Openai => "https://api.openai.com",
            Proveedor::Chatgpt => "https://chatgpt.com/backend-api",
            Proveedor::Gemini => "https://generativelanguage.googleapis.com",
        }
    }
}

/// Decide el proveedor por el principio de la ruta y devuelve la ruta que se reenvía.
///
/// - `/openai/v1/chat/completions` → OpenAI, `/v1/chat/completions`
/// - `/chatgpt/codex/responses` → ChatGPT, `/codex/responses`
/// - `/gemini/v1beta/models/x:generateContent` → Gemini, `/v1beta/models/x:generateContent`
/// - `/anthropic/v1/messages` o sin prefijo (`/v1/messages`) → Anthropic. Sin prefijo para que baste
///   con `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`.
pub fn resolver_ruta(ruta: &str) -> (Proveedor, &str) {
    for p in Proveedor::TODOS {
        let prefijo = match p {
            Proveedor::Anthropic => "/anthropic",
            Proveedor::Openai => "/openai",
            Proveedor::Chatgpt => "/chatgpt",
            Proveedor::Gemini => "/gemini",
        };
        if let Some(resto) = ruta.strip_prefix(prefijo) {
            if resto.is_empty() {
                return (p, "/");
            }
            if resto.starts_with('/') {
                return (p, resto);
            }
        }
    }
    (Proveedor::Anthropic, ruta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rutas() {
        assert_eq!(
            resolver_ruta("/v1/messages"),
            (Proveedor::Anthropic, "/v1/messages")
        );
        assert_eq!(
            resolver_ruta("/anthropic/v1/messages"),
            (Proveedor::Anthropic, "/v1/messages")
        );
        assert_eq!(
            resolver_ruta("/openai/v1/responses"),
            (Proveedor::Openai, "/v1/responses")
        );
        assert_eq!(
            resolver_ruta("/chatgpt/codex/responses"),
            (Proveedor::Chatgpt, "/codex/responses")
        );
        assert_eq!(
            resolver_ruta("/gemini/v1beta/models/g:generateContent"),
            (Proveedor::Gemini, "/v1beta/models/g:generateContent")
        );
        assert_eq!(resolver_ruta("/openai"), (Proveedor::Openai, "/"));
        // Un prefijo que solo se parece no cuenta.
        assert_eq!(
            resolver_ruta("/openaiX/v1"),
            (Proveedor::Anthropic, "/openaiX/v1")
        );
    }
}
