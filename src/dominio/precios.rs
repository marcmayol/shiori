//! Precios en dólares por millón de tokens y cálculo del coste y del ahorro por caché.
//!
//! La tabla integrada solo trae los modelos de Anthropic (precios públicos de la API a 3-nov-2026).
//! Los de OpenAI y Gemini, y cualquier cambio, se ponen en `prices:` de la configuración, que tiene
//! prioridad. Con suscripción (claude.ai, ChatGPT) el coste es el equivalente de la API, no lo que se paga.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::uso::Uso;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Precio {
    /// USD per million input tokens.
    pub input: f64,
    /// USD per million output tokens.
    pub output: f64,
    /// USD per million input tokens read from the provider's cache. Default: 10% of `input`.
    #[serde(default)]
    pub cache_read: Option<f64>,
    /// USD per million input tokens written to the provider's cache. Default: 125% of `input`.
    #[serde(default)]
    pub cache_write: Option<f64>,
}

impl Precio {
    const fn anthropic(input: f64, output: f64, cache_read: f64) -> Self {
        Precio {
            input,
            output,
            cache_read: Some(cache_read),
            cache_write: Some(input * 1.25),
        }
    }

    pub fn cache_read(&self) -> f64 {
        self.cache_read.unwrap_or(self.input * 0.1)
    }

    pub fn cache_write(&self) -> f64 {
        self.cache_write.unwrap_or(self.input * 1.25)
    }
}

/// Prefijo del id del modelo → precio. Se usa el prefijo más largo que encaje, así
/// `claude-opus-4-5-20251101` encuentra `claude-opus-4-5`.
const INTEGRADOS: &[(&str, Precio)] = &[
    ("claude-fable-5-2", Precio::anthropic(8.0, 40.0, 0.20)),
    ("claude-mythos-5-2", Precio::anthropic(8.0, 40.0, 0.20)),
    ("claude-fable-5-1", Precio::anthropic(10.0, 50.0, 1.0)),
    ("claude-mythos-5-1", Precio::anthropic(10.0, 50.0, 1.0)),
    ("claude-fable-5", Precio::anthropic(10.0, 50.0, 1.0)),
    ("claude-opus-5-5", Precio::anthropic(4.0, 20.0, 0.20)),
    ("claude-opus-5", Precio::anthropic(5.0, 25.0, 0.50)),
    ("claude-opus-4-8", Precio::anthropic(5.0, 25.0, 0.50)),
    ("claude-opus-4-7", Precio::anthropic(5.0, 25.0, 0.50)),
    ("claude-opus-4-6", Precio::anthropic(5.0, 25.0, 0.50)),
    ("claude-opus-4-5", Precio::anthropic(5.0, 25.0, 0.50)),
    ("claude-sonnet-5-5", Precio::anthropic(2.0, 10.0, 0.10)),
    ("claude-sonnet-5", Precio::anthropic(2.0, 10.0, 0.20)),
    ("claude-sonnet-4-6", Precio::anthropic(3.0, 15.0, 0.30)),
    ("claude-sonnet-4-5", Precio::anthropic(3.0, 15.0, 0.30)),
    ("claude-haiku-5-5", Precio::anthropic(0.10, 0.50, 0.01)),
    ("claude-haiku-4-5", Precio::anthropic(1.0, 5.0, 0.10)),
];

#[derive(Clone, Debug, Default)]
pub struct TablaPrecios {
    propios: BTreeMap<String, Precio>,
}

impl TablaPrecios {
    pub fn nueva(propios: BTreeMap<String, Precio>) -> Self {
        TablaPrecios { propios }
    }

    pub fn buscar(&self, modelo: &str) -> Option<Precio> {
        let propio = self
            .propios
            .iter()
            .filter(|(p, _)| modelo.starts_with(p.as_str()))
            .max_by_key(|(p, _)| p.len())
            .map(|(_, precio)| *precio);
        propio.or_else(|| {
            INTEGRADOS
                .iter()
                .filter(|(p, _)| modelo.starts_with(p))
                .max_by_key(|(p, _)| p.len())
                .map(|(_, precio)| *precio)
        })
    }

    /// (coste, ahorro por caché del proveedor) en dólares; `None` si el modelo no tiene precio.
    pub fn coste(&self, uso: &Uso) -> Option<(f64, f64)> {
        let precio = self.buscar(uso.modelo.as_deref()?)?;
        let m = 1_000_000.0;
        let coste = uso.entrada as f64 * precio.input / m
            + uso.salida as f64 * precio.output / m
            + uso.cache_leida as f64 * precio.cache_read() / m
            + uso.cache_escrita as f64 * precio.cache_write() / m;
        let ahorro = uso.cache_leida as f64 * (precio.input - precio.cache_read()) / m;
        Some((coste, ahorro))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefijo_mas_largo() {
        let t = TablaPrecios::default();
        assert_eq!(t.buscar("claude-opus-5-5").unwrap().input, 4.0);
        assert_eq!(t.buscar("claude-opus-5").unwrap().input, 5.0);
        assert_eq!(t.buscar("claude-opus-4-5-20251101").unwrap().input, 5.0);
        assert!(t.buscar("gpt-5").is_none());
    }

    #[test]
    fn propios_mandan() {
        let mut m = BTreeMap::new();
        m.insert(
            "claude-opus-5-5".to_string(),
            Precio {
                input: 1.0,
                output: 2.0,
                cache_read: None,
                cache_write: None,
            },
        );
        m.insert(
            "gpt-5".to_string(),
            Precio {
                input: 1.25,
                output: 10.0,
                cache_read: None,
                cache_write: None,
            },
        );
        let t = TablaPrecios::nueva(m);
        assert_eq!(t.buscar("claude-opus-5-5").unwrap().input, 1.0);
        assert_eq!(t.buscar("gpt-5-codex").unwrap().output, 10.0);
    }

    #[test]
    fn coste_y_ahorro() {
        let t = TablaPrecios::default();
        let uso = Uso {
            modelo: Some("claude-opus-5-5".into()),
            entrada: 1_000_000,
            salida: 1_000_000,
            cache_leida: 1_000_000,
            cache_escrita: 1_000_000,
        };
        let (coste, ahorro) = t.coste(&uso).unwrap();
        assert!((coste - (4.0 + 20.0 + 0.2 + 5.0)).abs() < 1e-9);
        assert!((ahorro - 3.8).abs() < 1e-9);
        assert!(t.coste(&Uso::default()).is_none());
    }
}
