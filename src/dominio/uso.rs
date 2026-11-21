//! Lectura del uso de tokens de las respuestas, sin cambiarlas: JSON entero o eventos SSE al vuelo.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::proveedor::Proveedor;

/// Tokens de una llamada, normalizados para los tres proveedores:
/// `entrada` son los de entrada que se cobran a precio completo (sin los servidos desde caché).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Uso {
    pub modelo: Option<String>,
    pub entrada: u64,
    pub salida: u64,
    pub cache_leida: u64,
    pub cache_escrita: u64,
}

impl Uso {
    pub fn vacio(&self) -> bool {
        self.entrada == 0 && self.salida == 0 && self.cache_leida == 0 && self.cache_escrita == 0
    }
}

fn num(v: &Value, clave: &str) -> Option<u64> {
    v.get(clave).and_then(Value::as_u64)
}

fn modelo(v: &Value, clave: &str, uso: &mut Uso) {
    if let Some(m) = v.get(clave).and_then(Value::as_str)
        && !m.is_empty()
    {
        uso.modelo = Some(m.to_string());
    }
}

/// Actualiza `uso` con lo que haya en un objeto JSON de respuesta o de evento. Cada campo presente
/// sustituye al anterior: en el streaming de Anthropic, `message_delta` trae el total final de salida.
pub fn actualizar(proveedor: Proveedor, v: &Value, uso: &mut Uso) {
    if let Value::Array(elementos) = v {
        // Gemini sin `alt=sse` devuelve el streaming como un array JSON.
        for e in elementos {
            actualizar(proveedor, e, uso);
        }
        return;
    }
    match proveedor {
        Proveedor::Anthropic => {
            for c in [Some(v), v.get("message")].into_iter().flatten() {
                modelo(c, "model", uso);
                if let Some(u) = c.get("usage").filter(|u| u.is_object()) {
                    if let Some(n) = num(u, "input_tokens") {
                        uso.entrada = n;
                    }
                    if let Some(n) = num(u, "output_tokens") {
                        uso.salida = n;
                    }
                    if let Some(n) = num(u, "cache_read_input_tokens") {
                        uso.cache_leida = n;
                    }
                    if let Some(n) = num(u, "cache_creation_input_tokens") {
                        uso.cache_escrita = n;
                    }
                }
            }
        }
        Proveedor::Openai | Proveedor::Chatgpt => {
            for c in [Some(v), v.get("response")].into_iter().flatten() {
                modelo(c, "model", uso);
                let Some(u) = c.get("usage").filter(|u| u.is_object()) else {
                    continue;
                };
                // Chat Completions: prompt_tokens incluye los servidos desde caché.
                if let Some(p) = num(u, "prompt_tokens") {
                    let cache = u
                        .get("prompt_tokens_details")
                        .and_then(|d| num(d, "cached_tokens"))
                        .unwrap_or(0);
                    uso.cache_leida = cache;
                    uso.entrada = p.saturating_sub(cache);
                    uso.salida = num(u, "completion_tokens").unwrap_or(uso.salida);
                }
                // Responses API (también la de Codex con ChatGPT).
                if let Some(p) = num(u, "input_tokens") {
                    let cache = u
                        .get("input_tokens_details")
                        .and_then(|d| num(d, "cached_tokens"))
                        .unwrap_or(0);
                    uso.cache_leida = cache;
                    uso.entrada = p.saturating_sub(cache);
                    uso.salida = num(u, "output_tokens").unwrap_or(uso.salida);
                }
            }
        }
        Proveedor::Gemini => {
            modelo(v, "modelVersion", uso);
            if let Some(u) = v.get("usageMetadata").filter(|u| u.is_object()) {
                let cache = num(u, "cachedContentTokenCount").unwrap_or(0);
                if let Some(p) = num(u, "promptTokenCount") {
                    uso.entrada = p.saturating_sub(cache);
                    uso.cache_leida = cache;
                }
                let salida = num(u, "candidatesTokenCount").unwrap_or(0)
                    + num(u, "thoughtsTokenCount").unwrap_or(0);
                if salida > 0 {
                    uso.salida = salida;
                }
            }
        }
    }
}

/// Lector incremental de SSE: recibe trozos de bytes en cualquier corte y entrega el `data` de cada
/// evento completo.
#[derive(Default)]
pub struct LectorSse {
    linea: Vec<u8>,
    datos: String,
    linea_descartada: bool,
}

/// Una línea más larga se descarta: el uso nunca va en eventos tan grandes y así la memoria no crece.
const MAX_LINEA: usize = 4 * 1024 * 1024;

impl LectorSse {
    pub fn alimentar(&mut self, trozo: &[u8], mut evento: impl FnMut(&str)) {
        for &b in trozo {
            if b == b'\n' {
                self.fin_de_linea(&mut evento);
            } else if self.linea.len() < MAX_LINEA {
                self.linea.push(b);
            } else {
                self.linea_descartada = true;
            }
        }
    }

    pub fn terminar(&mut self, mut evento: impl FnMut(&str)) {
        if !self.linea.is_empty() {
            self.fin_de_linea(&mut evento);
        }
        if !self.datos.is_empty() {
            evento(&self.datos);
            self.datos.clear();
        }
    }

    fn fin_de_linea(&mut self, evento: &mut impl FnMut(&str)) {
        if self.linea.last() == Some(&b'\r') {
            self.linea.pop();
        }
        if self.linea_descartada {
            self.linea_descartada = false;
            self.linea.clear();
            return;
        }
        if self.linea.is_empty() {
            if !self.datos.is_empty() {
                evento(&self.datos);
                self.datos.clear();
            }
            return;
        }
        if let Some(resto) = self.linea.strip_prefix(b"data:") {
            let resto = resto.strip_prefix(b" ").unwrap_or(resto);
            if !self.datos.is_empty() {
                self.datos.push('\n');
            }
            self.datos.push_str(&String::from_utf8_lossy(resto));
        }
        self.linea.clear();
    }
}

/// Cuerpo JSON que se guarda para leer el uso al final; por encima de este tamaño se deja de guardar.
const MAX_JSON: usize = 64 * 1024 * 1024;

/// Mide el uso de una respuesta mientras pasa, trozo a trozo.
pub struct Medidor {
    proveedor: Proveedor,
    modo: Modo,
    pub uso: Uso,
}

enum Modo {
    Sse(LectorSse),
    Json(Vec<u8>, bool),
}

impl Medidor {
    pub fn nuevo(proveedor: Proveedor, es_sse: bool) -> Self {
        let modo = if es_sse {
            Modo::Sse(LectorSse::default())
        } else {
            Modo::Json(Vec::new(), false)
        };
        Medidor {
            proveedor,
            modo,
            uso: Uso::default(),
        }
    }

    pub fn alimentar(&mut self, trozo: &[u8]) {
        let proveedor = self.proveedor;
        match &mut self.modo {
            Modo::Sse(lector) => {
                let uso = &mut self.uso;
                lector.alimentar(trozo, |datos| procesar_evento(proveedor, datos, uso));
            }
            Modo::Json(buffer, desbordado) => {
                if buffer.len() + trozo.len() <= MAX_JSON {
                    buffer.extend_from_slice(trozo);
                } else {
                    *desbordado = true;
                }
            }
        }
    }

    pub fn terminar(mut self) -> Uso {
        let proveedor = self.proveedor;
        match &mut self.modo {
            Modo::Sse(lector) => {
                let uso = &mut self.uso;
                lector.terminar(|datos| procesar_evento(proveedor, datos, uso));
            }
            Modo::Json(buffer, desbordado) => {
                if !*desbordado && let Ok(v) = serde_json::from_slice::<Value>(buffer) {
                    actualizar(proveedor, &v, &mut self.uso);
                }
            }
        }
        self.uso
    }
}

fn procesar_evento(proveedor: Proveedor, datos: &str, uso: &mut Uso) {
    // Solo se parsean los eventos que pueden traer uso o modelo: los deltas de texto son la mayoría.
    if !(datos.contains("usage") || datos.contains("Usage") || datos.contains("model")) {
        return;
    }
    if let Ok(v) = serde_json::from_str::<Value>(datos) {
        actualizar(proveedor, &v, uso);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_json() {
        let mut u = Uso::default();
        actualizar(
            Proveedor::Anthropic,
            &json!({"model":"claude-opus-5-5","usage":{"input_tokens":10,"output_tokens":20,
                "cache_read_input_tokens":30,"cache_creation_input_tokens":40}}),
            &mut u,
        );
        assert_eq!(
            u,
            Uso {
                modelo: Some("claude-opus-5-5".into()),
                entrada: 10,
                salida: 20,
                cache_leida: 30,
                cache_escrita: 40
            }
        );
    }

    #[test]
    fn anthropic_sse_troceado_en_cualquier_punto() {
        let sse = concat!(
            "event: message_start\r\n",
            "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-5-5\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1,\"cache_read_input_tokens\":100,\"cache_creation_input_tokens\":7}}}\r\n\r\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hola usage model\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
        );
        for corte in 1..sse.len() {
            let mut m = Medidor::nuevo(Proveedor::Anthropic, true);
            m.alimentar(&sse.as_bytes()[..corte]);
            m.alimentar(&sse.as_bytes()[corte..]);
            let u = m.terminar();
            assert_eq!(u.modelo.as_deref(), Some("claude-sonnet-5-5"));
            assert_eq!(
                (u.entrada, u.salida, u.cache_leida, u.cache_escrita),
                (5, 42, 100, 7),
                "corte {corte}"
            );
        }
    }

    #[test]
    fn openai_chat_y_responses() {
        let mut u = Uso::default();
        actualizar(
            Proveedor::Openai,
            &json!({"model":"gpt-x","usage":{"prompt_tokens":100,"completion_tokens":9,
                "prompt_tokens_details":{"cached_tokens":60}}}),
            &mut u,
        );
        assert_eq!((u.entrada, u.salida, u.cache_leida), (40, 9, 60));

        let mut u = Uso::default();
        actualizar(
            Proveedor::Chatgpt,
            &json!({"type":"response.completed","response":{"model":"gpt-5-codex","usage":{
                "input_tokens":50,"output_tokens":5,"input_tokens_details":{"cached_tokens":20}}}}),
            &mut u,
        );
        assert_eq!(u.modelo.as_deref(), Some("gpt-5-codex"));
        assert_eq!((u.entrada, u.salida, u.cache_leida), (30, 5, 20));
    }

    #[test]
    fn openai_chunk_con_usage_null_no_borra() {
        let mut u = Uso {
            salida: 3,
            ..Default::default()
        };
        actualizar(Proveedor::Openai, &json!({"usage": null}), &mut u);
        assert_eq!(u.salida, 3);
    }

    #[test]
    fn gemini_array_y_pensamiento() {
        let mut u = Uso::default();
        actualizar(
            Proveedor::Gemini,
            &json!([{"modelVersion":"gemini-x","usageMetadata":{"promptTokenCount":10}},
                {"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":4,
                "candidatesTokenCount":6,"thoughtsTokenCount":2}}]),
            &mut u,
        );
        assert_eq!(u.modelo.as_deref(), Some("gemini-x"));
        assert_eq!((u.entrada, u.salida, u.cache_leida), (6, 8, 4));
    }

    #[test]
    fn json_invalido_no_rompe() {
        let mut m = Medidor::nuevo(Proveedor::Anthropic, false);
        m.alimentar(b"{no es json");
        assert!(m.terminar().vacio());
    }

    #[test]
    fn linea_enorme_se_descarta_y_sigue() {
        let mut l = LectorSse::default();
        let mut eventos = Vec::new();
        let grande = vec![b'x'; MAX_LINEA + 10];
        l.alimentar(b"data: ", |_| {});
        l.alimentar(&grande, |_| {});
        l.alimentar(b"\n\ndata: ok\n\n", |d| eventos.push(d.to_string()));
        assert_eq!(eventos, vec!["ok"]);
    }
}
