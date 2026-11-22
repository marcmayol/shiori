//! Servidor falso que imita las APIs de Anthropic, OpenAI (Chat Completions y Responses, también la
//! de ChatGPT/Codex) y Gemini, en JSON y en streaming, para probar y medir sin gastar tokens.
//!
//! Cabeceras de control (no las mandan los clientes reales):
//! - `x-fake-status: 429` responde ese error con el formato de la API.
//! - `x-fake-tokens: 50` tokens de salida (por defecto, los de las opciones).
//! - `x-fake-delay-ms: 5` espera entre tokens en streaming.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::Response;
use axum::serve::ListenerExt;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Uso fijo que devuelve el servidor falso (la salida depende de los tokens pedidos).
pub const ENTRADA: u64 = 120;
pub const CACHE_LEIDA: u64 = 300;
pub const CACHE_ESCRITA: u64 = 40;

/// Cuántas peticiones recuerda (las últimas). Sin tope, un `shiori fake` de larga vida bajo carga
/// se quedaba sin memoria: con cuerpos de 55 KB crecía unos 11 MB/s a 200 peticiones/s.
pub const MAX_VISTAS: usize = 256;

#[derive(Clone, Debug)]
pub struct OpcionesFalso {
    pub tokens: u32,
    /// Espera entre tokens en streaming.
    pub retardo_token_ms: u64,
    /// Espera antes de empezar a responder (simula el tiempo de la API hasta el primer token).
    pub retardo_inicial_ms: u64,
}

impl Default for OpcionesFalso {
    fn default() -> Self {
        OpcionesFalso {
            tokens: 20,
            retardo_token_ms: 0,
            retardo_inicial_ms: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PeticionVista {
    pub metodo: String,
    pub ruta: String,
    pub cabeceras: HeaderMap,
    pub cuerpo: Bytes,
}

struct Estado {
    opciones: OpcionesFalso,
    vistas: Mutex<VecDeque<PeticionVista>>,
}

pub struct ServidorFalso {
    pub direccion: SocketAddr,
    estado: Arc<Estado>,
    apagar: Option<oneshot::Sender<()>>,
}

impl ServidorFalso {
    pub fn url(&self) -> String {
        format!("http://{}", self.direccion)
    }

    pub fn ultima(&self) -> Option<PeticionVista> {
        self.estado.vistas.lock().unwrap().back().cloned()
    }

    /// Las últimas peticiones recibidas (como mucho [`MAX_VISTAS`]), de la más antigua a la última.
    pub fn vistas(&self) -> Vec<PeticionVista> {
        let vistas = self.estado.vistas.lock().unwrap();
        vistas.iter().cloned().collect()
    }
}

impl Drop for ServidorFalso {
    fn drop(&mut self) {
        if let Some(a) = self.apagar.take() {
            let _ = a.send(());
        }
    }
}

pub async fn arrancar(direccion: &str, opciones: OpcionesFalso) -> std::io::Result<ServidorFalso> {
    let listener = TcpListener::bind(direccion).await?;
    let direccion = listener.local_addr()?;
    let estado = Arc::new(Estado {
        opciones,
        vistas: Mutex::new(VecDeque::new()),
    });
    let app = Router::new().fallback(atender).with_state(estado.clone());
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = axum::serve(
            // Sin Nagle: cada evento SSE sale en cuanto llega, sin esperar al ACK del anterior.
            listener.tap_io(|tcp| {
                let _ = tcp.set_nodelay(true);
            }),
            app,
        )
        .with_graceful_shutdown(async move {
            let _ = rx.await;
        })
        .await;
    });
    Ok(ServidorFalso {
        direccion,
        estado,
        apagar: Some(tx),
    })
}

fn num_cabecera(h: &HeaderMap, n: &str) -> Option<u64> {
    h.get(n)?.to_str().ok()?.trim().parse().ok()
}

#[derive(Clone, Copy, PartialEq)]
enum Api {
    Anthropic,
    Chat,
    Responses,
    Gemini,
}

async fn atender(State(estado): State<Arc<Estado>>, peticion: Request) -> Response {
    let (partes, cuerpo) = peticion.into_parts();
    let cuerpo = axum::body::to_bytes(cuerpo, usize::MAX)
        .await
        .unwrap_or_default();
    let ruta = partes
        .uri
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
    {
        let mut vistas = estado.vistas.lock().unwrap();
        if vistas.len() >= MAX_VISTAS {
            vistas.pop_front();
        }
        vistas.push_back(PeticionVista {
            metodo: partes.method.to_string(),
            ruta: ruta.clone(),
            cabeceras: partes.headers.clone(),
            cuerpo: cuerpo.clone(),
        });
    }
    let camino = ruta.split('?').next().unwrap_or("");
    let json_cuerpo: Value = serde_json::from_slice(&cuerpo).unwrap_or(Value::Null);
    let h = &partes.headers;

    let api = if camino == "/v1/messages" {
        Api::Anthropic
    } else if camino.ends_with("/chat/completions") {
        Api::Chat
    } else if camino.ends_with("/responses") {
        Api::Responses
    } else if camino.contains(":generateContent") || camino.contains(":streamGenerateContent") {
        Api::Gemini
    } else if camino == "/v1/messages/count_tokens" {
        return json_resp(StatusCode::OK, json!({"input_tokens": ENTRADA}));
    } else if camino.ends_with("/models") && partes.method == "GET" {
        return json_resp(
            StatusCode::OK,
            json!({"data": [{"id": "fake-model", "type": "model"}], "object": "list"}),
        );
    } else {
        return json_resp(
            StatusCode::NOT_FOUND,
            json!({"error": {"message": format!("fake: unknown route {camino}")}}),
        );
    };

    if let Some(s) = num_cabecera(h, "x-fake-status")
        && let Ok(estado_http) = StatusCode::from_u16(s as u16)
    {
        let mut r = json_resp(estado_http, cuerpo_error(api, estado_http));
        r.headers_mut()
            .insert("retry-after", HeaderValue::from_static("7"));
        r.headers_mut()
            .insert("x-fake-error", HeaderValue::from_static("1"));
        return r;
    }

    let op = &estado.opciones;
    let tokens = num_cabecera(h, "x-fake-tokens").unwrap_or(op.tokens as u64) as usize;
    let retardo = num_cabecera(h, "x-fake-delay-ms").unwrap_or(op.retardo_token_ms);
    if op.retardo_inicial_ms > 0 {
        tokio::time::sleep(Duration::from_millis(op.retardo_inicial_ms)).await;
    }
    let modelo = json_cuerpo
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            // Gemini lleva el modelo en la ruta: /v1beta/models/<modelo>:generateContent
            camino
                .split("/models/")
                .nth(1)
                .and_then(|r| r.split(':').next())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "fake-model".into());
    let palabras: Vec<String> = (0..tokens).map(|i| format!("w{i} ")).collect();
    let stream = match api {
        Api::Gemini => camino.contains(":streamGenerateContent"),
        _ => json_cuerpo
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let sse_gemini = ruta.contains("alt=sse");

    if !stream {
        let texto: String = palabras.concat();
        let cuerpo = match api {
            Api::Anthropic => json!({
                "id": "msg_fake", "type": "message", "role": "assistant", "model": modelo,
                "content": [{"type": "text", "text": texto}],
                "stop_reason": "end_turn", "stop_sequence": null,
                "usage": {"input_tokens": ENTRADA, "output_tokens": tokens,
                          "cache_read_input_tokens": CACHE_LEIDA, "cache_creation_input_tokens": CACHE_ESCRITA}
            }),
            Api::Chat => json!({
                "id": "chatcmpl-fake", "object": "chat.completion", "model": modelo,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": ENTRADA + CACHE_LEIDA, "completion_tokens": tokens,
                          "prompt_tokens_details": {"cached_tokens": CACHE_LEIDA}}
            }),
            Api::Responses => respuesta_responses(&modelo, &texto, tokens),
            Api::Gemini => respuesta_gemini(&modelo, &texto, tokens, true),
        };
        return json_resp(StatusCode::OK, cuerpo);
    }

    // Streaming: lista de trozos que se mandan con `retardo` entre ellos.
    let mut trozos: Vec<String> = Vec::new();
    match api {
        Api::Anthropic => {
            trozos.push(sse("message_start", &json!({"type": "message_start", "message": {
                "id": "msg_fake", "type": "message", "role": "assistant", "model": modelo, "content": [],
                "stop_reason": null, "usage": {"input_tokens": ENTRADA, "output_tokens": 1,
                "cache_read_input_tokens": CACHE_LEIDA, "cache_creation_input_tokens": CACHE_ESCRITA}}})));
            trozos.push(sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            ));
            for p in &palabras {
                trozos.push(sse(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": p}}),
                ));
            }
            trozos.push(sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ));
            trozos.push(sse(
                "message_delta",
                &json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"output_tokens": tokens}}),
            ));
            trozos.push(sse("message_stop", &json!({"type": "message_stop"})));
        }
        Api::Chat => {
            for p in &palabras {
                trozos.push(sse_datos(&json!({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                    "model": modelo, "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": null}],
                    "usage": null})));
            }
            trozos.push(sse_datos(&json!({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                "model": modelo, "choices": [], "usage": {"prompt_tokens": ENTRADA + CACHE_LEIDA,
                "completion_tokens": tokens, "prompt_tokens_details": {"cached_tokens": CACHE_LEIDA}}})));
            trozos.push("data: [DONE]\n\n".to_string());
        }
        Api::Responses => {
            trozos.push(sse("response.created", &json!({"type": "response.created",
                "response": {"id": "resp_fake", "model": modelo, "status": "in_progress", "usage": null}})));
            for p in &palabras {
                trozos.push(sse(
                    "response.output_text.delta",
                    &json!({"type": "response.output_text.delta", "delta": p}),
                ));
            }
            let texto: String = palabras.concat();
            trozos.push(sse(
                "response.completed",
                &json!({"type": "response.completed",
                "response": respuesta_responses(&modelo, &texto, tokens)}),
            ));
        }
        Api::Gemini => {
            let n = palabras.len();
            for (i, p) in palabras.iter().enumerate() {
                let v = respuesta_gemini(&modelo, p, i + 1, i + 1 == n);
                if sse_gemini {
                    trozos.push(sse_datos(&v));
                } else {
                    let sep = if i == 0 { "[" } else { "," };
                    trozos.push(format!("{sep}{v}"));
                }
            }
            if !sse_gemini {
                if n == 0 {
                    trozos.push("[".into());
                }
                trozos.push("]".into());
            }
        }
    }
    let tipo = if api == Api::Gemini && !sse_gemini {
        "application/json"
    } else {
        "text/event-stream"
    };
    // Plazos absolutos desde el primer trozo, no una espera tras otra: en Windows cada espera corta
    // dura un tic del sistema (~15,6 ms), y 50 tokens a 2 ms tardaban 800 ms en vez de 100.
    let inicio = Instant::now();
    let flujo =
        futures_util::stream::iter(trozos.into_iter().enumerate()).then(move |(i, t)| async move {
            if i > 0 && retardo > 0 {
                let plazo = Duration::from_millis(retardo).saturating_mul(i as u32);
                match inicio.checked_add(plazo) {
                    Some(p) => tokio::time::sleep_until(p).await,
                    None => tokio::time::sleep(plazo).await,
                }
            }
            Ok::<_, std::io::Error>(Bytes::from(t))
        });
    let mut r = Response::new(Body::from_stream(flujo));
    r.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(tipo));
    r.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    r.headers_mut()
        .insert("request-id", HeaderValue::from_static("req_fake"));
    r
}

fn respuesta_responses(modelo: &str, texto: &str, tokens: usize) -> Value {
    json!({"id": "resp_fake", "object": "response", "model": modelo, "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": texto}]}],
        "usage": {"input_tokens": ENTRADA + CACHE_LEIDA, "output_tokens": tokens,
                  "input_tokens_details": {"cached_tokens": CACHE_LEIDA}}})
}

fn respuesta_gemini(modelo: &str, texto: &str, tokens: usize, ultimo: bool) -> Value {
    let mut v = json!({"candidates": [{"content": {"role": "model", "parts": [{"text": texto}]}, "index": 0}],
        "modelVersion": modelo,
        "usageMetadata": {"promptTokenCount": ENTRADA + CACHE_LEIDA, "candidatesTokenCount": tokens,
                          "cachedContentTokenCount": CACHE_LEIDA}});
    if ultimo {
        v["candidates"][0]["finishReason"] = json!("STOP");
    }
    v
}

fn cuerpo_error(api: Api, estado: StatusCode) -> Value {
    let mensaje = format!("fake error {}", estado.as_u16());
    match api {
        Api::Anthropic => {
            json!({"type": "error", "error": {"type": "rate_limit_error", "message": mensaje}})
        }
        Api::Chat | Api::Responses => {
            json!({"error": {"message": mensaje, "type": "rate_limit_error"}})
        }
        Api::Gemini => {
            json!({"error": {"code": estado.as_u16(), "message": mensaje, "status": "RESOURCE_EXHAUSTED"}})
        }
    }
}

fn sse(evento: &str, datos: &Value) -> String {
    format!("event: {evento}\ndata: {datos}\n\n")
}

fn sse_datos(datos: &Value) -> String {
    format!("data: {datos}\n\n")
}

fn json_resp(estado: StatusCode, v: Value) -> Response {
    let mut r = Response::new(Body::from(v.to_string()));
    *r.status_mut() = estado;
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    r
}
