//! El caso de uso del proxy: recibe una petición, decide proveedor y proyecto, la reenvía tal cual y
//! devuelve la respuesta en streaming mientras mide el uso. Solo habla con puertos.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime};

use axum::http::header::{self, HeaderName, HeaderValue};
use axum::http::{HeaderMap, Method, StatusCode};
use bytes::Bytes;
use futures_util::Stream;
use serde_json::json;

use crate::configuracion::{self, ArchivoProyecto, Config, ErrorConfig};
use crate::dominio::autenticacion::{self, Autenticacion};
use crate::dominio::llamada::{Llamada, ahora_ms};
use crate::dominio::precios::TablaPrecios;
use crate::dominio::proveedor::{Proveedor, resolver_ruta};
use crate::dominio::puertos::{AlmacenMetricas, Destino, FlujoBytes, PeticionDestino};
use crate::dominio::texto;
use crate::dominio::uso::Medidor;

pub const CABECERA_PROYECTO: &str = "x-shiori-project";
pub const CABECERA_CARPETA: &str = "x-shiori-dir";
/// Prefijo de ruta para clientes que no pueden poner cabeceras: `/_shiori/dir/<carpeta codificada>/…`.
pub const PREFIJO_CARPETA: &str = "/_shiori/dir/";

pub struct Entrada {
    pub metodo: Method,
    /// Ruta con la query, tal como llegó (`/v1/messages?beta=true`).
    pub ruta: String,
    pub cabeceras: HeaderMap,
    pub cuerpo: Bytes,
}

pub struct Salida {
    pub estado: StatusCode,
    pub cabeceras: HeaderMap,
    pub cuerpo: FlujoBytes,
}

pub struct Proxy<D: Destino> {
    config: Arc<Config>,
    destino: D,
    metricas: Arc<dyn AlmacenMetricas>,
    precios: TablaPrecios,
    proyectos: CacheProyectos,
}

impl<D: Destino> Proxy<D> {
    pub fn nuevo(config: Arc<Config>, destino: D, metricas: Arc<dyn AlmacenMetricas>) -> Self {
        let precios = TablaPrecios::nueva(config.precios.clone());
        Proxy {
            config,
            destino,
            metricas,
            precios,
            proyectos: CacheProyectos::default(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn metricas(&self) -> &Arc<dyn AlmacenMetricas> {
        &self.metricas
    }

    pub async fn atender(&self, entrada: Entrada) -> Salida {
        let inicio = Instant::now();
        let momento_ms = ahora_ms();
        let (ruta_completa, query) = match entrada.ruta.split_once('?') {
            Some((r, q)) => (r.to_string(), Some(q.to_string())),
            None => (entrada.ruta.clone(), None),
        };

        // Carpeta del proyecto: por prefijo de ruta o por cabecera.
        let (carpeta_en_ruta, ruta) = match separar_carpeta(&ruta_completa) {
            Ok(v) => v,
            Err(m) => {
                let (p, _) = resolver_ruta(&ruta_completa);
                return error(p, StatusCode::BAD_REQUEST, &m);
            }
        };
        let (proveedor, ruta_destino) = resolver_ruta(&ruta);
        let carpeta = match carpeta_en_ruta {
            Some(c) => Some(c),
            None => match entrada.cabeceras.get(CABECERA_CARPETA) {
                Some(v) => match v.to_str().ok().and_then(texto::decodificar) {
                    Some(c) => Some(PathBuf::from(c)),
                    None => {
                        return error(
                            proveedor,
                            StatusCode::BAD_REQUEST,
                            "shiori: the x-shiori-dir header is not valid percent-encoded UTF-8",
                        );
                    }
                },
                None => None,
            },
        };
        let archivo_proyecto = match &carpeta {
            Some(c) => match self.proyectos.obtener(c) {
                Ok(a) => a,
                Err(e) => {
                    return error(
                        proveedor,
                        StatusCode::BAD_REQUEST,
                        &format!("shiori: invalid project configuration: {e}"),
                    );
                }
            },
            None => None,
        };
        let proyecto = nombre_proyecto(
            &entrada.cabeceras,
            archivo_proyecto.as_deref(),
            carpeta.as_deref(),
        );
        let autenticacion =
            autenticacion::detectar(proveedor, &entrada.cabeceras, query.as_deref());

        let base = self.config.destino(proveedor).trim_end_matches('/');
        let url = match &query {
            Some(q) => format!("{base}{ruta_destino}?{q}"),
            None => format!("{base}{ruta_destino}"),
        };
        let info = InfoLlamada {
            momento_ms,
            proyecto,
            proveedor,
            metodo: entrada.metodo.to_string(),
            ruta: ruta_destino.to_string(),
            autenticacion,
        };
        tracing::debug!(proyecto = %info.proyecto, proveedor = proveedor.nombre(), %url, "request");

        let peticion = PeticionDestino {
            metodo: entrada.metodo,
            url,
            cabeceras: cabeceras_de_ida(&entrada.cabeceras),
            cuerpo: entrada.cuerpo,
        };
        let respuesta = match self.destino.enviar(peticion).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(proveedor = proveedor.nombre(), "upstream unreachable: {e}");
                self.registrar(
                    &info,
                    StatusCode::BAD_GATEWAY,
                    false,
                    Default::default(),
                    inicio,
                    None,
                    0,
                    false,
                );
                return error(
                    proveedor,
                    StatusCode::BAD_GATEWAY,
                    &format!("shiori: cannot reach {}: {e}", proveedor.nombre()),
                );
            }
        };

        let es_sse = respuesta
            .cabeceras
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.starts_with("text/event-stream"))
            .unwrap_or(false);
        // HEAD, 204 y 304 no llevan cuerpo: terminar sin bytes no es un corte del cliente.
        let sin_cuerpo = info.metodo == "HEAD"
            || respuesta.estado == StatusCode::NO_CONTENT
            || respuesta.estado == StatusCode::NOT_MODIFIED;
        let esperado = if sin_cuerpo {
            Some(0)
        } else {
            respuesta
                .cabeceras
                .get(header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
        };
        let cuerpo = CuerpoMedido {
            interior: respuesta.cuerpo,
            medidor: Some(Medidor::nuevo(proveedor, es_sse)),
            info: Some(info),
            estado: respuesta.estado,
            es_sse,
            inicio,
            primer_byte: None,
            bytes: 0,
            esperado,
            metricas: self.metricas.clone(),
            precios: self.precios.clone(),
        };
        Salida {
            estado: respuesta.estado,
            cabeceras: cabeceras_de_vuelta(&respuesta.cabeceras),
            cuerpo: Box::pin(cuerpo),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn registrar(
        &self,
        info: &InfoLlamada,
        estado: StatusCode,
        stream: bool,
        uso: crate::dominio::uso::Uso,
        inicio: Instant,
        primer_byte: Option<u64>,
        bytes: u64,
        cancelada: bool,
    ) {
        registrar(
            &*self.metricas,
            &self.precios,
            info.clone(),
            estado,
            stream,
            uso,
            inicio,
            primer_byte,
            bytes,
            cancelada,
        );
    }
}

#[derive(Clone)]
struct InfoLlamada {
    momento_ms: i64,
    proyecto: String,
    proveedor: Proveedor,
    metodo: String,
    ruta: String,
    autenticacion: Autenticacion,
}

#[allow(clippy::too_many_arguments)]
fn registrar(
    metricas: &dyn AlmacenMetricas,
    precios: &TablaPrecios,
    info: InfoLlamada,
    estado: StatusCode,
    stream: bool,
    uso: crate::dominio::uso::Uso,
    inicio: Instant,
    primer_byte_ms: Option<u64>,
    bytes: u64,
    cancelada: bool,
) {
    let (coste_usd, ahorro_usd) = match precios.coste(&uso) {
        Some((c, a)) => (Some(c), Some(a)),
        None => (None, None),
    };
    metricas.registrar(Llamada {
        momento_ms: info.momento_ms,
        proyecto: info.proyecto,
        proveedor: info.proveedor,
        metodo: info.metodo,
        ruta: info.ruta,
        estado: estado.as_u16(),
        stream,
        autenticacion: info.autenticacion,
        uso,
        coste_usd,
        ahorro_usd,
        latencia_ms: inicio.elapsed().as_millis() as u64,
        primer_byte_ms,
        bytes,
        cancelada,
    });
}

/// El cuerpo de la respuesta tal cual, contando bytes y leyendo el uso al pasar. Registra la llamada
/// al terminar o, si el cliente corta (o hyper suelta el cuerpo tras el último byte), al soltarse.
struct CuerpoMedido {
    interior: FlujoBytes,
    medidor: Option<Medidor>,
    info: Option<InfoLlamada>,
    estado: StatusCode,
    es_sse: bool,
    inicio: Instant,
    primer_byte: Option<u64>,
    bytes: u64,
    esperado: Option<u64>,
    metricas: Arc<dyn AlmacenMetricas>,
    precios: TablaPrecios,
}

impl CuerpoMedido {
    fn cerrar(&mut self, cancelada: bool) {
        let (Some(info), Some(medidor)) = (self.info.take(), self.medidor.take()) else {
            return;
        };
        let uso = medidor.terminar();
        registrar(
            &*self.metricas,
            &self.precios,
            info,
            self.estado,
            self.es_sse,
            uso,
            self.inicio,
            self.primer_byte,
            self.bytes,
            cancelada,
        );
    }
}

impl Stream for CuerpoMedido {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let resultado = self.interior.as_mut().poll_next(cx);
        match &resultado {
            Poll::Ready(Some(Ok(trozo))) => {
                if self.primer_byte.is_none() {
                    self.primer_byte = Some(self.inicio.elapsed().as_millis() as u64);
                }
                self.bytes += trozo.len() as u64;
                if let Some(m) = self.medidor.as_mut() {
                    m.alimentar(trozo);
                }
            }
            Poll::Ready(Some(Err(e))) => {
                tracing::warn!("upstream body error: {e}");
                self.cerrar(true);
            }
            Poll::Ready(None) => self.cerrar(false),
            Poll::Pending => {}
        }
        resultado
    }
}

impl Drop for CuerpoMedido {
    fn drop(&mut self) {
        let completo = self.esperado.is_some_and(|e| e == self.bytes);
        self.cerrar(!completo);
    }
}

// ---------- Proyecto ----------

fn separar_carpeta(ruta: &str) -> Result<(Option<PathBuf>, String), String> {
    let Some(resto) = ruta.strip_prefix(PREFIJO_CARPETA) else {
        return Ok((None, ruta.to_string()));
    };
    let (codificada, despues) = match resto.find('/') {
        Some(i) => (&resto[..i], &resto[i..]),
        None => (resto, "/"),
    };
    let carpeta = texto::decodificar(codificada)
        .filter(|c| !c.is_empty())
        .ok_or_else(|| {
            "shiori: the folder in /_shiori/dir/ is not valid percent-encoded UTF-8".to_string()
        })?;
    Ok((Some(PathBuf::from(carpeta)), despues.to_string()))
}

fn nombre_proyecto(
    cabeceras: &HeaderMap,
    archivo: Option<&ArchivoProyecto>,
    carpeta: Option<&Path>,
) -> String {
    if let Some(p) = cabeceras
        .get(CABECERA_PROYECTO)
        .and_then(|v| v.to_str().ok())
        .and_then(texto::decodificar)
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
    {
        return p;
    }
    if let Some(p) = archivo.and_then(|a| a.project.clone()) {
        return p;
    }
    if let Some(n) = carpeta.and_then(|c| c.file_name()).and_then(|n| n.to_str()) {
        return n.to_string();
    }
    "default".to_string()
}

/// Ruta y fecha de modificación del archivo de proyecto: si cambia, se vuelve a leer.
type Firma = Option<(PathBuf, SystemTime)>;
type Leido = (Firma, Option<Arc<ArchivoProyecto>>);

/// Archivos de proyecto ya leídos, por carpeta; se vuelven a leer si cambian.
#[derive(Default)]
struct CacheProyectos {
    leidos: Mutex<HashMap<PathBuf, Leido>>,
}

impl CacheProyectos {
    fn obtener(&self, carpeta: &Path) -> Result<Option<Arc<ArchivoProyecto>>, ErrorConfig> {
        let ruta = configuracion::archivo_de_proyecto(carpeta)?;
        let firma = ruta
            .as_ref()
            .and_then(|r| Some((r.clone(), std::fs::metadata(r).ok()?.modified().ok()?)));
        let mut leidos = self.leidos.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((f, archivo)) = leidos.get(carpeta)
            && *f == firma
        {
            return Ok(archivo.clone());
        }
        let archivo = match &ruta {
            Some(r) => Some(Arc::new(configuracion::leer_proyecto(r)?)),
            None => None,
        };
        leidos.insert(carpeta.to_path_buf(), (firma, archivo.clone()));
        Ok(archivo)
    }
}

// ---------- Cabeceras ----------

fn es_de_salto(nombre: &str) -> bool {
    matches!(
        nombre,
        "connection"
            | "keep-alive"
            | "proxy-connection"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

/// Las cabeceras del cliente, sin las de salto, `host`, `content-length` (las pone la salida) ni las
/// `x-shiori-*`. Se pide la respuesta sin comprimir para poder leer el uso; entre el cliente y el
/// proxy, en el mismo equipo, comprimir no ahorra nada.
pub fn cabeceras_de_ida(origen: &HeaderMap) -> HeaderMap {
    let mut h = HeaderMap::with_capacity(origen.len());
    for (nombre, valor) in origen {
        let n = nombre.as_str();
        if es_de_salto(n)
            || n == "host"
            || n == "content-length"
            || n == "accept-encoding"
            || n.starts_with("x-shiori-")
        {
            continue;
        }
        h.append(nombre.clone(), valor.clone());
    }
    h.insert(
        header::ACCEPT_ENCODING,
        HeaderValue::from_static("identity"),
    );
    h
}

pub fn cabeceras_de_vuelta(origen: &HeaderMap) -> HeaderMap {
    let mut h = HeaderMap::with_capacity(origen.len());
    for (nombre, valor) in origen {
        if es_de_salto(nombre.as_str()) {
            continue;
        }
        h.append(nombre.clone(), valor.clone());
    }
    h
}

// ---------- Errores con la forma de cada API ----------

/// Error propio de Shiori con el formato JSON del proveedor, para que el cliente (Claude Code, Codex,
/// un SDK) lo muestre bien.
pub fn error(proveedor: Proveedor, estado: StatusCode, mensaje: &str) -> Salida {
    let cuerpo = match proveedor {
        Proveedor::Anthropic => json!({
            "type": "error",
            "error": {"type": tipo_anthropic(estado), "message": mensaje}
        }),
        Proveedor::Openai | Proveedor::Chatgpt => json!({
            "error": {"message": mensaje, "type": "shiori_error", "param": null, "code": estado.as_u16()}
        }),
        Proveedor::Gemini => json!({
            "error": {"code": estado.as_u16(), "message": mensaje, "status": estado_gemini(estado)}
        }),
    };
    let mut cabeceras = HeaderMap::new();
    cabeceras.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    cabeceras.insert(
        HeaderName::from_static("x-shiori-error"),
        HeaderValue::from_static("1"),
    );
    let bytes = Bytes::from(cuerpo.to_string());
    Salida {
        estado,
        cabeceras,
        cuerpo: Box::pin(futures_util::stream::once(async move { Ok(bytes) })),
    }
}

fn tipo_anthropic(estado: StatusCode) -> &'static str {
    match estado.as_u16() {
        400 => "invalid_request_error",
        413 => "request_too_large",
        _ => "api_error",
    }
}

fn estado_gemini(estado: StatusCode) -> &'static str {
    match estado.as_u16() {
        400 | 413 => "INVALID_ARGUMENT",
        _ => "UNAVAILABLE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cabeceras_ida() {
        let mut o = HeaderMap::new();
        o.insert("host", "127.0.0.1:8787".parse().unwrap());
        o.insert("authorization", "Bearer sk-ant-oat01-x".parse().unwrap());
        o.insert("anthropic-beta", "oauth-2025-04-20".parse().unwrap());
        o.insert("x-shiori-project", "p".parse().unwrap());
        o.insert("connection", "keep-alive".parse().unwrap());
        o.insert("accept-encoding", "gzip, br".parse().unwrap());
        o.insert("content-length", "10".parse().unwrap());
        o.append("x-multi", "a".parse().unwrap());
        o.append("x-multi", "b".parse().unwrap());
        let h = cabeceras_de_ida(&o);
        assert!(h.get("host").is_none());
        assert!(h.get("x-shiori-project").is_none());
        assert!(h.get("connection").is_none());
        assert!(h.get("content-length").is_none());
        assert_eq!(h["accept-encoding"], "identity");
        assert_eq!(h["authorization"], "Bearer sk-ant-oat01-x");
        assert_eq!(h["anthropic-beta"], "oauth-2025-04-20");
        assert_eq!(h.get_all("x-multi").iter().count(), 2);
    }

    #[test]
    fn carpeta_en_ruta() {
        let (c, r) = separar_carpeta("/_shiori/dir/%2Fhome%2Fa%20b/openai/v1/x").unwrap();
        assert_eq!(c, Some(PathBuf::from("/home/a b")));
        assert_eq!(r, "/openai/v1/x");
        let (c, r) = separar_carpeta("/v1/messages").unwrap();
        assert!(c.is_none());
        assert_eq!(r, "/v1/messages");
        assert!(separar_carpeta("/_shiori/dir/%ZZ/v1").is_err());
    }

    #[test]
    fn prioridad_nombre_proyecto() {
        let archivo = ArchivoProyecto {
            project: Some("del-archivo".into()),
            layers: None,
        };
        let carpeta = Path::new("/x/mi-repo");
        let mut h = HeaderMap::new();
        assert_eq!(nombre_proyecto(&h, None, None), "default");
        assert_eq!(nombre_proyecto(&h, None, Some(carpeta)), "mi-repo");
        assert_eq!(
            nombre_proyecto(&h, Some(&archivo), Some(carpeta)),
            "del-archivo"
        );
        h.insert(CABECERA_PROYECTO, "de-cabecera".parse().unwrap());
        assert_eq!(
            nombre_proyecto(&h, Some(&archivo), Some(carpeta)),
            "de-cabecera"
        );
    }
}
