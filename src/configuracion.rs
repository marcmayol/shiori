//! Configuración: valores por defecto < archivo global < variables de entorno para el servidor, y
//! archivo del proyecto < cabeceras para cada petición.
//!
//! YAML es el formato principal; TOML se acepta con las mismas claves. Las claves desconocidas son
//! error, y todo error dice el archivo y la línea.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::dominio::precios::Precio;
use crate::dominio::proveedor::Proveedor;

pub const PUERTO_POR_DEFECTO: u16 = 8787;
pub const NOMBRES_PROYECTO: [&str; 3] = ["shiori.yaml", "shiori.yml", "shiori.toml"];
const NOMBRES_GLOBAL: [&str; 3] = ["config.yaml", "config.yml", "config.toml"];

// ---------- Archivos tal como se escriben ----------

/// Global configuration file (`shiori config path` shows where it goes).
#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArchivoGlobal {
    pub server: Option<SeccionServidor>,
    /// Base URL of each real API.
    pub upstreams: Option<SeccionDestinos>,
    pub metrics: Option<SeccionMetricas>,
    /// Prices in USD per million tokens, by model id prefix. They override the built-in table.
    pub prices: Option<BTreeMap<String, Precio>>,
    pub layers: Option<SeccionCapas>,
}

/// Project file (`shiori.yaml` in the repository root). It only sets what belongs to the project:
/// it may come from someone else's repository, so it cannot change the server or the upstreams.
#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArchivoProyecto {
    /// Project name in the metrics. Default: the folder name.
    pub project: Option<String>,
    pub layers: Option<SeccionCapas>,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeccionServidor {
    /// Default: 127.0.0.1 (only this computer).
    pub host: Option<String>,
    /// Default: 8787.
    pub port: Option<u16>,
    /// Largest request body accepted, in MB. Default: 256.
    pub max_body_mb: Option<u64>,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeccionDestinos {
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    /// ChatGPT subscription API (Codex signed in with a ChatGPT account).
    pub chatgpt: Option<String>,
    pub gemini: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeccionMetricas {
    /// Default: true.
    pub enabled: Option<bool>,
    /// SQLite file. Default: `<data dir>/shiori/metrics.db`.
    pub path: Option<PathBuf>,
}

/// Optimisation layers. None is available yet in this version: they arrive in phase 3.
#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeccionCapas {
    pub exact_cache: Option<CapaSimple>,
    pub prompt_cache: Option<CapaSimple>,
    pub router: Option<CapaSimple>,
    pub semantic_cache: Option<CapaSemantica>,
    pub compaction: Option<CapaCompactacion>,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CapaSimple {
    pub enabled: Option<bool>,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CapaSemantica {
    pub enabled: Option<bool>,
    /// Similarity needed to reuse an answer, between 0 and 1.
    pub threshold: Option<f32>,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CapaCompactacion {
    pub enabled: Option<bool>,
    /// 1: trim old tool outputs; 2: also drop duplicates; 3: also summarise old history.
    pub level: Option<u8>,
}

// ---------- Errores ----------

#[derive(Debug, Clone, PartialEq)]
pub struct ErrorConfig {
    pub archivo: Option<PathBuf>,
    pub linea: Option<usize>,
    pub columna: Option<usize>,
    pub mensaje: String,
}

impl ErrorConfig {
    fn nuevo(mensaje: impl Into<String>) -> Self {
        ErrorConfig {
            archivo: None,
            linea: None,
            columna: None,
            mensaje: mensaje.into(),
        }
    }
}

impl fmt::Display for ErrorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(a) = &self.archivo {
            write!(f, "{}", a.display())?;
            if let Some(l) = self.linea {
                write!(f, ":{l}")?;
                if let Some(c) = self.columna {
                    write!(f, ":{c}")?;
                }
            }
            write!(f, ": ")?;
        }
        write!(f, "{}", self.mensaje)
    }
}

impl std::error::Error for ErrorConfig {}

// ---------- Lectura y validación de archivos ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Formato {
    Yaml,
    Toml,
}

pub fn formato_de(ruta: &Path) -> Formato {
    match ruta.extension().and_then(|e| e.to_str()) {
        Some("toml") => Formato::Toml,
        _ => Formato::Yaml,
    }
}

fn solo_comentarios(texto: &str, formato: Formato) -> bool {
    texto.lines().all(|l| {
        let l = l.trim();
        l.is_empty() || l.starts_with('#') || (formato == Formato::Yaml && l == "---")
    })
}

/// Línea (desde 1) de la primera aparición de `clave:` o `clave =`, para los errores de validación
/// que no vienen del parser.
fn linea_de_clave(texto: &str, clave: &str) -> Option<usize> {
    texto
        .lines()
        .position(|l| {
            let l = l.trim_start().trim_start_matches("- ");
            l.strip_prefix(clave)
                .map(|r| {
                    let r = r.trim_start();
                    r.starts_with(':') || r.starts_with('=')
                })
                .unwrap_or(false)
        })
        .map(|i| i + 1)
}

fn linea_y_columna(texto: &str, offset: usize) -> (usize, usize) {
    let antes = &texto[..offset.min(texto.len())];
    let linea = antes.matches('\n').count() + 1;
    let columna = antes
        .rsplit('\n')
        .next()
        .map(|s| s.chars().count())
        .unwrap_or(0)
        + 1;
    (linea, columna)
}

pub fn parsear<T: DeserializeOwned + Default>(
    texto: &str,
    formato: Formato,
    ruta: Option<&Path>,
) -> Result<T, ErrorConfig> {
    if solo_comentarios(texto, formato) {
        return Ok(T::default());
    }
    let con_ruta = |mut e: ErrorConfig| {
        e.archivo = ruta.map(Path::to_path_buf);
        e
    };
    match formato {
        Formato::Yaml => serde_yaml_ng::from_str::<T>(texto).map_err(|e| {
            let (linea, columna) = e
                .location()
                .map(|l| (Some(l.line()), Some(l.column())))
                .unwrap_or((None, None));
            let mut mensaje = e.to_string();
            if let Some(i) = mensaje.find(" at line ") {
                mensaje.truncate(i);
            }
            con_ruta(ErrorConfig {
                archivo: None,
                linea,
                columna,
                mensaje,
            })
        }),
        Formato::Toml => toml::from_str::<T>(texto).map_err(|e| {
            let (linea, columna) = e
                .span()
                .map(|s| {
                    let (l, c) = linea_y_columna(texto, s.start);
                    (Some(l), Some(c))
                })
                .unwrap_or((None, None));
            con_ruta(ErrorConfig {
                archivo: None,
                linea,
                columna,
                mensaje: e.message().trim().to_string(),
            })
        }),
    }
}

fn validar_url(clave: &str, valor: &str) -> Result<(), String> {
    let url =
        reqwest::Url::parse(valor).map_err(|e| format!("{clave}: invalid URL `{valor}`: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!(
            "{clave}: the URL must start with http:// or https://"
        ));
    }
    Ok(())
}

/// Comprueba los valores que el parser no puede comprobar. Devuelve (clave para buscar la línea, mensaje).
fn validar_capas(capas: &SeccionCapas) -> Result<(), (&'static str, String)> {
    if let Some(s) = &capas.semantic_cache
        && let Some(t) = s.threshold
        && !(t > 0.0 && t <= 1.0)
    {
        return Err((
            "threshold",
            format!("threshold must be between 0 and 1 (got {t})"),
        ));
    }
    if let Some(c) = &capas.compaction
        && let Some(n) = c.level
        && !(1..=3).contains(&n)
    {
        return Err(("level", format!("level must be 1, 2 or 3 (got {n})")));
    }
    let activas = [
        (
            "exact_cache",
            capas.exact_cache.as_ref().and_then(|c| c.enabled),
        ),
        (
            "prompt_cache",
            capas.prompt_cache.as_ref().and_then(|c| c.enabled),
        ),
        ("router", capas.router.as_ref().and_then(|c| c.enabled)),
        (
            "semantic_cache",
            capas.semantic_cache.as_ref().and_then(|c| c.enabled),
        ),
        (
            "compaction",
            capas.compaction.as_ref().and_then(|c| c.enabled),
        ),
    ];
    if let Some((nombre, _)) = activas.iter().find(|(_, a)| *a == Some(true)) {
        return Err((
            nombre,
            format!(
                "layer `{nombre}` is not available yet in Shiori {} (it arrives in a later version); \
                 set `enabled: false` or remove it",
                crate::VERSION
            ),
        ));
    }
    Ok(())
}

fn validar_global(a: &ArchivoGlobal) -> Result<(), (&'static str, String)> {
    if let Some(d) = &a.upstreams {
        for (clave, valor) in [
            ("anthropic", &d.anthropic),
            ("openai", &d.openai),
            ("chatgpt", &d.chatgpt),
            ("gemini", &d.gemini),
        ] {
            if let Some(v) = valor {
                validar_url(clave, v).map_err(|m| (clave, m))?;
            }
        }
    }
    if let Some(s) = &a.server
        && s.max_body_mb == Some(0)
    {
        return Err(("max_body_mb", "max_body_mb must be greater than 0".into()));
    }
    if let Some(precios) = &a.prices {
        for (modelo, p) in precios {
            let valores = [Some(p.input), Some(p.output), p.cache_read, p.cache_write];
            if valores.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
                return Err((
                    "prices",
                    format!("prices for `{modelo}` cannot be negative"),
                ));
            }
        }
    }
    if let Some(c) = &a.layers {
        validar_capas(c)?;
    }
    Ok(())
}

fn validar_proyecto(a: &ArchivoProyecto) -> Result<(), (&'static str, String)> {
    if let Some(p) = &a.project
        && p.trim().is_empty()
    {
        return Err(("project", "project cannot be empty".into()));
    }
    if let Some(c) = &a.layers {
        validar_capas(c)?;
    }
    Ok(())
}

fn leer(ruta: &Path) -> Result<String, ErrorConfig> {
    std::fs::read_to_string(ruta).map_err(|e| ErrorConfig {
        archivo: Some(ruta.to_path_buf()),
        linea: None,
        columna: None,
        mensaje: format!("cannot read the file: {e}"),
    })
}

fn con_linea(texto: &str, ruta: &Path, (clave, mensaje): (&str, String)) -> ErrorConfig {
    ErrorConfig {
        archivo: Some(ruta.to_path_buf()),
        linea: linea_de_clave(texto, clave),
        columna: None,
        mensaje,
    }
}

pub fn leer_global(ruta: &Path) -> Result<ArchivoGlobal, ErrorConfig> {
    let texto = leer(ruta)?;
    let a: ArchivoGlobal = parsear(&texto, formato_de(ruta), Some(ruta))?;
    validar_global(&a).map_err(|e| con_linea(&texto, ruta, e))?;
    Ok(a)
}

pub fn leer_proyecto(ruta: &Path) -> Result<ArchivoProyecto, ErrorConfig> {
    let texto = leer(ruta)?;
    let a: ArchivoProyecto = parsear(&texto, formato_de(ruta), Some(ruta))?;
    validar_proyecto(&a).map_err(|e| con_linea(&texto, ruta, e))?;
    Ok(a)
}

/// Busca un único archivo entre `nombres` en `carpeta`; dos a la vez es error, para que no haya dudas
/// de cuál manda.
fn buscar_unico(carpeta: &Path, nombres: &[&str]) -> Result<Option<PathBuf>, ErrorConfig> {
    let encontrados: Vec<PathBuf> = nombres
        .iter()
        .map(|n| carpeta.join(n))
        .filter(|p| p.is_file())
        .collect();
    match encontrados.len() {
        0 => Ok(None),
        1 => Ok(encontrados.into_iter().next()),
        _ => Err(ErrorConfig::nuevo(format!(
            "there are several configuration files in {}: {}; keep only one",
            carpeta.display(),
            encontrados
                .iter()
                .filter_map(|p| p.file_name()?.to_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

pub fn archivo_de_proyecto(carpeta: &Path) -> Result<Option<PathBuf>, ErrorConfig> {
    buscar_unico(carpeta, &NOMBRES_PROYECTO)
}

pub fn carpeta_config() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("shiori"))
}

/// Carpeta de datos (métricas, registro, clave de parada). `SHIORI_DATA_DIR` la cambia.
pub fn carpeta_datos() -> PathBuf {
    if let Some(d) = std::env::var_os("SHIORI_DATA_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    dirs::data_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("shiori")
}

/// La ruta del archivo global en uso: `SHIORI_CONFIG` o el que haya en la carpeta de configuración.
pub fn ruta_global(
    entorno: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<PathBuf>, ErrorConfig> {
    if let Some(r) = entorno("SHIORI_CONFIG").filter(|r| !r.is_empty()) {
        let r = PathBuf::from(r);
        if !r.is_file() {
            return Err(ErrorConfig::nuevo(format!(
                "SHIORI_CONFIG points to {}, which does not exist",
                r.display()
            )));
        }
        return Ok(Some(r));
    }
    match carpeta_config() {
        Some(c) => buscar_unico(&c, &NOMBRES_GLOBAL),
        None => Ok(None),
    }
}

/// Decide si un archivo es de proyecto (`shiori.*`) o global por su nombre.
pub fn es_de_proyecto(ruta: &Path) -> bool {
    ruta.file_name()
        .and_then(|n| n.to_str())
        .map(|n| NOMBRES_PROYECTO.contains(&n))
        .unwrap_or(false)
}

// ---------- Configuración efectiva ----------

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub max_cuerpo_bytes: usize,
    pub destinos: BTreeMap<Proveedor, String>,
    pub metricas: bool,
    pub ruta_metricas: PathBuf,
    pub precios: BTreeMap<String, Precio>,
    /// El archivo global leído, si había.
    pub archivo: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: "127.0.0.1".into(),
            port: PUERTO_POR_DEFECTO,
            max_cuerpo_bytes: 256 * 1024 * 1024,
            destinos: Proveedor::TODOS
                .into_iter()
                .map(|p| (p, p.destino_por_defecto().to_string()))
                .collect(),
            metricas: true,
            ruta_metricas: carpeta_datos().join("metrics.db"),
            precios: BTreeMap::new(),
            archivo: None,
        }
    }
}

impl Config {
    pub fn destino(&self, p: Proveedor) -> &str {
        self.destinos
            .get(&p)
            .map(String::as_str)
            .unwrap_or_else(|| p.destino_por_defecto())
    }

    /// La URL con la que se llega al proxy desde este equipo.
    pub fn url_local(&self) -> String {
        let host = match self.host.as_str() {
            "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
            h => h,
        };
        if host.contains(':') && !host.starts_with('[') {
            format!("http://[{host}]:{}", self.port)
        } else {
            format!("http://{host}:{}", self.port)
        }
    }

    fn aplicar_archivo(&mut self, a: ArchivoGlobal) {
        if let Some(s) = a.server {
            if let Some(h) = s.host {
                self.host = h;
            }
            if let Some(p) = s.port {
                self.port = p;
            }
            if let Some(mb) = s.max_body_mb {
                self.max_cuerpo_bytes = (mb as usize).saturating_mul(1024 * 1024);
            }
        }
        if let Some(d) = a.upstreams {
            for (p, v) in [
                (Proveedor::Anthropic, d.anthropic),
                (Proveedor::Openai, d.openai),
                (Proveedor::Chatgpt, d.chatgpt),
                (Proveedor::Gemini, d.gemini),
            ] {
                if let Some(v) = v {
                    self.destinos.insert(p, v);
                }
            }
        }
        if let Some(m) = a.metrics {
            if let Some(e) = m.enabled {
                self.metricas = e;
            }
            if let Some(r) = m.path {
                self.ruta_metricas = r;
            }
        }
        if let Some(p) = a.prices {
            self.precios = p;
        }
    }

    fn aplicar_entorno(
        &mut self,
        entorno: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), ErrorConfig> {
        let var = |n: &str| entorno(n).filter(|v| !v.trim().is_empty());
        if let Some(h) = var("SHIORI_HOST") {
            self.host = h;
        }
        if let Some(p) = var("SHIORI_PORT") {
            self.port = p.trim().parse().map_err(|_| {
                ErrorConfig::nuevo(format!("SHIORI_PORT: `{p}` is not a valid port"))
            })?;
        }
        if let Some(m) = var("SHIORI_METRICS") {
            self.metricas = match m.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                _ => {
                    return Err(ErrorConfig::nuevo(format!(
                        "SHIORI_METRICS: `{m}` must be true or false"
                    )));
                }
            };
        }
        if let Some(r) = var("SHIORI_METRICS_PATH") {
            self.ruta_metricas = PathBuf::from(r);
        }
        for p in Proveedor::TODOS {
            let nombre = format!("SHIORI_UPSTREAM_{}", p.nombre().to_ascii_uppercase());
            if let Some(v) = var(&nombre) {
                validar_url(&nombre, &v).map_err(ErrorConfig::nuevo)?;
                self.destinos.insert(p, v);
            }
        }
        Ok(())
    }
}

/// Carga la configuración del servidor: por defecto < archivo global < entorno.
pub fn cargar(entorno: &dyn Fn(&str) -> Option<String>) -> Result<Config, ErrorConfig> {
    let mut c = Config::default();
    if let Some(ruta) = ruta_global(entorno)? {
        let a = leer_global(&ruta)?;
        c.aplicar_archivo(a);
        c.archivo = Some(ruta);
    }
    c.aplicar_entorno(entorno)?;
    Ok(c)
}

pub fn entorno_real(nombre: &str) -> Option<String> {
    std::env::var(nombre).ok()
}

pub fn esquema(de_proyecto: bool) -> String {
    let s = if de_proyecto {
        schemars::schema_for!(ArchivoProyecto)
    } else {
        schemars::schema_for!(ArchivoGlobal)
    };
    serde_json::to_string_pretty(&s).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml<T: DeserializeOwned + Default>(t: &str) -> Result<T, ErrorConfig> {
        parsear(t, Formato::Yaml, Some(Path::new("c.yaml")))
    }

    #[test]
    fn vacio_y_comentarios() {
        let a: ArchivoGlobal = yaml("").unwrap();
        assert!(a.server.is_none());
        let a: ArchivoGlobal = yaml("# nada\n---\n").unwrap();
        assert!(a.server.is_none());
    }

    #[test]
    fn clave_desconocida_con_linea() {
        let e = yaml::<ArchivoGlobal>("server:\n  port: 9000\n  hots: x\n").unwrap_err();
        assert_eq!(e.linea, Some(3), "{e}");
        assert!(e.mensaje.contains("hots"), "{e}");
        assert!(e.to_string().starts_with("c.yaml:3"), "{e}");
    }

    #[test]
    fn tipo_erroneo_con_linea() {
        let e = yaml::<ArchivoGlobal>("server:\n  port: muchos\n").unwrap_err();
        assert_eq!(e.linea, Some(2), "{e}");
    }

    #[test]
    fn toml_con_linea() {
        let e = parsear::<ArchivoGlobal>(
            "[server]\nport = 1\nfoo = 2\n",
            Formato::Toml,
            Some(Path::new("c.toml")),
        )
        .unwrap_err();
        assert_eq!(e.linea, Some(3), "{e}");
    }

    #[test]
    fn proyecto_no_toca_servidor() {
        let e = yaml::<ArchivoProyecto>("project: x\nserver:\n  port: 1\n").unwrap_err();
        assert_eq!(e.linea, Some(2), "{e}");
        let e = yaml::<ArchivoProyecto>("upstreams:\n  anthropic: http://malo\n").unwrap_err();
        assert_eq!(e.linea, Some(1), "{e}");
    }

    #[test]
    fn capas_no_disponibles_y_rangos() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path().join("shiori.yaml");
        std::fs::write(
            &r,
            "project: a\nlayers:\n  exact_cache:\n    enabled: true\n",
        )
        .unwrap();
        let e = leer_proyecto(&r).unwrap_err();
        assert!(e.mensaje.contains("not available"), "{e}");
        assert_eq!(e.linea, Some(3));

        std::fs::write(
            &r,
            "layers:\n  semantic_cache:\n    enabled: false\n    threshold: 1.5\n",
        )
        .unwrap();
        let e = leer_proyecto(&r).unwrap_err();
        assert_eq!(e.linea, Some(4), "{e}");

        std::fs::write(&r, "layers:\n  exact_cache:\n    enabled: false\n").unwrap();
        assert!(leer_proyecto(&r).is_ok());
    }

    #[test]
    fn prioridad_global_y_entorno() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path().join("config.yaml");
        std::fs::write(
            &r,
            "server:\n  port: 9100\n  host: 127.0.0.2\nupstreams:\n  openai: http://localhost:1\nmetrics:\n  enabled: false\n",
        )
        .unwrap();
        let ruta = r.to_string_lossy().to_string();
        let entorno = |n: &str| match n {
            "SHIORI_CONFIG" => Some(ruta.clone()),
            "SHIORI_PORT" => Some("9200".into()),
            _ => None,
        };
        let c = cargar(&entorno).unwrap();
        assert_eq!(c.port, 9200, "el entorno gana al archivo");
        assert_eq!(c.host, "127.0.0.2", "el archivo gana al defecto");
        assert_eq!(c.destino(Proveedor::Openai), "http://localhost:1");
        assert_eq!(c.destino(Proveedor::Anthropic), "https://api.anthropic.com");
        assert!(!c.metricas);
        assert_eq!(c.archivo.as_deref(), Some(r.as_path()));
    }

    #[test]
    fn entorno_invalido() {
        let e = cargar(&|n: &str| (n == "SHIORI_PORT").then(|| "x".to_string())).unwrap_err();
        assert!(e.mensaje.contains("SHIORI_PORT"));
        let e = cargar(&|n: &str| (n == "SHIORI_UPSTREAM_GEMINI").then(|| "ftp://x".to_string()))
            .unwrap_err();
        assert!(e.mensaje.contains("http"));
    }

    #[test]
    fn url_mala_en_archivo() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path().join("config.toml");
        std::fs::write(
            &r,
            "[server]\nport = 1\n\n[upstreams]\nanthropic = \"nada\"\n",
        )
        .unwrap();
        let e = leer_global(&r).unwrap_err();
        assert_eq!(e.linea, Some(5), "{e}");
    }

    #[test]
    fn dos_archivos_de_proyecto() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("shiori.yaml"), "").unwrap();
        std::fs::write(dir.path().join("shiori.toml"), "").unwrap();
        assert!(archivo_de_proyecto(dir.path()).is_err());
    }

    #[test]
    fn url_local() {
        let mut c = Config::default();
        assert_eq!(c.url_local(), "http://127.0.0.1:8787");
        c.host = "0.0.0.0".into();
        assert_eq!(c.url_local(), "http://127.0.0.1:8787");
        c.host = "::1".into();
        assert_eq!(c.url_local(), "http://[::1]:8787");
    }

    #[test]
    fn esquemas() {
        assert!(esquema(false).contains("upstreams"));
        assert!(!esquema(true).contains("\"upstreams\""));
    }
}
