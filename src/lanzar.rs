//! `shiori run`: arranca el servidor en segundo plano si no está y lanza un programa (Claude Code,
//! Codex...) apuntando al proxy, con la carpeta actual como proyecto.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::adaptadores::entrada_axum::CABECERA_CLAVE_PARADA;
use crate::aplicacion::PREFIJO_CARPETA;
use crate::configuracion::{Config, carpeta_datos};
use crate::dominio::texto;

pub enum Estado {
    /// Shiori responde en esa URL (con su versión y pid).
    EnMarcha { version: String, pid: u64 },
    /// Nadie escucha en el puerto.
    Parado,
    /// Escucha otro programa.
    Ocupado,
}

fn cliente() -> reqwest::Client {
    crate::instalar_tls();
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .expect("cliente HTTP local")
}

/// Si se puede ocupar el puerto, nadie escucha en él. Es más fiable que esperar a que rechacen la
/// conexión: en Windows, conectar a un puerto local cerrado tarda unos 2 s en fallar.
fn puerto_libre(config: &Config) -> bool {
    let host = config.host.trim_start_matches('[').trim_end_matches(']');
    std::net::TcpListener::bind((host, config.port)).is_ok()
}

pub async fn estado(config: &Config) -> Estado {
    if puerto_libre(config) {
        return Estado::Parado;
    }
    let url = format!("{}/_shiori/health", config.url_local());
    match cliente().get(&url).send().await {
        Ok(r) => match r.json::<serde_json::Value>().await {
            Ok(v) if v["name"] == "shiori" => Estado::EnMarcha {
                version: v["version"].as_str().unwrap_or("?").to_string(),
                pid: v["pid"].as_u64().unwrap_or(0),
            },
            _ => Estado::Ocupado,
        },
        Err(e) if e.is_connect() => Estado::Parado,
        Err(_) => Estado::Ocupado,
    }
}

pub fn ruta_clave(config: &Config) -> PathBuf {
    carpeta_datos().join(format!("server-{}.key", config.port))
}

pub fn ruta_log() -> PathBuf {
    carpeta_datos().join("shiori.log")
}

/// Crea la clave aleatoria de `/_shiori/stop` y la guarda, solo legible por el usuario.
pub fn nueva_clave(config: &Config) -> Result<String, String> {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).map_err(|e| format!("cannot create the stop key: {e}"))?;
    let clave: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let ruta = ruta_clave(config);
    if let Some(c) = ruta.parent() {
        std::fs::create_dir_all(c).map_err(|e| format!("cannot create {}: {e}", c.display()))?;
    }
    escribir_privado(&ruta, &clave).map_err(|e| format!("cannot write {}: {e}", ruta.display()))?;
    Ok(clave)
}

#[cfg(unix)]
fn escribir_privado(ruta: &Path, texto: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(ruta)?;
    f.write_all(texto.as_bytes())
}

#[cfg(not(unix))]
fn escribir_privado(ruta: &Path, texto: &str) -> std::io::Result<()> {
    // En Windows la carpeta de datos del usuario ya es solo suya.
    std::fs::write(ruta, texto)
}

pub async fn parar(config: &Config) -> Result<bool, String> {
    let Estado::EnMarcha { .. } = estado(config).await else {
        return Ok(false);
    };
    let clave = std::fs::read_to_string(ruta_clave(config)).map_err(|e| {
        format!(
            "cannot read the stop key ({}): {e}",
            ruta_clave(config).display()
        )
    })?;
    let r = cliente()
        .post(format!("{}/_shiori/stop", config.url_local()))
        .header(CABECERA_CLAVE_PARADA, clave.trim())
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("the server refused to stop ({})", r.status()));
    }
    for _ in 0..100 {
        if matches!(estado(config).await, Estado::Parado) {
            return Ok(true);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("the server did not stop within 10 seconds".into())
}

/// Si Shiori no está escuchando, lo arranca en segundo plano (`shiori serve`) y espera a que responda.
pub async fn asegurar_servidor(config: &Config) -> Result<(), String> {
    match estado(config).await {
        Estado::EnMarcha { .. } => return Ok(()),
        Estado::Ocupado => {
            return Err(format!(
                "port {} is in use by another program; set server.port or SHIORI_PORT",
                config.port
            ));
        }
        Estado::Parado => {}
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let log = ruta_log();
    if let Some(c) = log.parent() {
        std::fs::create_dir_all(c).map_err(|e| e.to_string())?;
    }
    let salida = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
    let error = salida.try_clone().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(exe);
    cmd.arg("serve")
        .env("SHIORI_HOST", &config.host)
        .env("SHIORI_PORT", config.port.to_string())
        .stdin(Stdio::null())
        .stdout(salida)
        .stderr(error);
    separar_del_terminal(&mut cmd);
    no_heredar_stdio_propio();
    cmd.spawn()
        .map_err(|e| format!("cannot start the Shiori server: {e}"))?;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if let Estado::EnMarcha { .. } = estado(config).await {
            return Ok(());
        }
    }
    Err(format!(
        "the Shiori server did not start within 10 seconds; see {}",
        log.display()
    ))
}

/// El servidor no debe morir con el Ctrl+C del terminal ni con su cierre.
#[cfg(unix)]
fn separar_del_terminal(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
}

#[cfg(windows)]
fn separar_del_terminal(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(any(unix, windows)))]
fn separar_del_terminal(_: &mut Command) {}

/// En Windows, `CreateProcess` pasa al hijo todos los handles heredables, no solo los de su stdio.
/// Si el servidor en segundo plano se quedara con la salida de `shiori run` (una tubería, cuando
/// la lee un script), quien la lee esperaría para siempre. Rust duplica como heredables los handles
/// que pasa a cada hijo, así que quitar el flag a los originales no cambia lo que ve el programa
/// lanzado. En Unix no hace falta: Rust abre todo con `O_CLOEXEC`.
#[cfg(windows)]
fn no_heredar_stdio_propio() {
    use windows_sys::Win32::Foundation::{
        HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, SetHandleInformation,
    };
    use windows_sys::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    for id in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
        // SAFETY: GetStdHandle y SetHandleInformation solo leen y cambian flags de un handle propio;
        // un handle nulo o inválido se salta y un fallo no tiene consecuencias.
        unsafe {
            let h = GetStdHandle(id);
            if !h.is_null() && h != INVALID_HANDLE_VALUE {
                SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0);
            }
        }
    }
}

#[cfg(not(windows))]
fn no_heredar_stdio_propio() {}

/// Cómo inició sesión Codex: con la cuenta de ChatGPT (suscripción) o con una API key.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AuthCodex {
    Suscripcion,
    ApiKey,
}

pub fn auth_codex(entorno: &dyn Fn(&str) -> Option<String>) -> AuthCodex {
    if entorno("OPENAI_API_KEY").is_some_and(|k| !k.is_empty()) {
        return AuthCodex::ApiKey;
    }
    let carpeta = entorno("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".codex")));
    let Some(carpeta) = carpeta else {
        return AuthCodex::ApiKey;
    };
    let Ok(texto) = std::fs::read_to_string(carpeta.join("auth.json")) else {
        return AuthCodex::ApiKey;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&texto) else {
        return AuthCodex::ApiKey;
    };
    if v["OPENAI_API_KEY"].as_str().is_some_and(|k| !k.is_empty()) {
        AuthCodex::ApiKey
    } else if v.get("tokens").is_some_and(|t| t.is_object()) {
        AuthCodex::Suscripcion
    } else {
        AuthCodex::ApiKey
    }
}

/// Variables de entorno que apuntan los clientes al proxy, con la carpeta como proyecto.
pub fn variables(config: &Config, carpeta: &Path, auth: AuthCodex) -> Vec<(String, String)> {
    let base = format!(
        "{}{}{}",
        config.url_local(),
        PREFIJO_CARPETA,
        texto::codificar(&carpeta.to_string_lossy())
    );
    let openai = match auth {
        AuthCodex::Suscripcion => format!("{base}/chatgpt/codex"),
        AuthCodex::ApiKey => format!("{base}/openai/v1"),
    };
    vec![
        ("ANTHROPIC_BASE_URL".into(), base.clone()),
        ("OPENAI_BASE_URL".into(), openai),
        ("SHIORI_URL".into(), config.url_local()),
    ]
}

/// En Windows, `claude` suele ser `claude.cmd` o `claude.exe`: se busca con PATHEXT.
fn resolver_programa(nombre: &str) -> PathBuf {
    #[cfg(windows)]
    {
        let p = Path::new(nombre);
        if p.extension().is_none() && p.components().count() == 1 {
            let extensiones =
                std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
            if let Some(path) = std::env::var_os("PATH") {
                for carpeta in std::env::split_paths(&path) {
                    for ext in extensiones.split(';').filter(|e| !e.is_empty()) {
                        let c = carpeta.join(format!("{nombre}{ext}"));
                        if c.is_file() {
                            return c;
                        }
                    }
                }
            }
        }
    }
    PathBuf::from(nombre)
}

pub async fn ejecutar(config: &Config, comando: &[String]) -> Result<i32, String> {
    let (programa, argumentos) = comando
        .split_first()
        .ok_or("say which program to run, for example: shiori run claude")?;
    asegurar_servidor(config).await?;
    let carpeta = std::env::current_dir().map_err(|e| e.to_string())?;
    let auth = auth_codex(&crate::configuracion::entorno_real);
    let vars = variables(config, &carpeta, auth);
    if let Ok(previa) = std::env::var("ANTHROPIC_BASE_URL")
        && !previa.starts_with(&config.url_local())
    {
        eprintln!(
            "shiori: ANTHROPIC_BASE_URL was {previa}; Shiori replaces it for this run \
             (to keep going there, set upstreams.anthropic)"
        );
    }
    let mut cmd = Command::new(resolver_programa(programa));
    cmd.args(argumentos).envs(vars);
    let mut hijo = cmd
        .spawn()
        .map_err(|e| format!("cannot run `{programa}`: {e}"))?;
    // El Ctrl+C del terminal le llega al programa; Shiori lo ignora y espera a que termine.
    let ignorar = tokio::spawn(async {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                break;
            }
        }
    });
    let estado = tokio::task::spawn_blocking(move || hijo.wait())
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    ignorar.abort();
    Ok(codigo_salida(estado))
}

#[cfg(unix)]
fn codigo_salida(e: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    e.code().unwrap_or_else(|| 128 + e.signal().unwrap_or(1))
}

#[cfg(not(unix))]
fn codigo_salida(e: std::process::ExitStatus) -> i32 {
    e.code().unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_de_codex() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().to_string_lossy().to_string();
        let e = |clave: Option<&'static str>| {
            let home = home.clone();
            move |n: &str| match n {
                "CODEX_HOME" => Some(home.clone()),
                "OPENAI_API_KEY" => clave.map(str::to_string),
                _ => None,
            }
        };
        assert_eq!(auth_codex(&e(None)), AuthCodex::ApiKey, "sin auth.json");
        std::fs::write(
            dir.path().join("auth.json"),
            r#"{"OPENAI_API_KEY":null,"tokens":{"id_token":"x"}}"#,
        )
        .unwrap();
        assert_eq!(auth_codex(&e(None)), AuthCodex::Suscripcion);
        assert_eq!(
            auth_codex(&e(Some("sk-1"))),
            AuthCodex::ApiKey,
            "la variable manda"
        );
        std::fs::write(dir.path().join("auth.json"), r#"{"OPENAI_API_KEY":"sk-2"}"#).unwrap();
        assert_eq!(auth_codex(&e(None)), AuthCodex::ApiKey);
    }

    #[test]
    fn variables_con_carpeta() {
        let c = Config::default();
        let v = variables(&c, Path::new("/home/m/mi repo"), AuthCodex::Suscripcion);
        assert_eq!(
            v[0],
            (
                "ANTHROPIC_BASE_URL".to_string(),
                "http://127.0.0.1:8787/_shiori/dir/%2Fhome%2Fm%2Fmi%20repo".to_string()
            )
        );
        assert!(
            v[1].1
                .ends_with("/_shiori/dir/%2Fhome%2Fm%2Fmi%20repo/chatgpt/codex")
        );
    }
}
