//! El binario tal como lo usa una persona: validar la configuración, arrancar, ver el estado, parar y
//! lanzar un programa a través del proxy.

use std::path::Path;
use std::process::{Command, Output};

const SHIORI: &str = env!("CARGO_BIN_EXE_shiori");

/// Un `shiori` aislado: su configuración, sus datos y un puerto libre.
fn shiori(datos: &Path, puerto: u16, args: &[&str]) -> Output {
    Command::new(SHIORI)
        .args(args)
        .env("SHIORI_DATA_DIR", datos)
        .env("SHIORI_METRICS_PATH", datos.join("metrics.db"))
        .env("SHIORI_CONFIG", "")
        .env("SHIORI_PORT", puerto.to_string())
        .env_remove("SHIORI_HOST")
        .env("SHIORI_LOG", "warn")
        .output()
        .unwrap()
}

fn puerto_libre() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn texto(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

#[test]
fn validar_dice_la_linea() {
    let dir = tempfile::tempdir().unwrap();
    let malo = dir.path().join("config.yaml");
    std::fs::write(&malo, "server:\n  port: 8787\n  hots: x\n").unwrap();
    let o = shiori(
        dir.path(),
        1,
        &["config", "validate", malo.to_str().unwrap()],
    );
    assert!(!o.status.success());
    let t = texto(&o);
    assert!(t.contains("config.yaml:3"), "{t}");
    assert!(t.contains("hots"), "{t}");

    let bueno = dir.path().join("shiori.toml");
    std::fs::write(
        &bueno,
        "project = \"x\"\n\n[layers.exact_cache]\nenabled = false\n",
    )
    .unwrap();
    let o = shiori(
        dir.path(),
        1,
        &["config", "validate", bueno.to_str().unwrap()],
    );
    assert!(o.status.success(), "{}", texto(&o));
    assert!(texto(&o).contains("OK"));

    // El mismo archivo con una clave solo del global, como archivo de proyecto, es error.
    std::fs::write(&bueno, "project = \"x\"\n[server]\nport = 1\n").unwrap();
    let o = shiori(
        dir.path(),
        1,
        &["config", "validate", bueno.to_str().unwrap()],
    );
    assert!(!o.status.success());
    assert!(texto(&o).contains("shiori.toml:2"), "{}", texto(&o));
}

#[test]
fn esquema_y_config_en_uso() {
    let dir = tempfile::tempdir().unwrap();
    let o = shiori(dir.path(), 9999, &["config", "schema", "--project"]);
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert!(v["properties"]["layers"].is_object());
    assert!(v["properties"]["server"].is_null());

    let o = shiori(dir.path(), 9999, &["config", "show"]);
    assert!(o.status.success());
    assert!(texto(&o).contains("port: 9999"), "{}", texto(&o));
}

#[test]
fn metricas_vacias() {
    let dir = tempfile::tempdir().unwrap();
    let o = shiori(dir.path(), 1, &["metrics"]);
    assert!(o.status.success());
    assert!(texto(&o).contains("No calls recorded yet"));
    let o = shiori(dir.path(), 1, &["metrics", "--since", "ayer"]);
    assert!(!o.status.success());
}

#[test]
fn run_arranca_en_segundo_plano_y_stop_lo_para() {
    let dir = tempfile::tempdir().unwrap();
    let puerto = puerto_libre();
    let o = shiori(dir.path(), puerto, &["status"]);
    assert!(texto(&o).contains("not running"), "{}", texto(&o));

    // El programa lanzado ve ANTHROPIC_BASE_URL y OPENAI_BASE_URL apuntando a Shiori.
    #[cfg(unix)]
    let programa: &[&str] = &[
        "sh",
        "-c",
        "echo \"A=$ANTHROPIC_BASE_URL\"; echo \"O=$OPENAI_BASE_URL\"; exit 3",
    ];
    #[cfg(windows)]
    let programa: &[&str] = &[
        "cmd",
        "/C",
        "echo A=%ANTHROPIC_BASE_URL%& echo O=%OPENAI_BASE_URL%& exit 3",
    ];
    let mut args = vec!["run", "--"];
    args.extend_from_slice(programa);
    let o = shiori(dir.path(), puerto, &args);
    let t = texto(&o);
    assert_eq!(
        o.status.code(),
        Some(3),
        "devuelve el código del programa: {t}"
    );
    let base = format!("http://127.0.0.1:{puerto}/_shiori/dir/");
    assert!(t.contains(&format!("A={base}")), "{t}");
    assert!(t.contains("O=http://127.0.0.1"), "{t}");

    // El servidor sigue en marcha después de que el programa termine.
    let o = shiori(dir.path(), puerto, &["status"]);
    assert!(texto(&o).contains("is running"), "{}", texto(&o));

    let o = shiori(dir.path(), puerto, &["stop"]);
    assert!(texto(&o).contains("Shiori stopped"), "{}", texto(&o));
    let o = shiori(dir.path(), puerto, &["status"]);
    assert!(texto(&o).contains("not running"), "{}", texto(&o));
    borrar_cuando_termine(dir);
}

/// En Windows `stop` vuelve cuando el puerto ya no escucha, pero el proceso tarda un poco más en
/// soltar la base y el log; sin esperar, la carpeta se quedaría en %TEMP%.
fn borrar_cuando_termine(dir: tempfile::TempDir) {
    for _ in 0..100 {
        match std::fs::remove_dir_all(dir.path()) {
            Ok(()) => return,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
    panic!(
        "{} sigue en uso 10 s después de parar",
        dir.path().display()
    );
}

#[test]
fn serve_avisa_si_el_puerto_esta_ocupado() {
    let dir = tempfile::tempdir().unwrap();
    let ocupado = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let puerto = ocupado.local_addr().unwrap().port();
    let o = shiori(dir.path(), puerto, &["serve"]);
    assert!(!o.status.success());
    assert!(texto(&o).contains("already in use"), "{}", texto(&o));
}

#[test]
fn bench_rechaza_cero() {
    let dir = tempfile::tempdir().unwrap();
    for args in [
        ["bench", "--requests", "0"],
        ["bench", "--concurrency", "0"],
    ] {
        let o = shiori(dir.path(), puerto_libre(), &args);
        assert_eq!(o.status.code(), Some(2), "{}", texto(&o));
        assert!(texto(&o).contains("must be at least 1"), "{}", texto(&o));
    }
}

/// Espera a que `serve` responda en su puerto.
fn esperar_salud(puerto: u16) {
    const SALUD: &[u8] = b"GET /_shiori/health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
    for _ in 0..100 {
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", puerto)) {
            use std::io::{Read, Write};
            let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let _ = s.write_all(SALUD);
            let mut r = String::new();
            let _ = s.read_to_string(&mut r);
            if r.contains("\"shiori\"") {
                return;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("shiori serve no arrancó en el puerto {puerto}");
}

/// Una petición HTTP/1.1 a mano: devuelve la respuesta entera.
fn post(puerto: u16, ruta: &str, cuerpo: &str) -> String {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(("127.0.0.1", puerto)).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(20)))
        .unwrap();
    write!(
        s,
        "POST {ruta} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n\
         x-api-key: sk-ant-api-prueba\r\nContent-Length: {}\r\n\r\n{cuerpo}",
        cuerpo.len()
    )
    .unwrap();
    let mut r = String::new();
    let _ = s.read_to_string(&mut r);
    r
}

/// Un proceso hijo que se para (y se espera) aunque el test falle.
struct Hijo(std::process::Child);

impl Drop for Hijo {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn serve(datos: &Path, puerto: u16, entorno: &[(&str, String)]) -> Hijo {
    let mut c = Command::new(SHIORI);
    c.arg("serve")
        .env("SHIORI_DATA_DIR", datos)
        .env("SHIORI_METRICS_PATH", datos.join("metrics.db"))
        .env("SHIORI_CONFIG", "")
        .env("SHIORI_PORT", puerto.to_string())
        .env_remove("SHIORI_HOST")
        .env("SHIORI_LOG", "warn")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    for (k, v) in entorno {
        c.env(k, v);
    }
    Hijo(c.spawn().unwrap())
}

#[test]
fn proxy_del_sistema_no_se_usa_para_destinos_locales() {
    let dir = tempfile::tempdir().unwrap();
    let puerto_falso = puerto_libre();
    let _falso = Hijo(
        Command::new(SHIORI)
            .args(["fake", "--port", &puerto_falso.to_string()])
            .env("SHIORI_LOG", "warn")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    // Un proxy del sistema en un puerto donde no escucha nadie: si se usara, la llamada daría 502.
    let cerrado = format!("http://127.0.0.1:{}", puerto_libre());
    let puerto = puerto_libre();
    let falso_url = format!("http://127.0.0.1:{puerto_falso}");
    let mut entorno = vec![("SHIORI_UPSTREAM_ANTHROPIC", falso_url)];
    for k in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        entorno.push((k, cerrado.clone()));
    }
    let _serve = serve(dir.path(), puerto, &entorno);
    esperar_salud(puerto);
    // Que el falso también esté escuchando.
    for _ in 0..100 {
        if std::net::TcpStream::connect(("127.0.0.1", puerto_falso)).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let r = post(
        puerto,
        "/v1/messages",
        r#"{"model":"claude-haiku-5-5","max_tokens":10,"messages":[]}"#,
    );
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.contains("msg_fake"), "{r}");
}

#[cfg(unix)]
#[test]
fn sigterm_para_en_orden() {
    let dir = tempfile::tempdir().unwrap();
    let puerto = puerto_libre();
    let mut s = serve(dir.path(), puerto, &[]);
    esperar_salud(puerto);
    let clave = dir.path().join(format!("server-{puerto}.key"));
    assert!(clave.exists());
    let k = Command::new("kill")
        .args(["-TERM", &s.0.id().to_string()])
        .status()
        .unwrap();
    assert!(k.success());
    let mut estado = None;
    for _ in 0..100 {
        if let Some(e) = s.0.try_wait().unwrap() {
            estado = Some(e);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let estado = estado.expect("serve no paró con SIGTERM");
    assert_eq!(estado.code(), Some(0), "{estado:?}");
    assert!(!clave.exists(), "la clave de parada tiene que borrarse");
}

/// La primera señal deja acabar las llamadas en curso; con un streaming largo, la segunda tiene que
/// parar al momento (si no, solo queda `kill -9`).
#[cfg(unix)]
#[test]
fn segunda_sigterm_para_al_momento() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let puerto_falso = puerto_libre();
    // 100 tokens cada 150 ms: unos 15 s de streaming.
    let _falso = Hijo(
        Command::new(SHIORI)
            .args(["fake", "--port", &puerto_falso.to_string()])
            .args(["--tokens", "100", "--token-delay-ms", "150"])
            .env("SHIORI_LOG", "warn")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let falso_listo = (0..100).any(|_| {
        let ok = std::net::TcpStream::connect(("127.0.0.1", puerto_falso)).is_ok();
        if !ok {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        ok
    });
    assert!(falso_listo, "shiori fake no arrancó");
    let puerto = puerto_libre();
    let entorno = [(
        "SHIORI_UPSTREAM_ANTHROPIC",
        format!("http://127.0.0.1:{puerto_falso}"),
    )];
    let mut s = serve(dir.path(), puerto, &entorno);
    esperar_salud(puerto);
    let clave = dir.path().join(format!("server-{puerto}.key"));
    assert!(clave.exists());
    // La señal va cuando el streaming ya ha empezado: si no, el cierre en orden acabaría solo.
    let mut llamada = std::net::TcpStream::connect(("127.0.0.1", puerto)).unwrap();
    llamada
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let cuerpo = r#"{"model":"claude-haiku-5-5","max_tokens":100,"stream":true,"messages":[]}"#;
    write!(
        llamada,
        "POST /v1/messages HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n\
         x-api-key: sk-ant-api-prueba\r\nContent-Length: {}\r\n\r\n{cuerpo}",
        cuerpo.len()
    )
    .unwrap();
    let mut trozo = [0u8; 64];
    let n = llamada.read(&mut trozo).unwrap();
    assert!(
        trozo[..n].starts_with(b"HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&trozo[..n])
    );
    let matar = |s: &Hijo| {
        let k = Command::new("kill")
            .args(["-TERM", &s.0.id().to_string()])
            .status()
            .unwrap();
        assert!(k.success());
    };
    matar(&s);
    // La primera señal hace dejar de escuchar (el apagado empezó), pero no mata el proceso.
    let cerrado = (0..30).any(|_| {
        let libre = std::net::TcpStream::connect(("127.0.0.1", puerto)).is_err();
        if !libre {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        libre
    });
    assert!(
        cerrado,
        "la primera SIGTERM no hizo dejar de escuchar a serve"
    );
    assert!(
        s.0.try_wait().unwrap().is_none(),
        "la primera SIGTERM tiene que esperar al streaming en curso"
    );
    matar(&s);
    let mut estado = None;
    for _ in 0..30 {
        if let Some(e) = s.0.try_wait().unwrap() {
            estado = Some(e);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // Al streaming le quedan más de 10 s: salir antes de 3 s solo puede ser la salida forzada.
    let estado = estado.expect("la segunda SIGTERM no paró serve en 3 s");
    assert_eq!(estado.code(), Some(1), "{estado:?}");
    assert!(!clave.exists(), "la clave de parada tiene que borrarse");
    drop(llamada);
}

#[cfg(windows)]
#[test]
fn un_solo_serve_por_puerto_en_windows() {
    use std::os::windows::fs::OpenOptionsExt;
    let dir = tempfile::tempdir().unwrap();
    let puerto = puerto_libre();
    // Otro `serve` del mismo puerto tiene el bloqueo abierto.
    let _bloqueo = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .share_mode(0)
        .open(dir.path().join(format!("server-{puerto}.lock")))
        .unwrap();
    // Con `output()`, un `serve` que arrancara colgaría el test: se lanza y se espera con plazo.
    let mut c = Command::new(SHIORI);
    c.arg("serve")
        .env("SHIORI_DATA_DIR", dir.path())
        .env("SHIORI_METRICS_PATH", dir.path().join("metrics.db"))
        .env("SHIORI_CONFIG", "")
        .env("SHIORI_PORT", puerto.to_string())
        .env_remove("SHIORI_HOST")
        .env("SHIORI_LOG", "warn")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut s = Hijo(c.spawn().unwrap());
    let mut estado = None;
    for _ in 0..100 {
        if let Some(e) = s.0.try_wait().unwrap() {
            estado = Some(e);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let estado = estado.expect("el segundo serve arrancó en vez de negarse");
    assert!(!estado.success());
    let mut error = String::new();
    std::io::Read::read_to_string(s.0.stderr.as_mut().unwrap(), &mut error).unwrap();
    assert!(
        error.contains("another Shiori is already running"),
        "{error}"
    );
    assert!(
        !dir.path().join(format!("server-{puerto}.key")).exists(),
        "no tiene que pisar la clave de parada del otro"
    );
}

/// Un streaming largo no hace fallar `stop`: el servidor deja de escuchar enseguida y termina la
/// llamada antes de salir. En Windows el bloqueo sigue cogido hasta entonces y `stop` lo dice.
#[test]
fn stop_con_streaming_en_curso_sale_bien() {
    let dir = tempfile::tempdir().unwrap();
    let puerto_falso = puerto_libre();
    // 40 tokens cada 300 ms: unos 12 s de streaming, más que los 10 s que espera `stop`.
    let _falso = Hijo(
        Command::new(SHIORI)
            .args(["fake", "--port", &puerto_falso.to_string()])
            .args(["--tokens", "40", "--token-delay-ms", "300"])
            .env("SHIORI_LOG", "warn")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    for _ in 0..100 {
        if std::net::TcpStream::connect(("127.0.0.1", puerto_falso)).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let puerto = puerto_libre();
    let falso_url = format!("http://127.0.0.1:{puerto_falso}");
    let mut s = serve(
        dir.path(),
        puerto,
        &[("SHIORI_UPSTREAM_ANTHROPIC", falso_url)],
    );
    esperar_salud(puerto);
    let llamada = std::thread::spawn(move || {
        post(
            puerto,
            "/v1/messages",
            r#"{"model":"claude-haiku-5-5","max_tokens":50,"stream":true,"messages":[]}"#,
        )
    });
    // Que la llamada ya esté en curso al parar.
    std::thread::sleep(std::time::Duration::from_secs(1));
    let o = shiori(dir.path(), puerto, &["stop"]);
    assert!(o.status.success(), "{}", texto(&o));
    if cfg!(windows) {
        assert!(
            texto(&o).contains("Shiori stopped listening"),
            "{}",
            texto(&o)
        );
        let st = shiori(dir.path(), puerto, &["status"]);
        assert!(texto(&st).contains("Shiori is stopping"), "{}", texto(&st));
    } else {
        assert!(texto(&o).contains("Shiori stopped"), "{}", texto(&o));
    }
    let r = llamada.join().unwrap();
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(
        r.contains("message_stop"),
        "el streaming tiene que acabar entero: {r}"
    );
    let mut estado = None;
    for _ in 0..100 {
        if let Some(e) = s.0.try_wait().unwrap() {
            estado = Some(e);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let estado = estado.expect("serve no salió al acabar la llamada");
    assert!(estado.success(), "{estado:?}");
}

/// En Unix nada impide un `serve` nuevo mientras el anterior termina sus llamadas tras un `stop`.
/// Al salir, el viejo no puede borrar la clave de parada del nuevo: si no, `stop` ya no lo para.
#[cfg(unix)]
#[test]
fn el_serve_viejo_no_borra_la_clave_del_nuevo() {
    let dir = tempfile::tempdir().unwrap();
    let puerto_falso = puerto_libre();
    // 20 tokens cada 150 ms: unos 3 s de streaming.
    let _falso = Hijo(
        Command::new(SHIORI)
            .args(["fake", "--port", &puerto_falso.to_string()])
            .args(["--tokens", "20", "--token-delay-ms", "150"])
            .env("SHIORI_LOG", "warn")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    for _ in 0..100 {
        if std::net::TcpStream::connect(("127.0.0.1", puerto_falso)).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let puerto = puerto_libre();
    let falso_url = format!("http://127.0.0.1:{puerto_falso}");
    let entorno = [("SHIORI_UPSTREAM_ANTHROPIC", falso_url)];
    let mut viejo = serve(dir.path(), puerto, &entorno);
    esperar_salud(puerto);
    let llamada = std::thread::spawn(move || {
        post(
            puerto,
            "/v1/messages",
            r#"{"model":"claude-haiku-5-5","max_tokens":50,"stream":true,"messages":[]}"#,
        )
    });
    std::thread::sleep(std::time::Duration::from_secs(1));
    let o = shiori(dir.path(), puerto, &["stop"]);
    assert!(o.status.success(), "{}", texto(&o));
    let mut nuevo = serve(dir.path(), puerto, &entorno);
    esperar_salud(puerto);
    // Si no, el caso no se da: el viejo ya habría salido borrando su propia clave.
    assert!(
        viejo.0.try_wait().unwrap().is_none(),
        "el serve viejo acabó antes de arrancar el nuevo"
    );
    assert!(llamada.join().unwrap().starts_with("HTTP/1.1 200"));
    let salida = |h: &mut Hijo| {
        for _ in 0..100 {
            if let Some(e) = h.0.try_wait().unwrap() {
                return Some(e);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        None
    };
    assert!(
        salida(&mut viejo)
            .expect("el serve viejo no salió")
            .success()
    );
    assert!(dir.path().join(format!("server-{puerto}.key")).exists());
    let o = shiori(dir.path(), puerto, &["stop"]);
    assert!(o.status.success(), "{}", texto(&o));
    assert!(
        salida(&mut nuevo)
            .expect("stop no paró el serve nuevo")
            .success()
    );
}

#[test]
fn bench_no_deja_la_base_en_la_carpeta_temporal() {
    let dir = tempfile::tempdir().unwrap();
    let temporal = dir.path().join("tmp");
    std::fs::create_dir(&temporal).unwrap();
    let o = Command::new(SHIORI)
        .args(["bench", "--requests", "4", "--concurrency", "2"])
        .args(["--tokens", "2", "--token-delay-ms", "0"])
        .env("TEMP", &temporal)
        .env("TMP", &temporal)
        .env("TMPDIR", &temporal)
        .env("SHIORI_LOG", "warn")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", texto(&o));
    assert!(!texto(&o).contains("cannot delete"), "{}", texto(&o));
    let quedan: Vec<_> = std::fs::read_dir(&temporal)
        .unwrap()
        .map(|f| f.unwrap().file_name())
        .collect();
    assert!(quedan.is_empty(), "{quedan:?}");
}
