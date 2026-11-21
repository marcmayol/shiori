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
