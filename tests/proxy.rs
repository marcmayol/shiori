//! El proxy pasa las peticiones tal cual, en JSON y en streaming, para los tres proveedores y con
//! suscripción, y mide el uso sin tocar nada.

mod comun;

use std::time::{Duration, Instant};

use comun::{cuerpo_anthropic, entorno, entorno_con};
use futures_util::StreamExt;
use serde_json::{Value, json};
use shiori::dominio::autenticacion::Autenticacion;
use shiori::dominio::proveedor::Proveedor;
use shiori::dominio::texto;
use shiori::falso::{CACHE_ESCRITA, CACHE_LEIDA, ENTRADA, OpcionesFalso};

#[tokio::test(flavor = "multi_thread")]
async fn json_identico_a_llamar_directo() {
    let e = entorno(OpcionesFalso::default()).await;
    let directo = e
        .cliente
        .post(e.directo("/v1/messages"))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap();
    let estado_d = directo.status();
    let tipo_d = directo.headers()["content-type"].clone();
    let cuerpo_d = directo.bytes().await.unwrap();

    let via = e
        .cliente
        .post(e.url("/v1/messages"))
        .header("x-api-key", "sk-ant-api03-test")
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap();
    assert_eq!(via.status(), estado_d);
    assert_eq!(via.headers()["content-type"], tipo_d);
    assert_eq!(
        via.bytes().await.unwrap(),
        cuerpo_d,
        "el cuerpo llega byte a byte"
    );

    let l = &e.llamadas(1).await[0];
    assert_eq!(l.proveedor, Proveedor::Anthropic);
    assert_eq!(l.autenticacion, Autenticacion::ApiKey);
    assert_eq!(l.uso.modelo.as_deref(), Some("claude-opus-5-5"));
    assert_eq!(
        (
            l.uso.entrada,
            l.uso.salida,
            l.uso.cache_leida,
            l.uso.cache_escrita
        ),
        (ENTRADA, 20, CACHE_LEIDA, CACHE_ESCRITA)
    );
    assert!(!l.stream);
    assert!(!l.cancelada);
    assert_eq!(l.proyecto, "default");
    assert!(l.coste_usd.unwrap() > 0.0);
    assert!(l.ahorro_usd.unwrap() > 0.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_identico_y_al_vuelo() {
    let e = entorno(OpcionesFalso {
        tokens: 10,
        retardo_token_ms: 40,
        retardo_inicial_ms: 0,
    })
    .await;
    let directo = e
        .cliente
        .post(e.directo("/v1/messages"))
        .json(&cuerpo_anthropic(true))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let inicio = Instant::now();
    let r = e
        .cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo_anthropic(true))
        .send()
        .await
        .unwrap();
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let mut flujo = r.bytes_stream();
    let mut recibido = Vec::new();
    let mut primer_trozo = None;
    while let Some(t) = flujo.next().await {
        if primer_trozo.is_none() {
            primer_trozo = Some(inicio.elapsed());
        }
        recibido.extend_from_slice(&t.unwrap());
    }
    let total = inicio.elapsed();
    assert_eq!(
        recibido,
        directo.to_vec(),
        "mismos eventos, en el mismo orden"
    );
    // 15 trozos con 40 ms entre ellos: el primero tiene que llegar mucho antes que el final.
    assert!(
        primer_trozo.unwrap() < Duration::from_millis(200) && total > Duration::from_millis(400),
        "primer trozo {primer_trozo:?}, total {total:?}: el proxy no está reenviando al vuelo"
    );

    let l = &e.llamadas(1).await[0];
    assert!(l.stream);
    assert_eq!((l.uso.entrada, l.uso.salida), (ENTRADA, 10));
    assert!(l.primer_byte_ms.unwrap() < l.latencia_ms);
}

#[tokio::test(flavor = "multi_thread")]
async fn suscripcion_de_claude_code_pasa_intacta() {
    let e = entorno(OpcionesFalso::default()).await;
    let r = e
        .cliente
        .post(e.url("/v1/messages?beta=true"))
        .header("authorization", "Bearer sk-ant-oat01-SECRETO")
        .header("anthropic-beta", "oauth-2025-04-20,claude-code-20250219")
        .header("anthropic-version", "2023-06-01")
        .header("user-agent", "claude-cli/9.9.9 (external, cli)")
        .header("x-app", "cli")
        .header("x-shiori-project", "mi-proyecto")
        .header("accept-encoding", "gzip, br")
        .json(&cuerpo_anthropic(true))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    r.bytes().await.unwrap();

    let vista = e.falso.ultima().unwrap();
    assert_eq!(vista.ruta, "/v1/messages?beta=true", "la query se conserva");
    let h = &vista.cabeceras;
    assert_eq!(h["authorization"], "Bearer sk-ant-oat01-SECRETO");
    assert_eq!(h["anthropic-beta"], "oauth-2025-04-20,claude-code-20250219");
    assert_eq!(h["anthropic-version"], "2023-06-01");
    assert_eq!(h["user-agent"], "claude-cli/9.9.9 (external, cli)");
    assert_eq!(h["x-app"], "cli");
    assert_eq!(h["accept-encoding"], "identity");
    assert_eq!(h["host"], e.falso.direccion.to_string());
    assert!(h.keys().all(|k| !k.as_str().starts_with("x-shiori")));

    let l = &e.llamadas(1).await[0];
    assert_eq!(l.autenticacion, Autenticacion::Subscription);
    assert_eq!(l.proyecto, "mi-proyecto");
    assert_eq!(l.ruta, "/v1/messages");
}

#[tokio::test(flavor = "multi_thread")]
async fn errores_del_proveedor_pasan_intactos() {
    let e = entorno(OpcionesFalso::default()).await;
    let pedir = |url: String| {
        e.cliente
            .post(url)
            .header("x-fake-status", "429")
            .json(&cuerpo_anthropic(false))
            .send()
    };
    let d = pedir(e.directo("/v1/messages")).await.unwrap();
    let cuerpo_d = d.bytes().await.unwrap();
    let r = pedir(e.url("/v1/messages")).await.unwrap();
    assert_eq!(r.status(), 429);
    assert_eq!(r.headers()["retry-after"], "7");
    assert!(
        r.headers().get("x-shiori-error").is_none(),
        "no es un error de Shiori"
    );
    assert_eq!(r.bytes().await.unwrap(), cuerpo_d);
    let l = &e.llamadas(1).await[0];
    assert_eq!(l.estado, 429);
    assert!(l.uso.vacio());
}

#[tokio::test(flavor = "multi_thread")]
async fn api_caida_da_502_con_el_formato_de_cada_proveedor() {
    // Un puerto donde no escucha nadie.
    let libre = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let muerto = format!("http://{}", libre.local_addr().unwrap());
    drop(libre);
    let e = entorno_con(OpcionesFalso::default(), |c| {
        for p in Proveedor::TODOS {
            c.destinos.insert(p, muerto.clone());
        }
    })
    .await;

    let r = e
        .cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    assert_eq!(r.headers()["x-shiori-error"], "1");
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "api_error");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot reach anthropic")
    );

    let v: Value = e
        .cliente
        .post(e.url("/openai/v1/chat/completions"))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["error"]["type"], "shiori_error");

    let v: Value = e
        .cliente
        .post(e.url("/gemini/v1beta/models/g:generateContent"))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["error"]["status"], "UNAVAILABLE");

    let ls = e.llamadas(3).await;
    assert!(ls.iter().all(|l| l.estado == 502));
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_con_api_key() {
    let e = entorno(OpcionesFalso::default()).await;
    let r = e
        .cliente
        .post(e.url("/openai/v1/chat/completions"))
        .header("authorization", "Bearer sk-proj-abc")
        .json(&json!({"model": "gpt-5", "messages": [], "stream": true,
                      "stream_options": {"include_usage": true}}))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    let texto = r.text().await.unwrap();
    assert!(texto.ends_with("data: [DONE]\n\n"));
    let vista = e.falso.ultima().unwrap();
    assert_eq!(vista.ruta, "/v1/chat/completions");
    assert_eq!(vista.cabeceras["authorization"], "Bearer sk-proj-abc");

    let l = &e.llamadas(1).await[0];
    assert_eq!(l.proveedor, Proveedor::Openai);
    assert_eq!(l.autenticacion, Autenticacion::ApiKey);
    assert_eq!(l.uso.modelo.as_deref(), Some("gpt-5"));
    assert_eq!(
        (l.uso.entrada, l.uso.salida, l.uso.cache_leida),
        (ENTRADA, 20, CACHE_LEIDA)
    );
    assert!(l.coste_usd.is_none(), "gpt-5 no está en la tabla integrada");
}

#[tokio::test(flavor = "multi_thread")]
async fn codex_con_suscripcion_de_chatgpt() {
    let e = entorno_con(OpcionesFalso::default(), |c| {
        c.precios.insert(
            "gpt-5".into(),
            shiori::dominio::precios::Precio {
                input: 1.25,
                output: 10.0,
                cache_read: None,
                cache_write: None,
            },
        );
    })
    .await;
    let r = e
        .cliente
        .post(e.url("/chatgpt/codex/responses"))
        .header("authorization", "Bearer eyJhbGciOiJSUzI1NiJ9.x.y")
        .header("chatgpt-account-id", "cuenta-123")
        .header("originator", "codex_cli_rs")
        .json(&json!({"model": "gpt-5-codex", "input": [], "stream": true}))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    r.bytes().await.unwrap();
    let vista = e.falso.ultima().unwrap();
    assert_eq!(vista.ruta, "/codex/responses");
    assert_eq!(vista.cabeceras["chatgpt-account-id"], "cuenta-123");
    assert_eq!(vista.cabeceras["originator"], "codex_cli_rs");
    assert_eq!(
        vista.cabeceras["authorization"],
        "Bearer eyJhbGciOiJSUzI1NiJ9.x.y"
    );

    let l = &e.llamadas(1).await[0];
    assert_eq!(l.proveedor, Proveedor::Chatgpt);
    assert_eq!(l.autenticacion, Autenticacion::Subscription);
    assert_eq!(l.uso.modelo.as_deref(), Some("gpt-5-codex"));
    assert_eq!(
        (l.uso.entrada, l.uso.salida, l.uso.cache_leida),
        (ENTRADA, 20, CACHE_LEIDA)
    );
    assert!(l.coste_usd.unwrap() > 0.0, "precio propio por prefijo");
}

#[tokio::test(flavor = "multi_thread")]
async fn gemini_sse_y_array() {
    let e = entorno(OpcionesFalso::default()).await;
    let r = e
        .cliente
        .post(e.url("/gemini/v1beta/models/gemini-x:streamGenerateContent?alt=sse&key=CLAVE"))
        .json(&json!({"contents": []}))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    r.bytes().await.unwrap();
    assert_eq!(
        e.falso.ultima().unwrap().ruta,
        "/v1beta/models/gemini-x:streamGenerateContent?alt=sse&key=CLAVE"
    );

    let r = e
        .cliente
        .post(e.url("/gemini/v1beta/models/gemini-y:streamGenerateContent"))
        .header("x-goog-api-key", "CLAVE")
        .json(&json!({"contents": []}))
        .send()
        .await
        .unwrap();
    let v: Value = r.json().await.unwrap();
    assert!(v.is_array());

    let ls = e.llamadas(2).await;
    for (l, modelo) in ls.iter().zip(["gemini-x", "gemini-y"]) {
        assert_eq!(l.proveedor, Proveedor::Gemini);
        assert_eq!(l.autenticacion, Autenticacion::ApiKey);
        assert_eq!(l.uso.modelo.as_deref(), Some(modelo));
        assert_eq!(
            (l.uso.entrada, l.uso.salida, l.uso.cache_leida),
            (ENTRADA, 20, CACHE_LEIDA)
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn get_sin_cuerpo() {
    let e = entorno(OpcionesFalso::default()).await;
    let v: Value = e
        .cliente
        .get(e.url("/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["data"][0]["id"], "fake-model");
    assert_eq!(e.falso.ultima().unwrap().metodo, "GET");
}

#[tokio::test(flavor = "multi_thread")]
async fn proyecto_por_carpeta_y_archivo() {
    let e = entorno(OpcionesFalso::default()).await;
    let repo = e.dir.path().join("mi repo ñ");
    std::fs::create_dir_all(&repo).unwrap();
    let cod = texto::codificar(&repo.to_string_lossy());

    // Sin archivo: el nombre de la carpeta.
    let r = e
        .cliente
        .post(e.url(&format!("/_shiori/dir/{cod}/v1/messages")))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    r.bytes().await.unwrap();
    assert_eq!(e.falso.ultima().unwrap().ruta, "/v1/messages");

    // Con archivo: el nombre del archivo. Por cabecera y por ruta con prefijo de proveedor.
    std::fs::write(repo.join("shiori.yaml"), "project: nombre-propio\n").unwrap();
    e.cliente
        .post(e.url("/v1/messages"))
        .header("x-shiori-dir", &cod)
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    e.cliente
        .post(e.url(&format!("/_shiori/dir/{cod}/openai/v1/chat/completions")))
        .json(&json!({"model": "gpt-5"}))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(e.falso.ultima().unwrap().ruta, "/v1/chat/completions");

    let ls = e.llamadas(3).await;
    assert_eq!(ls[0].proyecto, "mi repo ñ");
    assert_eq!(ls[1].proyecto, "nombre-propio");
    assert_eq!(ls[2].proyecto, "nombre-propio");
    assert_eq!(ls[2].proveedor, Proveedor::Openai);

    // Un archivo de proyecto que intenta cambiar el destino se rechaza con su línea, sin llamar a la API.
    let vistas_antes = e.falso.vistas().len();
    std::fs::write(
        repo.join("shiori.yaml"),
        "project: x\nupstreams:\n  anthropic: http://evil\n",
    )
    .unwrap();
    let r = e
        .cliente
        .post(e.url(&format!("/_shiori/dir/{cod}/v1/messages")))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    let m = v["error"]["message"].as_str().unwrap();
    assert!(m.contains("shiori.yaml:2"), "{m}");
    assert!(m.contains("upstreams"), "{m}");
    assert_eq!(e.falso.vistas().len(), vistas_antes);
}

#[tokio::test(flavor = "multi_thread")]
async fn cliente_que_corta_queda_como_cancelada() {
    let e = entorno(OpcionesFalso {
        tokens: 50,
        retardo_token_ms: 30,
        retardo_inicial_ms: 0,
    })
    .await;
    let r = e
        .cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo_anthropic(true))
        .send()
        .await
        .unwrap();
    let mut flujo = r.bytes_stream();
    flujo.next().await.unwrap().unwrap();
    drop(flujo);
    let l = &e.llamadas(1).await[0];
    assert!(l.cancelada);
    assert_eq!(l.uso.entrada, ENTRADA, "lo leído antes de cortar cuenta");
}

#[tokio::test(flavor = "multi_thread")]
async fn cuerpos_grandes_y_limite() {
    let e = entorno_con(OpcionesFalso::default(), |c| {
        c.max_cuerpo_bytes = 4 * 1024 * 1024
    })
    .await;
    let grande = "x".repeat(3 * 1024 * 1024);
    let cuerpo =
        json!({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": grande}]});
    let r = e
        .cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success());
    r.bytes().await.unwrap();
    assert_eq!(
        e.falso.ultima().unwrap().cuerpo.len(),
        serde_json::to_vec(&cuerpo).unwrap().len()
    );

    let demasiado = json!({"content": "x".repeat(5 * 1024 * 1024)});
    let r = e
        .cliente
        .post(e.url("/v1/messages"))
        .json(&demasiado)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 413);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["type"], "request_too_large");
}

#[tokio::test(flavor = "multi_thread")]
async fn muchas_a_la_vez() {
    let e = entorno(OpcionesFalso {
        tokens: 30,
        retardo_token_ms: 2,
        retardo_inicial_ms: 0,
    })
    .await;
    let esperado = e
        .cliente
        .post(e.directo("/v1/messages"))
        .json(&cuerpo_anthropic(true))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let tareas: Vec<_> = (0..64)
        .map(|_| {
            let c = e.cliente.clone();
            let url = e.url("/v1/messages");
            tokio::spawn(async move {
                c.post(url)
                    .json(&cuerpo_anthropic(true))
                    .send()
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap()
            })
        })
        .collect();
    for t in tareas {
        assert_eq!(t.await.unwrap(), esperado);
    }
    let ls = e.llamadas(64).await;
    assert!(ls.iter().all(|l| l.uso.salida == 30 && !l.cancelada));
}

#[tokio::test(flavor = "multi_thread")]
async fn salud_y_parada_con_clave() {
    let e = entorno(OpcionesFalso::default()).await;
    let v: Value = e
        .cliente
        .get(e.url("/_shiori/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["name"], "shiori");
    assert_eq!(v["version"], shiori::VERSION);

    let r = e
        .cliente
        .post(e.url("/_shiori/stop"))
        .header("x-shiori-stop-key", "mala")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let r = e
        .cliente
        .post(e.url("/_shiori/stop"))
        .header("x-shiori-stop-key", "clave-test")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    tokio::time::timeout(Duration::from_secs(5), e.proxy.esperar())
        .await
        .expect("el servidor tiene que parar")
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn sin_metricas_no_crea_archivo() {
    let e = entorno_con(OpcionesFalso::default(), |c| c.metricas = false).await;
    e.cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    e.proxy.vaciar_metricas();
    assert!(!e.metricas.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn resumen_por_proyecto_y_modelo() {
    use shiori::adaptadores::metricas_sqlite::{Filtro, resumen};
    let e = entorno(OpcionesFalso::default()).await;
    for proyecto in ["a", "a", "b"] {
        e.cliente
            .post(e.url("/v1/messages"))
            .header("x-shiori-project", proyecto)
            .header("authorization", "Bearer sk-ant-oat01-x")
            .json(&cuerpo_anthropic(false))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }
    e.llamadas(3).await;
    let todo = resumen(
        &e.metricas,
        &Filtro {
            proyecto: None,
            desde_ms: None,
        },
    )
    .unwrap();
    assert_eq!(todo.len(), 2);
    let a = todo.iter().find(|f| f.project == "a").unwrap();
    assert_eq!(a.calls, 2);
    assert_eq!(a.input_tokens, 2 * ENTRADA);
    assert_eq!(a.auth, "subscription");
    assert_eq!(a.model, "claude-opus-5-5");
    let solo_b = resumen(
        &e.metricas,
        &Filtro {
            proyecto: Some("b".into()),
            desde_ms: None,
        },
    )
    .unwrap();
    assert_eq!(solo_b.len(), 1);
    let futuro = resumen(
        &e.metricas,
        &Filtro {
            proyecto: None,
            desde_ms: Some(i64::MAX),
        },
    )
    .unwrap();
    assert!(futuro.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn head_sin_cuerpo_no_es_un_corte() {
    // Claude Code hace `HEAD /api/hello` al arrancar.
    let e = entorno(OpcionesFalso::default()).await;
    let r = e.cliente.head(e.url("/api/hello")).send().await.unwrap();
    assert_eq!(r.status(), 404);
    let l = &e.llamadas(1).await[0];
    assert_eq!(l.metodo, "HEAD");
    assert!(!l.cancelada);
}

#[tokio::test(flavor = "multi_thread")]
async fn al_parar_la_base_de_metricas_queda_cerrada() {
    let e = entorno(OpcionesFalso::default()).await;
    e.cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    e.llamadas(1).await;
    e.proxy.parar().await;
    // En Windows no se puede borrar un archivo abierto: si la conexión siguiera viva, fallaría.
    for r in shiori::bench::archivos_metricas(&e.metricas) {
        match std::fs::remove_file(&r) {
            Err(err) if err.kind() != std::io::ErrorKind::NotFound => {
                panic!("{} sigue abierto: {err}", r.display())
            }
            _ => {}
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn al_soltar_el_servidor_sin_parar_la_base_queda_cerrada() {
    let e = entorno(OpcionesFalso::default()).await;
    e.cliente
        .post(e.url("/v1/messages"))
        .json(&cuerpo_anthropic(false))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    e.llamadas(1).await;
    // Como hace `Entorno` al acabar cada test: la tarea del servidor sigue viva con sus métricas.
    drop(e.proxy);
    for r in shiori::bench::archivos_metricas(&e.metricas) {
        match std::fs::remove_file(&r) {
            Err(err) if err.kind() != std::io::ErrorKind::NotFound => {
                panic!("{} sigue abierto: {err}", r.display())
            }
            _ => {}
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn bench_no_deja_su_base_en_temp() {
    let informe = shiori::bench::ejecutar(shiori::bench::OpcionesBench {
        peticiones: 20,
        concurrencia: 2,
        tokens: 5,
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(informe.results.iter().all(|m| m.errors == 0));
    for r in shiori::bench::archivos_metricas(&shiori::bench::ruta_metricas_bench()) {
        assert!(!r.exists(), "quedó {}", r.display());
    }
    let cero = shiori::bench::ejecutar(shiori::bench::OpcionesBench {
        peticiones: 0,
        ..Default::default()
    })
    .await;
    assert!(cero.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn el_falso_solo_recuerda_las_ultimas() {
    let falso = shiori::falso::arrancar("127.0.0.1:0", OpcionesFalso::default())
        .await
        .unwrap();
    let cliente = comun::cliente();
    let total = shiori::falso::MAX_VISTAS + 10;
    for i in 0..total {
        cliente
            .get(format!("{}/v1/models?n={i}", falso.url()))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }
    let vistas = falso.vistas();
    assert_eq!(vistas.len(), shiori::falso::MAX_VISTAS);
    assert_eq!(vistas[0].ruta, "/v1/models?n=10");
    assert_eq!(
        falso.ultima().unwrap().ruta,
        format!("/v1/models?n={}", total - 1)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn el_retraso_entre_tokens_cumple_lo_pedido() {
    let falso = shiori::falso::arrancar("127.0.0.1:0", OpcionesFalso::default())
        .await
        .unwrap();
    let cliente = comun::cliente();
    // 50 tokens a 2 ms: unos 110 ms (55 trozos). Con una espera tras otra, en Windows cada una
    // duraba un tic de ~15,6 ms y el total pasaba de 800 ms.
    let inicio = Instant::now();
    let cuerpo = cliente
        .post(format!("{}/v1/messages", falso.url()))
        .header("x-fake-tokens", "50")
        .header("x-fake-delay-ms", "2")
        .json(&cuerpo_anthropic(true))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let ms = inicio.elapsed().as_millis();
    assert!(String::from_utf8_lossy(&cuerpo).contains("message_stop"));
    assert!(ms >= 100, "demasiado rápido: {ms} ms");
    assert!(ms < 500, "demasiado lento: {ms} ms");
}
