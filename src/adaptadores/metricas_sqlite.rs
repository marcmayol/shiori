//! Métricas en SQLite. Las llamadas se mandan por un canal a un hilo propio que escribe por lotes:
//! el proxy nunca espera al disco.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};

use rusqlite::{Connection, params};
use serde::Serialize;

use crate::dominio::llamada::Llamada;
use crate::dominio::puertos::AlmacenMetricas;

const ESQUEMA: &str = "
CREATE TABLE IF NOT EXISTS llamadas (
    id INTEGER PRIMARY KEY,
    momento_ms INTEGER NOT NULL,
    proyecto TEXT NOT NULL,
    proveedor TEXT NOT NULL,
    modelo TEXT,
    metodo TEXT NOT NULL,
    ruta TEXT NOT NULL,
    estado INTEGER NOT NULL,
    stream INTEGER NOT NULL,
    autenticacion TEXT NOT NULL,
    entrada INTEGER NOT NULL,
    salida INTEGER NOT NULL,
    cache_leida INTEGER NOT NULL,
    cache_escrita INTEGER NOT NULL,
    coste_usd REAL,
    ahorro_usd REAL,
    latencia_ms INTEGER NOT NULL,
    primer_byte_ms INTEGER,
    bytes INTEGER NOT NULL,
    cancelada INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS llamadas_momento ON llamadas(momento_ms);
CREATE INDEX IF NOT EXISTS llamadas_proyecto ON llamadas(proyecto, momento_ms);
";

enum Mensaje {
    Llamada(Box<Llamada>),
    Vaciar(mpsc::Sender<()>),
}

pub struct MetricasSqlite {
    tx: SyncSender<Mensaje>,
}

pub fn abrir(ruta: &Path) -> Result<Connection, String> {
    if let Some(carpeta) = ruta.parent()
        && !carpeta.as_os_str().is_empty()
    {
        std::fs::create_dir_all(carpeta)
            .map_err(|e| format!("cannot create {}: {e}", carpeta.display()))?;
    }
    let c = Connection::open(ruta).map_err(|e| format!("cannot open {}: {e}", ruta.display()))?;
    c.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| e.to_string())?;
    c.pragma_update(None, "journal_mode", "WAL")
        .map_err(|e| e.to_string())?;
    c.pragma_update(None, "synchronous", "NORMAL")
        .map_err(|e| e.to_string())?;
    c.execute_batch(ESQUEMA)
        .map_err(|e| format!("cannot prepare {}: {e}", ruta.display()))?;
    c.pragma_update(None, "user_version", 1)
        .map_err(|e| e.to_string())?;
    Ok(c)
}

impl MetricasSqlite {
    pub fn nuevo(ruta: &Path) -> Result<Self, String> {
        let conexion = abrir(ruta)?;
        let (tx, rx) = mpsc::sync_channel(10_000);
        std::thread::Builder::new()
            .name("shiori-metricas".into())
            .spawn(move || escritor(conexion, rx))
            .map_err(|e| e.to_string())?;
        Ok(MetricasSqlite { tx })
    }
}

impl AlmacenMetricas for MetricasSqlite {
    fn registrar(&self, llamada: Llamada) {
        match self.tx.try_send(Mensaje::Llamada(Box::new(llamada))) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!("metrics queue full: one call was not recorded")
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::warn!("metrics writer stopped: one call was not recorded")
            }
        }
    }

    fn vaciar(&self) {
        let (tx, rx) = mpsc::channel();
        if self.tx.send(Mensaje::Vaciar(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

fn escritor(mut conexion: Connection, rx: Receiver<Mensaje>) {
    while let Ok(primero) = rx.recv() {
        let mut lote = vec![primero];
        while let Ok(m) = rx.try_recv() {
            lote.push(m);
            if lote.len() >= 500 {
                break;
            }
        }
        let mut avisos = Vec::new();
        let resultado = (|| -> rusqlite::Result<()> {
            let t = conexion.transaction()?;
            {
                let mut insertar = t.prepare_cached(
                    "INSERT INTO llamadas (momento_ms, proyecto, proveedor, modelo, metodo, ruta, estado,
                     stream, autenticacion, entrada, salida, cache_leida, cache_escrita, coste_usd,
                     ahorro_usd, latencia_ms, primer_byte_ms, bytes, cancelada)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
                )?;
                for m in &mut lote {
                    match m {
                        Mensaje::Llamada(l) => {
                            insertar.execute(params![
                                l.momento_ms,
                                l.proyecto,
                                l.proveedor.nombre(),
                                l.uso.modelo,
                                l.metodo,
                                l.ruta,
                                l.estado,
                                l.stream,
                                l.autenticacion.nombre(),
                                l.uso.entrada as i64,
                                l.uso.salida as i64,
                                l.uso.cache_leida as i64,
                                l.uso.cache_escrita as i64,
                                l.coste_usd,
                                l.ahorro_usd,
                                l.latencia_ms as i64,
                                l.primer_byte_ms.map(|v| v as i64),
                                l.bytes as i64,
                                l.cancelada,
                            ])?;
                        }
                        Mensaje::Vaciar(tx) => avisos.push(tx.clone()),
                    }
                }
            }
            t.commit()
        })();
        if let Err(e) = resultado {
            tracing::warn!("cannot write metrics: {e}");
        }
        for a in avisos {
            let _ = a.send(());
        }
    }
}

// ---------- Consulta (para `shiori metrics`) ----------

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FilaResumen {
    pub project: String,
    pub provider: String,
    pub model: String,
    pub auth: String,
    pub calls: u64,
    pub errors: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// `None` si ninguna llamada del grupo tenía precio conocido.
    pub cost_usd: Option<f64>,
    pub cache_savings_usd: Option<f64>,
    pub avg_latency_ms: f64,
}

pub struct Filtro {
    pub proyecto: Option<String>,
    pub desde_ms: Option<i64>,
}

pub fn resumen(ruta: &Path, filtro: &Filtro) -> Result<Vec<FilaResumen>, String> {
    if !ruta.exists() {
        return Ok(Vec::new());
    }
    let c = abrir(ruta)?;
    let mut sentencia = c
        .prepare(
            "SELECT proyecto, proveedor, COALESCE(modelo, '-'), autenticacion, COUNT(*),
                    SUM(estado >= 400), SUM(entrada), SUM(salida), SUM(cache_leida), SUM(cache_escrita),
                    SUM(coste_usd), SUM(ahorro_usd), AVG(latencia_ms)
             FROM llamadas
             WHERE (?1 IS NULL OR proyecto = ?1) AND (?2 IS NULL OR momento_ms >= ?2)
             GROUP BY 1, 2, 3, 4
             ORDER BY SUM(COALESCE(coste_usd, 0)) DESC, COUNT(*) DESC",
        )
        .map_err(|e| e.to_string())?;
    let filas = sentencia
        .query_map(params![filtro.proyecto, filtro.desde_ms], |f| {
            Ok(FilaResumen {
                project: f.get(0)?,
                provider: f.get(1)?,
                model: f.get(2)?,
                auth: f.get(3)?,
                calls: f.get::<_, i64>(4)? as u64,
                errors: f.get::<_, i64>(5)? as u64,
                input_tokens: f.get::<_, i64>(6)? as u64,
                output_tokens: f.get::<_, i64>(7)? as u64,
                cache_read_tokens: f.get::<_, i64>(8)? as u64,
                cache_write_tokens: f.get::<_, i64>(9)? as u64,
                cost_usd: f.get(10)?,
                cache_savings_usd: f.get(11)?,
                avg_latency_ms: f.get(12)?,
            })
        })
        .map_err(|e| e.to_string())?;
    filas.collect::<Result<_, _>>().map_err(|e| e.to_string())
}

/// Todas las llamadas, de la más reciente a la más antigua (para los tests y `--json`).
pub fn llamadas(ruta: &Path, limite: usize) -> Result<Vec<Llamada>, String> {
    use crate::dominio::autenticacion::Autenticacion;
    use crate::dominio::proveedor::Proveedor;
    use crate::dominio::uso::Uso;
    let c = abrir(ruta)?;
    let mut s = c
        .prepare(
            "SELECT momento_ms, proyecto, proveedor, modelo, metodo, ruta, estado, stream, autenticacion,
                    entrada, salida, cache_leida, cache_escrita, coste_usd, ahorro_usd, latencia_ms,
                    primer_byte_ms, bytes, cancelada
             FROM llamadas ORDER BY id DESC LIMIT ?1",
        )
        .map_err(|e| e.to_string())?;
    let filas = s
        .query_map(params![limite as i64], |f| {
            let proveedor: String = f.get(2)?;
            let auth: String = f.get(8)?;
            Ok(Llamada {
                momento_ms: f.get(0)?,
                proyecto: f.get(1)?,
                proveedor: Proveedor::desde_nombre(&proveedor).unwrap_or(Proveedor::Anthropic),
                metodo: f.get(4)?,
                ruta: f.get(5)?,
                estado: f.get(6)?,
                stream: f.get(7)?,
                autenticacion: Autenticacion::desde_nombre(&auth),
                uso: Uso {
                    modelo: f.get(3)?,
                    entrada: f.get::<_, i64>(9)? as u64,
                    salida: f.get::<_, i64>(10)? as u64,
                    cache_leida: f.get::<_, i64>(11)? as u64,
                    cache_escrita: f.get::<_, i64>(12)? as u64,
                },
                coste_usd: f.get(13)?,
                ahorro_usd: f.get(14)?,
                latencia_ms: f.get::<_, i64>(15)? as u64,
                primer_byte_ms: f.get::<_, Option<i64>>(16)?.map(|v| v as u64),
                bytes: f.get::<_, i64>(17)? as u64,
                cancelada: f.get(18)?,
            })
        })
        .map_err(|e| e.to_string())?;
    filas.collect::<Result<_, _>>().map_err(|e| e.to_string())
}
