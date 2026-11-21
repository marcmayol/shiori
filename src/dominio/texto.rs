//! Codificación por porcentajes para meter una carpeta (con `/`, `\`, `:`, espacios o acentos) en un
//! solo segmento de URL o en una cabecera HTTP, que solo admite ASCII visible.

pub fn codificar(texto: &str) -> String {
    let mut salida = String::with_capacity(texto.len());
    for b in texto.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            salida.push(b as char);
        } else {
            salida.push_str(&format!("%{b:02X}"));
        }
    }
    salida
}

/// `None` si hay un `%` mal formado o el resultado no es UTF-8.
pub fn decodificar(texto: &str) -> Option<String> {
    let bytes = texto.as_bytes();
    let mut salida = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = texto.get(i + 1..i + 3)?;
            salida.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            salida.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(salida).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ida_y_vuelta() {
        for t in [
            "/Users/marc/Proyectos/año nuevo",
            r"C:\Users\marc\D:\x y",
            "",
            "100%",
        ] {
            let c = codificar(t);
            assert!(c.bytes().all(|b| b.is_ascii_graphic()));
            assert!(!c.contains('/'));
            assert_eq!(decodificar(&c).as_deref(), Some(t));
        }
    }

    #[test]
    fn mal_formado() {
        assert_eq!(decodificar("%"), None);
        assert_eq!(decodificar("%G1"), None);
        assert_eq!(decodificar("%FF"), None);
    }
}
