use criterion::{Criterion, criterion_group, criterion_main};
use shiori::dominio::proveedor::Proveedor;
use shiori::dominio::uso::Medidor;

fn sse_anthropic(n: usize) -> Vec<u8> {
    let mut s = String::from(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-5-5\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
    );
    for i in 0..n {
        s.push_str(&format!("event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"palabra{i} \"}}}}\n\n"));
    }
    s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1000}}\n\n");
    s.into_bytes()
}

fn medir_uso(c: &mut Criterion) {
    let cuerpo = sse_anthropic(1000);
    c.bench_function("uso_sse_anthropic_1000_tokens", |b| {
        b.iter(|| {
            let mut m = Medidor::nuevo(Proveedor::Anthropic, true);
            for trozo in cuerpo.chunks(512) {
                m.alimentar(trozo);
            }
            m.terminar()
        })
    });
}

criterion_group!(benches, medir_uso);
criterion_main!(benches);
