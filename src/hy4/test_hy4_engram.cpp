// test_hy4_engram <gguf> - Fase 1 gate: the Engram layout, the n-gram hash and the FP8 row decode, checked against the
// model's own file and against PR #32517's algorithm. No weights are computed; the hash invariants and a few real
// rows of the 63 GB table are what is tested here.
#include "hy4_engram.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace almond;
using namespace almond::hy4;

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_hy4_engram <hy4.gguf>\n"); return 2; }
    int fails = 0;
    auto check = [&](const char * what, bool ok, const std::string & detail = "") {
        std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  ", detail.c_str());
        fails += !ok;
    };

    std::unique_ptr<GgufModel> mp;
    try { mp = std::make_unique<GgufModel>(GgufModel::open(argv[1])); }
    catch (const std::exception & e) { std::fprintf(stderr, "open: %s\n", e.what()); return 1; }
    Hy4Geometry G;
    std::string e = hy4_read_geometry(*mp, G);
    if (!e.empty()) { std::fprintf(stderr, "geometry: %s\n", e.c_str()); return 1; }
    const EngramLayout L = hy4_engram_layout(G);
    check("four modules", L.modules.size() == 4 && L.modules[0].layer == 1 && L.modules[3].layer == 31);
    check("16 columns per module", L.heads == 8 && L.n_gram == 3 && (int64_t) L.modules[0].prime.size() == 16);

    // The converter rebases each module's buckets into one table: offsets must start at zero and the ranges must
    // tile it with no gap or overlap.
    uint64_t rows = 0;
    bool tiled = true;
    for (const auto & m : L.modules)
        for (size_t c = 0; c < m.prime.size(); ++c) {
            if (m.offset[c] != rows || m.prime[c] < 2) tiled = false;
            rows += m.prime[c];
        }
    check("bucket ranges tile the table", tiled, "rows=" + std::to_string(rows));
    check("table row count", rows == 660006416ull);

    // Every hash must land inside its head's bucket range, and the pad token's compressed id must be in range.
    bool in_range = true, deterministic = true, varies = true;
    for (size_t em = 0; em < L.modules.size(); ++em) {
        for (int64_t tok : {0, 1, 42, 1000, 120000, 120025, 120831, -1, 120832}) {
            const int64_t prev[2] = { 7, 9 };
            int32_t ids[16] = {}, ids2[16] = {};
            float valid[16] = {}, valid2[16] = {};
            hy4_engram_hash(L, em, tok, prev, ids, valid);
            hy4_engram_hash(L, em, tok, prev, ids2, valid2);
            deterministic &= !std::memcmp(ids, ids2, sizeof ids) && !std::memcmp(valid, valid2, sizeof valid);
            for (size_t c = 0; c < L.modules[em].prime.size(); ++c) {
                const uint32_t lo = L.modules[em].offset[c], hi = lo + L.modules[em].prime[c];
                in_range &= (uint32_t) ids[c] >= lo && (uint32_t) ids[c] < hi;
            }
            if (tok == 0) { int32_t other[16]; float v[16];
                hy4_engram_hash(L, em, 1, prev, other, v);
                varies &= std::memcmp(ids, other, 8 * sizeof(int32_t)) != 0;
            }
            if (tok == -1 || tok == 120832) {
                for (size_t c = 0; c < L.modules[em].prime.size(); ++c) valid[c] = 0.0f;
            }
        }
    }
    check("hashes inside their buckets", in_range);
    check("hash is deterministic", deterministic);
    check("a different token hashes differently", varies);

    // FP8 E4M3 decoding, including the subnormal and NaN ends of the format.
    struct { uint8_t b; float want; const char * name; } cases[] = {
        { 0x00, 0.0f, "+0" }, { 0x80, 0.0f, "-0" }, { 0x04, 0.0078125f, "subnormal" },
        { 0x38, 1.0f, "1" }, { 0xB8, -1.0f, "-1" }, { 0x3C, 1.5f, "1.5" }, { 0x40, 2.0f, "2" },
    };
    bool decode = true;
    for (auto & c : cases) {
        float got = hy4_f8_e4m3_to_float(c.b);
        decode &= std::fabs(got - c.want) < 1e-6f;
        std::printf("     fp8 0x%02x %-10s -> %g\n", c.b, c.name, got);
    }
    check("FP8 E4M3 decode", decode);
    check("FP8 NaN", std::isnan(hy4_f8_e4m3_to_float(0x7F)));

    // One real row from the downloaded table: 96 FP8 bytes, read straight from shard 2.  This only checks that the
    // bytes decode to finite values and that different rows differ - not that the quantizer's values are right.
    size_t which = 0;
    const TensorInfo * table = mp->find("per_layer_token_embd.weight", &which);
    if (!table) { std::fprintf(stderr, "no per_layer_token_embd.weight\n"); return 1; }
    check("table is FP8", table->type == 43 && table->shape.size() == 2 && table->shape[0] == 96);
    const uint8_t * data = mp->shard(which).tensor_data(*table);
    auto row = [&](uint64_t r, float * out) {
        for (int64_t j = 0; j < L.dim; ++j) out[j] = hy4_f8_e4m3_to_float(data[r * (uint64_t) L.dim + (uint64_t) j]);
    };
    float r0[96], r1[96], hashed[96];
    row(0, r0); row(1, r1);
    bool finite = true, distinct = false;
    for (int64_t j = 0; j < L.dim; ++j) {
        finite &= std::isfinite(r0[j]) && std::isfinite(r1[j]);
        distinct |= r0[j] != r1[j];
    }
    check("row 0 and row 1 are finite", finite);
    check("rows 0 and 1 differ", distinct);
    std::printf("     row 0: first %g %g %g, row 1: %g %g %g\n", r0[0], r0[1], r0[2], r1[0], r1[1], r1[2]);

    int32_t ids[16]; float valid[16];
    hy4_engram_hash(L, 0, 1234, (const int64_t[]){ 12, 34 }, ids, valid);
    if (ids[0] >= 0) { row((uint64_t) ids[0], hashed);
        bool ok = true; for (int64_t j = 0; j < L.dim; ++j) ok &= std::isfinite(hashed[j]);
        check("hashed row is finite", ok, "id=" + std::to_string(ids[0]));
    }
    std::printf("%s\n", fails ? "FAILED" : "all checks passed");
    return fails ? 1 : 0;
}
