# Speed levers for DeepSeek and GLM on this box (2026-11-10)

Where a token's time goes on the two big models, read off the measurements in `docs/ds4/V41_EXACT_FAST.md`,
`docs/MODELS.md` and the engines' own logs of the last weeks (`local/serve-engine.log`, `local/serve-glm-engine.log`:
real agent sessions, not benchmarks), and what could take time away without changing the routing or the weights.
Every "measured" number below was measured on this box; every "estimate" is derived from one and says from which.
The box: two RX 7900 XTX (card 1 on a PCIe 4.0 x2 chipset link, ~3 GB/s), Ryzen 9 9900X, 122 GiB RAM, MSI MAG
X670E Tomahawk WiFi.

## What the sessions say

**DeepSeek, 662 turns of the serve log (376,999 decode tokens).** Wall time: decode 20,205 s (70%), prompt
phases 7,867 s (27%), the VRAM cache re-seeded after a batch 920 s (3%). Decode 18.7 tok/s mean; per token
127.7 experts from the two cards, 30.5 computed by the CPU and 2.53 read from the NVMe. The decode speed follows
the file reads (turns of 100+ tokens, grouped by file reads a token):

| file reads / token | CPU experts / token | tok/s |
| --- | ---: | ---: |
| 0.1-0.9 | 20.0 | 21.9 |
| 1.8-2.5 | 28.5 | 19.2 |
| 3.1-3.7 | 32.5 | 18.1 |
| 5.1-7.0 | 41.2 | 16.6 |
| 7.0-8.8 | 45.7 | 15.0 |

The prompt phases by new tokens (the part of a turn read again or new; the rest is resumed from the conversation):

| new tokens | turns | seconds | median tok/s | dense share |
| --- | ---: | ---: | ---: | ---: |
| <= 96 (decode passes) | 248 | 275 | 20 | - |
| 97-512 (batch) | 137 | 681 | 57 | 22% |
| 513-2048 | 158 | 1,560 | 105 | 51% |
| 2049-8192 | 98 | 1,951 | 169 | 66% |
| > 8192 | 21 | 3,401 | 452 | 58% |

A short batch is mostly its experts: every routed expert of the batch is copied from the RAM arena (or read from the
NVMe) over PCIe, the VRAM cache having been released for the prompt's buffers, and re-seeded after it (1.0-1.8 s a
turn). A 214-token turn: 3.44 s of prompt (experts 3.02) + 1.07 s of seed before the first token.

**GLM, 2,660 decode windows and 57 prompts.** Decode median 36.9 ms a token (27 tok/s; the benchmark's short
prompts run 33.8): VRAM hits 77.5%, 23 experts a token copied from RAM into VRAM, 57 computed by the CPU lane in
17.7 ms - half of the token. Prompts: 113 tok/s median under 1K tokens, 241 at 1-4K, 373 at 4-16K; card 1's part of
the layer split takes 1.4-2.3x card 0's wall (its copy lane runs at 3.0 GB/s, card 0's at 26), so for prompts under
~4K tokens card 0 waits for card 1 for half of the time.

## Where a DeepSeek decode token goes (measured, V41_EXACT_FAST.md)

~43 ms on a short prompt (23 tok/s), ~53 in the sessions:

| part | ms | what it is |
| --- | ---: | --- |
| begin + finish | 2.4 | the Engram rows, the head |
| attention + router, card 0 | 22.5-25 | 40 layers x 0.56 ms: ~0.18 ms of weight reads (mul_mat_vec_q at 670-920 GB/s), ~0.23 ms in ~54 kernels of 2-5 us each (norms, rope, gathers, clamps, the router's chain), ~0.15 ms of gaps between dependent kernels inside the HIP graph |
| the tier | 14.7-27 | the CPU pool 7-17 ms (35-74 experts at 41-47 GB/s, the RAM's limit for this kernel), the file reads, card 1's rows and the sum |
| host round trips | ~6 | 40 x (router readback, the tier's decision, the next graph) |

The expert kernels on the cards are not the cost (3.9 ms busy on card 0, 1.1 on card 1, arithmetic-bound).

## Levers, by what they could take (estimates from the numbers above)

### Measured in this pass (2026-11-10)

**GLM, one CPU pool for both cards** (`ALMOND_GLM_CPU_SHARED=1`, 11 threads, against the default 6 a card;
`local/bench-v41/ab-glm-cpu-shared.sh`, the server, `speed.py` 2 rounds of 3 prompts, arms alternated twice):
base 35.0 / 34.7 tok/s, shared 35.3 / 34.9. The same. The lane's own calibration says why: an expert costs
0.206-0.211 ms on 6 threads and 0.210-0.211 on 11 (card 0's lane; card 1's 0.177 / 0.176): 10.8 MB in 0.21 ms is
~52 GB/s, the RAM's limit, so the lane is bound by the memory, not the cores. Not adopted (lever 7 is closed).

**DeepSeek, one-shot levers** (`local/bench-v41/ab-speed-levers.sh`: the configuration's flags, forced on the
reference streams, the 17K prompt with 128 tokens and Promessi Sposi with 300, 2 rounds alternated, power-tune
off - which is why the base reads 17 and 21 tok/s here against 18.4 and 23.6 in V41_EXACT_FAST.md: the file-tier
reads took 2.6-3.7 ms each instead of ~1.3; the arms compare among themselves), tok/s r1 / r2:

| arm | 17K context | Promessi Sposi | file reads (17K / Sposi) | argmax differences |
| --- | --- | --- | --- | --- |
| the configuration | 17.08 / 16.72 | 21.14 / 21.19 | 286-296 / 193 | 1 / 2 |
| `ALMOND_ADVISE_AHEAD=2` (page-cache readahead of layer l+2's predicted file experts) | 14.57 / 16.67 | 20.32 / 19.70 | 292 / 193 | 1 / 2 |
| `--arena-gib 88` | 13.84 / 16.60 | 18.89 / 21.06 | 198-208 / 145 | 1 / 5-6 |
| `--threads 24` (the SMT siblings, 13 of them unpinned) | 7.10 / 7.54 | 9.96 / 11.65 | | 1 / 2 |
| `--skip-miss 0.15 --skip-miss-from 21` | 18.38 / 18.09 | 21.77 / 18.92 | 226-230 / 109 | 1 / 12 |

- The readahead and the larger arena do not pay: the arena at 88 GiB reads a third fewer blobs but the box is left
  with ~25 GiB (the dense half's mmap and the Engram rows then compete with the arena for the page cache) and the
  decode is no faster; the readahead's hints land in a page cache that is already under pressure. Both stay off.
- 24 threads halve the speed: the pool pins one worker a physical core and the 13 extra ones float, and the
  CPU-bound part of the kernel (the gathers and table lookups) gains nothing from a sibling. 12 stays.
- The late-layer skip is +6-8% at 17K (7,330 CPU experts instead of 9,400) and inside the spread short; it changes
  the answers (12 argmax differences in 300 tokens against the configuration's 2) - its PPL is not measured
  (Lucebox's eval runs the prompt through the batch path, which the skip does not touch; a decode-path PPL is to be
  written first). Off.

**DeepSeek, the VRAM cache lent to a small batch** (`ALMOND_SERVE_BATCH_KEEP=<tokens>`; built 2026-11-11, 1024 in the DeepSeek configurations since).
A batch of up to that many new tokens keeps the VRAM cache: the tier (`Ds41MoeTier::lend_cache`) takes the experts
out of the cache's last slots (the lowest ranked), demotes each into the RAM arena (device -> host into the slot of
the layer's least recently used arena expert, 0.06-0.21 s for 540-700 experts) and unmaps whole segments of the
cache until `ALMOND_SERVE_LEND_GIB` (6.0) of VRAM is free; the chunk path computes the ~1,400 experts still resident
in place and streams the rest from the arena with 256 MiB halves (`ALMOND_LEND_STAGE_MIB`) and no prestage; after
the batch the pass graphs' buffers go back, the segments are mapped again and the experts copied back from the
arena (`reclaim_cache`, 0.2-0.35 s), their LRU age as it was. A lent batch that does not fit is run again the old
way (measured: with 4.96 GiB free the pass graphs' third growth, 2.15 GiB, failed; 5.25 was enough - hence 6.0).
The cache is opened as a segmented arena for this (`--cache-segment-mib`, 256 by default with the loan: #533's
elastic cache, which was CUDA-only, now on HIP through hipMemAddressReserve / hipMemCreate / hipMemMap).
A VRAM-resident expert's arena copy no longer counts as used under `--arena-skip-vram`, so the duplicates a loan
or an LRU promotion leave behind age out of the arena.

Measured through the server (`local/bench-v41/ab-batch-keep.sh`, `local/sweep-batch-keep/`): fresh 3-turn
conversations (a ~2,100-2,900-token first turn, then two turns of 435-567 new tokens each, 48 tokens of answer at
temperature 0), arms alternated on different slices of the same text, power-tune on. The turns the loan took
(new tokens within the limit) against every base turn of the same shape:

| ~500-token turn | prompt | the cache after it | the 48 tokens after it | the turn's wall |
| --- | ---: | ---: | ---: | ---: |
| the configuration (6 turns) | 5.9-7.8 s | re-seeded in 1.5-7.9 s (1,846-2,115 experts) | 11.1-17.0 tok/s | 10.5-18.5 s |
| lent (7 turns of 454-547 new tokens, the limit at 512 and at 1024) | 4.05-5.12 s | reclaimed in 0.22-0.35 s (all experts back) | 16.4-19.1 tok/s | 6.8-8.3 s |

In a lent turn 41,000-54,000 of the ~115,000-131,000 routed rows were computed from the cache in place and
66,000-78,000 copied over PCIe (a base turn copies all 118,000-136,000); the decode after it starts warm (the base's starts from the rank,
with the cache 12% smaller until ggml gives the batch's buffers back). The first turn (a 2-3K batch from scratch) and
turns past the limit are unchanged. Not yet measured: a turn of 600-1024 new tokens (two pieces of 512 columns), agent sessions
at 100K+ (the pass graphs grow with the context; a batch that does not fit falls back), the decode's hit rate over a
long session with the demoted duplicates in the arena.

### DeepSeek

1. **Fewer file-tier reads.** Each costs ~1.2 ms of NVMe time and, in the sessions, 2.5 a token comes with -15% of
   the speed against turns with under one. (a) A larger arena (`--arena-gib`): 80 GiB holds 10,001 of the 15,360
   experts, the file tier 3,177; the machine has 122 GiB and the engine pins ~86 (arena, conversation pool,
   prestage). (b) `ALMOND_ADVISE_AHEAD=2`: layer l predicts layer l+2's routing (top-8 covers 59-72% of file
   misses, V41_EXACT_FAST.md) and asks the kernel to read those blobs into the page cache; off by default, never
   measured on this pack. (c) `ALMOND_SPEC_TOP`: O_DIRECT reads into a pinned stash, measured a wash on another
   configuration (2026-10-12: the tier 1.8 ms faster, the bigger graph and the issuing 2.8 ms slower).
2. **The CPU tier at the RAM's speed.** The kernel reads 41-47 GB/s cold with 12 threads and computes at 58-65 from
   L3; the RAM streams 62. Levers: the SMT siblings (`--threads 24`: the per-thread decode has gathers and table
   lookups, where a second thread a core fills the stalls), an L3 prefetch of the next layer's predicted CPU experts
   during the attention (60-70% of them from L3 at 1.4-1.6x: +3-6%, estimate in V41_EXACT_FAST.md), a kernel that
   keeps two rows in flight a thread.
3. **The dense half's small kernels.** Of the 0.56 ms a layer's attention takes, ~0.4 is kernel time and only
   ~0.18 is the weights; the other ~0.2 ms is ~54 kernels of 2-5 us plus ~0.15 ms of gaps. A hand-fused attention
   layer (norm + MLA/DSA projections + rope + the KV write + the indexer + the router's chain in a few kernels) would
   bring the layer towards the weights' 0.2-0.25 ms: **6-10 ms a token, 15-20%**, the largest software lever, and
   the hardest (gfx1100 kernels for 512-wide heads and the compressed KV). Half-way: more ggml-graph fusions
   (the router's 10 kernels became 1 for 1.3 ms a token; the K/V gather, the clamps and the norms are of the same
   size).
4. **The host round trips.** ~6 ms a token is the 40 per-layer waits (router readback -> the tier's decision -> the
   expert launches). The hits on card 0 (84% of the experts short, 63% at 17K) do not need the host: a kernel on
   card 0 could read the router's ids, look each one up in a device-side residency table and run the resident
   ones at once (indirect launch, graph-captured), while the host reads back the ids and handles only the misses
   (CPU, card 1, file). The readback then overlaps card 0's own expert work instead of preceding it: **estimate
   2-4 ms a token (5-8%)**. The VRAM cache's bookkeeping (LRU, promotions) stays on the host, published after the
   layer as now.
5. **Agent turns: keep the VRAM cache for small batches: built, measured above (opt-in).** A 100-1,000-token batch needs little of the VRAM the
   8,192-token prompt path takes (the state columns are ~2.5 GiB at 8,192 columns, elastic), yet releases the whole
   cache: the ~4,900 experts the two cards hold (32% of the model) are streamed again from RAM, and the seed after
   the batch costs 1-1.8 s. A batch that keeps the cache and serves the resident experts in place would save a
   third of the expert copies of those turns and the seed: on the sessions above **~900 s of seeds and ~700 s of
   expert copies out of ~29,000 s (5%)**, and 1-2 s less before the first token of every turn.
6. **Light misses left out of the late layers** (`--skip-miss W --skip-miss-from L`, as GLM's `ALMOND_GLM_MISS_SKIP_W`
   from layer 23: +30% for +0.5% PPL). On DeepSeek `--skip-miss 0.15` on every layer measured +7% for +0.6-0.8%
   decode PPL (HANDOVER.md); from a late layer only it should keep most of the speed at a fraction of the PPL, as on
   GLM. It changes the answers: a quality measurement (Lucebox) decides it.

### GLM

7. **One CPU pool instead of one per card: measured, no gain (above).** The lane is sized `cores / parts` = 6 threads a card, a pool each,
   because two parts can run at once (the pipelined decode of several conversations); one conversation visits the
   cards in turn, so half the cores idle while the other half computes 57 experts a token in 17.7 ms.
   `ALMOND_GLM_CPU_SHARED=1`: one pool of 11 threads for both parts. The split CPU/PCIe is re-planned from the
   measured cost (0.233 ms an expert at 6 threads vs 0.504 over PCIe), so the lane also takes more of the misses.
8. **The prompt's slow half (card 1 at 3 GB/s).** For prompts under ~4K tokens card 1's part takes 1.4-2.3x card
   0's, and card 0 idles meanwhile. Card 1's non-resident experts could be computed on card 0 from the RAM arena
   (card 0's copy lane: 26 GB/s) with the chunk's activations hopping the link (T x 4096 x 2 bytes each way a
   layer: 8 MB at 1K tokens, 2.7 ms), while card 1 computes its resident ones: **a 1K-token prompt from 4.9 s
   towards ~2 s (estimate from the two parts' copy lanes: 24 + 8 GB at 26 GB/s)**. For prompts past ~16K card 0's
   lane is 88% busy already (V41_EXACT_FAST.md) and the gain shrinks.
9. **The dense half's Q8_0.** 9.6 GB of Q8_0 dense weights (attn_output 2.0, q/k/v 3.6, the shared experts 1.1,
   the head 0.7, ...) are read for every verify window: ~12 ms at the cards' ~800 GB/s. q6_K for the projections and
   the shared experts would read 2.3 GB less (**~2.5 ms a window, 6-8%**); on DeepSeek the same requant of its
   Q8_0 matrices cost ppl +0.003 (V41_EXACT_FAST.md), here it is to be measured (PPL on `long-docs.txt`); GLM's
   dense path has Q8_0 kernels only, so it is a kernel and a pack change, not a flag.

### The hardware

10. **Card 1's link.** PCIe 4.0 x2 (~3 GB/s, 26-46 us a small copy) sets GLM's prompt speed under 4K tokens and
    card 1's hand-off latency in DeepSeek's decode. The board has one CPU x16 slot; a CPU-attached M.2 (Gen4/5 x4)
    with an OCuLink adapter would give card 1 x4 (8 GB/s, 2x); a bifurcation riser in the x16 slot (x8/x8, if
    the BIOS offers it) would give it 16 GB/s but halve card 0's 26 GB/s copy lane, which DeepSeek's prompts and
    GLM's long prompts are bound by: a trade to measure, not a clear win.
11. **RAM bandwidth.** The CPU tiers read at the RAM's limit (62 GB/s streaming, 47 in the kernels); the speed of
    the DIMMs (dmidecode needs root here) and the fabric clock (FCLK 2100, UCLK = MEMCLK) are worth 10-15% of that:
    1-2 ms a DeepSeek token, 2-3 ms a GLM token.
