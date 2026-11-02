# Revisione del motore Almond (2026-10-30)

Revisione in sola lettura di `src/ds41`, `src/glm`, delle parti condivise in `src/` e `include/`, di
`src/engine_main.cpp`, di `CMakeLists.txt`, di `build.sh` e di come `serve/server.py` usa il protocollo del motore.
Nessun codice del motore è stato cambiato; nessun server o motore è stato avviato o fermato.

Limiti: il codice conta ~150 000 righe. Ho letto per intero la modifica non committata, `engine_main.cpp`, i CMake,
`build.sh`, il ciclo `--serve` di DeepSeek nelle parti toccate e il lato server dei comandi; il resto (pool CPU degli
esperti, flussi GPU, `glm_fast_path.cu`, `generate.cpp` di GLM) l'ho solo campionato. Le gare tra pool CPU e stream
GPU non sono state verificate: restano da guardare (sotto, "Non coperto").

## Gravità alta

### A1. Un'allocazione fallita delle colonne lascia `Ds41Dense` senza stato, e il server continua
`src/ds41/ds41_dense.cpp:1089-1139` (`alloc_cols`), `:1142` (`reset_columns`), `:1149` (`set_columns`).
`alloc_cols` libera prima `nbuf` e `nctx`, scrive `im.NC = NC`, crea i tensori e solo alla fine alloca il buffer. Se
`ggml_backend_alloc_ctx_tensors` fallisce torna `false` con `NC` già al valore nuovo, tensori senza memoria e grafi
svuotati. Chi chiama dopo con la stessa larghezza trova `want == im.NC` e `set_columns` torna `true` senza fare nulla.
- Nella modifica nuova: `open_cache` (`src/ds41/ds41_generate.cpp:1181-1185`) chiama `reset_columns`; se fallisce
  (la VRAM serve proprio dopo un prompt con immagini) il ciclo `--serve` stampa `ERR VRAM seed failed` e fa
  `continue` (`:1894`, `:2099`). La richiesta seguente gira su tensori con `data == nullptr`: crash o scritture fuori
  posto sulla GPU.
- Già prima: `:1442` (`set_columns(0)` dopo un batch) ha lo stesso effetto se fallisce; `:1290` riprova a metà
  larghezza e quindi si riprende, ma solo perché la larghezza cambia.

Correzione proposta: in `alloc_cols`, se l'allocazione fallisce, rimettere `im.NC = -1` (così la chiamata dopo
ricostruisce davvero) oppure ricadere sulla larghezza minima `im.S`; nel ciclo `--serve` non tornare in attesa con lo
stato a metà.

## Gravità media

### M1. Il rilascio della VRAM dopo le immagini copre solo il percorso con la cache differita
`src/ds41/ds41_generate.cpp:1179` (`img_prompt`), `:1936`, `:731`.
`img_prompt` è consumato solo dentro `open_cache`, che in `--serve` gira solo se `defer_vram` (prefill a blocchi o
`prompt_batch > 1`) e la cache non è già aperta. Con `defer_vram` falso, o con la cache già aperta (percorso a passi,
`:1893`), il ~1,9 GiB tenuto dal prompt con immagini resta occupato fino al prompt con immagini seguente. Le
configurazioni in `local/` usano batch, quindi oggi il caso non si vede; va scritto, o il reset va fatto alla fine
della richiesta GENI invece che in `open_cache`.

Nota sulla modifica: il commento a `alloc_cols` (`:1086-1087`) dice che ogni colonna è scratch di una passata, quindi
ricostruirle tra il prompt e la decodifica non perde stato; nel percorso DSpark (`:2014`) `open_cache` gira prima della
coda di passate, che ricrea le sue colonne: corretto. `rdev1` viene riletto (`:1184`) come dopo `set_columns`: corretto;
non ho trovato altri puntatori a dispositivo presi da `dense` e tenuti tra le richieste (`:1341` lo rilegge ogni volta).
Manca una misura scritta della VRAM libera prima/dopo (l'opzione `ALMOND_VRAM_TRACE` aggiunta serve a questo).

### M2. `quantize.cu` di ggml compilato due volte nello stesso eseguibile
`CMakeLists.txt:118` (`almond_mmq`) e `src/glm/CMakeLists.txt:104-106` (`glm_mmq`).
Cinque simboli globali sono definiti in entrambe le librerie (`quantize_mmq_q8_1_cuda`, `quantize_row_q8_1_cuda`,
`quantize_mmq_fp4_cuda`, `quantize_scatter_mmq_q8_1_cuda`, `quantize_scatter_mmq_fp4_cuda`; trovati con `nm` sulle
`.a` in `build-hip-fused`). Il linker prende la prima copia senza avvisare. Oggi la sorgente è la stessa
(`ALMOND_GGML_DIR`), quindi è innocuo; se un albero cambia flag o patch, un percorso userà in silenzio la versione
dell'altro. È il primo candidato naturale del passo 5: una sola `almond_mmq` per i due modelli.

### M3. `build.sh` usa `rm -rf`
`build.sh:25`: `rm -rf "$B.old"; mv "$B" "$B.old"`. Contro la regola del progetto (`gio trash`); cancella senza
chiedere una vecchia build di ggml (decine di minuti di compilazione).

## Gravità bassa

### B1. Commento di `release_pass` staccato dal suo metodo
`src/ds41/ds41_dense.hpp:88-91`: la dichiarazione di `reset_columns` è finita tra il commento doc di `release_pass` e
`release_pass`, quindi il commento ora descrive `reset_columns`. Spostarla sopra il commento.

### B2. Rinomina meccanica che ha rotto le frasi
- `src/ds41/ds41_dense.cpp:1071` e `src/ds41/glm_trunk.cpp:28`: "llama.cpp's ... say "glm5-next", the "glm5next"" —
  è sparito il soggetto (prima "project-maya's"). Es.: "the GLM packs' own writer says "glm5next"".
- `CMakeLists.txt:3`: "Of the Almond libraries this tree started from" e `:31` "left with the Almond tree": qui
  "Almond" ha sostituito "Strata", che era l'albero d'origine; la frase ora non ha senso.
- `CMakeLists.txt:12`: indica `--target ds41_generate`; l'eseguibile è `almond-engine` (il target ha ancora il nome
  vecchio, solo `OUTPUT_NAME` cambia, `:148`).
- `src/ds41/v41_engine.cmake:1`: "included from tools/ds4/cmake/fork.cmake" — il file non esiste; è incluso da
  `CMakeLists.txt:135`.
- `src/ds41/ds41_generate.cpp:14`: "~/AI/Almond/serve/server.py" — il server è `serve/server.py` di questo repo.

### B3. Nomi non ancora Almond
- `src/ds41/ds41_moe.cpp`: macro e variabili `DS4_MOE_CUDA` (51 occorrenze), `DS4_MOE_MMQ` (7), `DS4_CHECK_LRU`,
  `DS4_CHECK_GPU`, `DS4_CHECK_READ`, `DS4_TIER_PROF`, `DS4_SERIAL_SEED`, funzioni `ds4_moe*`, `ds4_key_f`.
  (`DS41_*` per DeepSeek-V4.1 è un nome di modello, non del progetto: va bene.)
- `include/almond/artifact/ds4_geometry.hpp` (103 occorrenze di `ds4_`).
- `src/glm/prefill/moe_fused_rdna4.cu:179,402,438`: `-DSTRATA_W_NO_T16`, `-DSTRATA_W_LB`, `-DSTRATA_W_NO_OCC`.
- `src/glm/LICENSE:4` "the Strata contributors": è un'attribuzione di licenza, va lasciata.
- `build-hip-fused/src/glm/libstrata_*.a`: librerie vecchie rimaste nella cartella di build (nessun target `strata_*`
  esiste più nei CMake); da spostare nel cestino con la build, non sono linkate.

### B4. `command()` del server può attribuire una riga vecchia
`serve/server.py:884-903`. `command()` scarta le righe che non attende ma prende la prima `ERR` come risposta: una
`ERR` rimasta in coda da una richiesta fermata appena prima diventa il fallimento di `VLEND`. E se scade il timeout,
la risposta tardiva (`VLENT ...`) resta in coda per la richiesta seguente (che la ignora, ma il prestito della VRAM è
avvenuto senza che il server lo sappia: il `VRECLAIM` non viene mandato).

### B5. Comandi presenti solo da una parte
- `VLEND`/`VRECLAIM` e `LOGP` esistono solo nel percorso GLM (`src/glm/program/generate.cpp:3991-4024`); DeepSeek
  risponde `ERR unsupported command` (`src/ds41/ds41_generate.cpp:1608`). Il server li manda solo con
  `vision.on_demand: true`, che oggi c'è solo in `local/glm-local-vision.json`; una configurazione DeepSeek con
  quell'opzione farebbe fallire ogni richiesta con immagini. Conviene rifiutarla all'avvio.
- `LOGP` non è mai mandato da `serve/server.py`: usato solo da strumenti esterni, o codice morto.
- `BGEN`/`BGENI` (`serve/server.py:1387`) esistono solo in GLM.

### B6. `engine_main.cpp` cerca `--glm-pack` in tutti gli argomenti
`src/engine_main.cpp:10`: anche come valore di un'altra opzione. Innocuo oggi; con un `ALMOND_MODEL` o un primo
argomento `glm`/`ds41` sarebbe esplicito.

### B7. `getenv("ALMOND_VRAM_TRACE")` letto due volte a ogni `open_cache`
`src/ds41/ds41_generate.cpp:1186-1190`: costa nulla, ma lo stile del file legge le variabili una volta (`static`).

## Duplicati per il passo 5 (docs/ENGINE_PLAN.md)

File con lo stesso percorso in `src/` (DeepSeek) e `src/glm/` (GLM), righe e righe diverse (`diff`):

| file | src/ | src/glm/ | righe diverse |
|---|---:|---:|---:|
| core/expert_cache.cpp | 778 | 452 | 340 |
| core/pinned.cu | 800 | 366 | 496 |
| platform/memory.cpp | 198 | 76 | 130 |
| prefill/ggml_cuda_host.cu | 158 | 235 | 169 |
| prefill/moe_mmq.cu | 358 | 276 | 280 |
| kernels/cuda/native_mmvq.cu | 2661 | 1493 | 1286 |
| kernels/cuda/iq_kernels.cu | 3686 | 1293 | 3559 |
| kernels/cpu/pool.cpp | 916 | 576 | 540 |
| kernels/cpu/expert.cpp | 633 | 700 | 101 |
| kernels/cpu/native_expert.cpp | 283 | 375 | 430 |
| kernels/cpu/expert_layout.cpp | 478 | 236 | 304 |
| kernels/cpu/q2_avx2.cpp | 271 | 130 | 211 |
| kernels/cpu/iq_avx2.cpp | 523 | 729 | 926 |
| kernels/cpu/iq_avx512.cpp | 482 | 409 | 421 |

Più `quantize.cu` di ggml (M2). Le copie vivono in namespace diversi (`almond::` e `almond::glm::`), quindi non c'è
violazione ODR tra di loro, ma ci sono due pool di thread CPU nello stesso processo (solo uno parte per modello).
Ordine suggerito, dal più vicino al più lontano: `expert.cpp` (101 righe diverse), `quantize.cu`/`almond_mmq`,
`platform/memory.cpp`, poi il pool CPU (che il piano mette per primo, ma diverge per 540 righe).

## Non coperto
- Gare tra il pool CPU degli esperti e gli stream HIP (`src/kernels/cpu/pool.cpp`, `src/ds41/ds41_moe.cpp`,
  `src/glm/core/expert_source.cpp`): non lette per intero.
- Percorsi d'errore di `src/glm/core/glm_fast_path.cu` e `src/glm/program/generate.cpp` (16 000 righe).
- Nessuna misura: la revisione non ha usato le GPU.

## Dopo la revisione: la VRAM che DeepSeek non riprende (2026-10-30, misurato)

Le due 7900 XTX, DeepSeek-V4.1-Flash-1017 RCO, `ALMOND_VRAM_TRACE=1`. Non dipende dalle immagini: qualsiasi prompt
che passa dal percorso batch (oltre ~128 token nuovi) lascia circa 2,5 GiB occupati sulla scheda 0, e la cache
degli esperti si riapre in 15,2-15,4 GiB liberi invece di 17,9 (un prompt di testo da 465 token: 17,90 -> 15,44;
uno con un'immagine, 486 token: 15,24). Le richieste dopo vanno alla stessa velocità (19-21 tok/s), con circa 2,5 GiB
di esperti in meno in VRAM.

Escluso, una prova ciascuno:
- le colonne di `Ds41Dense`: ricostruite (`reset_columns`), nessun GiB torna;
- la pool di ggml-cuda: svuotarla restituisce 0,21 GiB e poi il motore va in crash (illegal memory access); tolto;
- le allocazioni `hipMalloc` di 1 MiB o più (intercettate con LD_PRELOAD): dopo il batch tengono 2.288 MiB in meno,
  non di più; nessuna chiamata a `hipMallocAsync`, `hipExtMallocWithFlags`, `hipMallocManaged`, `hipMemCreate`;
- i moduli di kernel caricati durante il batch (rocBLAS, 4 moduli): ~24 MiB;
- `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1` e `HSA_SCRATCH_SINGLE_LIMIT=256 MiB`: nessun effetto.

Trovato: la memoria torna davvero. La VRAM del processo vista dal kernel (`drm-memory-vram` in `/proc/<pid>/fdinfo`)
scende di 2,1 GB sulla scheda 0 dopo il batch, e `hipFree` restituisce subito (prova a parte). Il punto è quando:
all'apertura della cache dopo il batch ci sono 15,44 GiB liberi, all'inizio della richiesta seguente 2,53 GiB restano
inutilizzati - ggml rende i ~2,4 GiB del batch durante la decodifica, dopo che la cache si è già dimensionata.

Correzione (`ds41_generate.cpp`, `regrow`): la prima richiesta letta a passi dopo un batch, se la scheda 0 ha più di
1 GiB oltre il margine, riapre la cache una volta in quello spazio. `ALMOND_VRAM_REGROW=0` la spegne. Misurato:

| | VRAM libera scheda 0 | decode tok/s |
| --- | ---: | ---: |
| prima di un batch | 0,07-0,59 GiB | 21,6-21,8 |
| dopo un batch, senza la correzione | 2,5-2,6 GiB | 19-21,5 |
| dopo un batch (testo, 465 token) o un'immagine, con la correzione | 0,59-0,60 GiB | 20,7-21,9 |

La riapertura costa 1,3-2,9 s una volta (gli esperti ricaricati dalla RAM).

## ggml_backend_cuda_context's layout (2026-11-02)

ggml-hip is built with `GGML_HIP_GRAPHS`, which gives `ggml_backend_cuda_context` its CUDA-graph members (4,400 bytes);
the MMQ libraries that create one (DeepSeek's `almond_mmq`, GLM's `glm_mmq`, Qwen's `qwen_mmq`) were compiled without
it (4,336 bytes, the pools 64 bytes earlier), while in `almond-engine` the destructor and the other out-of-line members
are ggml-hip's. Now every one is compiled with the ggml build's own setting (`almond_ggml_layout`, the top
`CMakeLists.txt`). Measured on the same binary with only GLM's library left without it: the engine stopped at load
with `malloc(): invalid size (unsorted)`; with it, GLM's PPL over 16 windows of 512 of `long-docs.txt` 13.68 and 13.62
(the engine ended with 0; 13.51 written on 2026-11-02 with an earlier expert profile), DeepSeek through the server 19.0
/ 21.3 / 19.9 tok/s (thinking low, 600 tokens), stopped in 2 s.

