#pragma once

// Internal, disabled-by-default model numerical diagnostic. Not part of the
// supported public inference API. The implementation resides in libllama and
// is used by the local server's opt-in QSA logits/GDN trace only.
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Append raw logical R/S/P recurrent-state snapshots for seq_id to an
// existing JSONL manifest in dir (the caller writes the matching event first).
// The caller must ensure its checkpoint is logically aligned across runs.
// Synchronously reads the selected current cell, including pending rs_idx
// rollback plane. Returns false without modifying inference state on error.
LLAMA_API bool llama_qsa_debug_dump_gdn(
    struct llama_context * ctx,
           llama_seq_id     seq_id,
        const char *        dir,
             unsigned      step,
        const char *        phase,
              llama_pos    position);

#ifdef __cplusplus
}
#endif
