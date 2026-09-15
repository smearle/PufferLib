// T2: peer-channel Double Take intrinsic reward. Included by pufferl.cu after
// PuffeRL when built with --t2 (-DPUFFER_T2). Runtime switch: [t2] enabled.
//
// Every rollout step scores the transition (o_t, a_t) -> o_{t+1} under the
// world model (WM) weights frozen for that rollout:
//   q_ctrl = -log p(tok(o_{t+1}) | h_t, support = prefix + (o_t, PAD))
//   q1     = -log p(tok(o_{t+1}) | h_t, support = prefix + (o_{t+1}, PAD))
//   r_t    = clip(scale * (q_ctrl - q1) / ln 2, 0, scale * cap)
// h_t is a streaming recurrent encoding of the lane's episode, and tok() is
// the env's byte codec (pufferenv.h). The reward lands on the rollout row of
// o_{t+1}, the same row as the env reward for that transition.
//
// The WM trains on a reservoir of complete episodes. Half the rows use the
// causal self prefix as support (the scoring q_ctrl construction); the other
// half use the partner lane's trial of the same reset world (peer support),
// which teaches the support channel what transfers between independent
// experiences of one world. Gradients flow through a truncated window of the
// recurrent encoder; peer summaries are cached at episode end. Both are
// declared execution approximations of the reference full-history model.

#ifndef PUF_T2_TOKENS
#define T2_TOKENS OBS_SIZE
#define T2_GENERIC_CODEC 1
#else
#define T2_TOKENS PUF_T2_TOKENS
#define T2_GENERIC_CODEC 0
#endif
#ifdef PUF_T2_PAIRED
#define T2_PAIRED 1
#else
#define T2_PAIRED 0
#endif
#define T2_VOCAB 256
#define T2_MAX_LAYERS 4
#define T2_LN2 0.69314718056f
#define T2_FXP 16777216.0f

void alloc_register(Allocator* a, Byte* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(unsigned char));
}

enum {
    T2_STAT_REWARD, T2_STAT_NLL_CTRL, T2_STAT_NLL_1, T2_STAT_CAPPED,
    T2_STAT_VALID, T2_STAT_WM_LOSS, T2_STAT_WM_STEPS, T2_STAT_PEER_ROWS,
    T2_STAT_ROWS, T2_STAT_N,
};

struct T2Weights {
    Prec tok_embed;   // (T2_TOKENS * T2_VOCAB, E): per-position token bag
    Prec act_embed;   // (sum_h (act_sizes[h] + 1), EA); each head has a PAD row
    Prec w_in;        // (H, E + EA)
    Prec gru[T2_MAX_LAYERS];  // (3H, H) minGRU per layer
    Prec role;        // (1, H): added to every support summary
    Prec w_h;         // (D, 2H): head input projection of [query; support]
    Prec pos_embed;   // (T2_TOKENS, D): per-position head bias
    Prec w_out;       // (T2_VOCAB, D)
};

// Rollout-time buffers, one set per vec buffer (disjoint agent ranges).
struct T2Rollout {
    Byte tok_cur, tok_next;   // (B, L) tokens of o_t and o_{t+1}
    Byte target2;             // (2B, L) tok_{t+1} for both scoring rows
    Int act;                  // (B, heads) a_t as ints
    Int pad;                  // (B, heads) PAD per head
    Prec x_in;                // (3B, E + EA)
    Prec x;                   // (3B, H) layer input
    Prec combined;            // (3B, 3H)
    Prec out;                 // (3B, H) layer output
    Prec z;                   // (2B, 2H) head input
    Prec u;                   // (2B, D)
    Prec hid;                 // (2B * L, D)
    Prec logits;              // (2B * L, V)
    Float nll_rp;             // (2B, L)
    Float nll;                // (2B,)
    cudaEvent_t start, end;   // scoring time on the worker stream
    int timed;                // an unread start/end pair is pending
};

// Training buffers for one optimizer step of R rows (2R scan sequences).
struct T2Train {
    Int sample_slot, sample_t, sample_mode, sample_partner;  // (R,)
    Byte tok_seq;             // (2R, W, L)
    Int act_seq;              // (2R * W, heads)
    Prec term;                // (2R, W) window resets
    Byte target;              // (R, L)
    Prec init_state;          // (Lg, 2R, H)
    Prec x_in;                // (2R * W, E + EA)
    Prec x[T2_MAX_LAYERS];    // (2R * W, H) layer inputs (saved)
    Prec combined[T2_MAX_LAYERS];  // (2R * W, 3H)
    PrefixScan scan[T2_MAX_LAYERS];
    Prec z;                   // (R, 2H)
    Prec u;                   // (R, D)
    Prec pre;                 // (R * L, D) head pre-activation
    Prec hid;                 // (R * L, D)
    Prec logits;              // (R * L, V)
    Float nll_rp;             // (R, L)
    Float nll;                // (R,)
    Prec d_hid;               // (R * L, D)
    Prec d_u;                 // (R, D)
    Prec d_z;                 // (R, 2H)
    Prec d_out;               // (2R, W, H) gradient into the last layer output
    Prec d_x;                 // (2R * W, H) gradient into a layer input
    Prec d_x_in;              // (2R * W, E + EA)
    Prec grad_next_state;     // (2R, H)
    Long tok_embed_grad_i;    // (T2_TOKENS * T2_VOCAB * E,) fixed point
    Long act_embed_grad_i;    // ((A + 1) * EA,)
    Float partials;           // (256,)
    float* norm;              // device scalar
};

struct T2Grads {
    Prec tok_embed, act_embed, w_in, gru[T2_MAX_LAYERS], role, w_h, pos_embed, w_out;
};

// Active steps beyond the GPU replay-prefix capacity. Raw CPU records allow
// exact current-weight scoring carries without an unbounded GPU state history.
struct T2Overflow {
    unsigned char* tok;
    int* act;
    size_t len, capacity;
};

struct T2 {
    // Config
    int A;                    // total agents (lanes)
    int L;                    // tokens per observation
    int E, EA, H, Lg, D;      // embed, action embed, hidden, layers, head width
    int heads;                // action heads (NUM_ATNS)
    int act_rows;             // sum_h (act_sizes[h] + 1) embedding rows
    int act_sizes[PUF_MAX_DIMS * 4];
    int head_off[PUF_MAX_DIMS * 4];
    int* act_sizes_dev;       // device copies of the two tables
    int* head_off_dev;
    int reencode;             // re-encode lane/reservoir states after WM updates
    int M;                    // max recorded episode length
    int C;                    // reservoir episodes (pair slots = C / 2)
    int R;                    // training rows per step
    int W;                    // BPTT window
    int steps;                // optimizer steps per train epoch
    int horizon, num_buffers, slots;
    float reward_scale, reward_cap, peer_prob;
    float lr, beta1, beta2, eps, max_grad_norm;
    int intrinsic_only;
    long adam_step;
    unsigned int rng;

    // Parameters: one flat buffer (params_alloc) + fp32 master + Adam moments.
    T2Weights w;
    T2Grads g;
    Allocator params_alloc, grads_alloc, buf_alloc;
    Prec param, grad;
    Float master, adam_m, adam_v;

    // Lane state
    Prec state, state_prev;   // (Lg, A, H) recurrent state after (o_t, a_t) and before
    Prec out_last;            // (A, H) last query output (peer summary source)
    Byte ep_tok;              // (A, M, L)
    Int ep_act;               // (A, M, heads)
    Prec ep_state;            // (A, M, Lg, H) state after each recorded step
    Int ep_len;               // (A,) device copy of the recorded length
    int* host_len;            // mirror
    T2Overflow* overflow;     // (A,) CPU suffix after the first M recorded steps
    int* host_episode;        // resets seen per lane (episode index)
    unsigned char* host_tok;  // pinned (A, L) tokens written by env workers
    Prec rewards;             // (slots * T, A) intrinsic rewards per rollout row
    T2Rollout* roll;          // [num_buffers]

    // Reservoir (device rows + host index). Slots 2j, 2j+1 hold pair j.
    Byte res_tok;             // (C, M, L)
    Int res_act;              // (C, M, heads)
    Prec res_state;           // (C, M, Lg, H)
    Prec res_final;           // (C, H) partner summary
    int* res_len;             // host (C,) 0 = empty
    int* res_pair;            // host (C,) pair id
    int* res_episode;         // host (C,) episode index
    pthread_mutex_t reservoir_mutex; // terminal workers share reservoir metadata/copies
    long pairs_seen;
    long episodes_complete;

    // Training
    T2Train tr;
    float* stats;             // device (T2_STAT_N,)
    float train_ms;           // WM training wall time since the last log
    float* rollout_ms;        // [num_buffers] scoring GPU time since the last log
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ float t2_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

__device__ __forceinline__ float t2_gelu_grad(float x) {
    float cdf = 0.5f * (1.0f + erff(x * 0.70710678f));
    float pdf = 0.39894228f * expf(-0.5f * x * x);
    return cdf + x * pdf;
}

// x_in[row] = [sum_p tok_embed[p, tok[row, p]] ; sum_h act_embed[off_h + act[row, h]]]
// with act (rows, heads); a head's PAD row is off_h + act_sizes[h].
__global__ void t2_embed_kernel(precision_t* __restrict__ x_in,
        const unsigned char* __restrict__ tok, const int* __restrict__ act,
        const precision_t* __restrict__ tok_embed,
        const precision_t* __restrict__ act_embed, const int* __restrict__ head_off,
        int rows, int L, int E, int EA, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int width = E + EA;
    if (idx >= rows * width) {
        return;
    }
    int row = idx / width, col = idx % width;
    float v = 0.0f;
    if (col < E) {
        const unsigned char* t = tok + (long)row * L;
        for (int p = 0; p < L; p++) {
            v += to_float(tok_embed[((long)p * T2_VOCAB + t[p]) * E + col]);
        }
    } else {
        for (int h = 0; h < heads; h++) {
            v += to_float(act_embed[(long)(head_off[h] + act[(long)row * heads + h]) * EA
                + (col - E)]);
        }
    }
    x_in[idx] = from_float(v);
}

// Token embedding gradient, one block per (position, row chunk, 32 columns):
// each thread owns one column of a shared [vocab, 32] table, so the row loop
// needs no atomics; the table is flushed with fixed-point integer atomics
// (associative, so the result does not depend on block order).
#define T2_EMBED_CHUNKS 16
#define T2_EMBED_COLS 32
__global__ void t2_tok_embed_backward_kernel(long long* __restrict__ tok_grad_i,
        const precision_t* __restrict__ d_x_in,
        const unsigned char* __restrict__ tok, int rows, int L, int E, int EA) {
    __shared__ float acc[T2_VOCAB * T2_EMBED_COLS];
    int p = blockIdx.x, chunk = blockIdx.y, e = blockIdx.z * T2_EMBED_COLS + threadIdx.x;
    int c = threadIdx.x;
    for (int i = c; i < T2_VOCAB * T2_EMBED_COLS; i += T2_EMBED_COLS) {
        acc[i] = 0.0f;
    }
    __syncthreads();
    int per = (rows + T2_EMBED_CHUNKS - 1) / T2_EMBED_CHUNKS;
    int r0 = chunk * per, r1 = min(rows, r0 + per);
    int width = E + EA;
    for (int row = r0; row < r1; row++) {
        int v = tok[(long)row * L + p];
        acc[v * T2_EMBED_COLS + c] += to_float(d_x_in[(long)row * width + e]);
    }
    __syncthreads();
    for (int v = 0; v < T2_VOCAB; v++) {
        float g = acc[v * T2_EMBED_COLS + c];
        if (g != 0.0f) {
            atomicAdd((unsigned long long*)&tok_grad_i[((long)p * T2_VOCAB + v) * E + e],
                (unsigned long long)__float2ll_rn(g * T2_FXP));
        }
    }
}

__global__ void t2_act_embed_backward_kernel(long long* __restrict__ act_grad_i,
        const precision_t* __restrict__ d_x_in, const int* __restrict__ act,
        const int* __restrict__ head_off, int rows, int E, int EA, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * EA) {
        return;
    }
    int row = idx / EA, col = idx % EA;
    int width = E + EA;
    long long g = __float2ll_rn(to_float(d_x_in[(long)row * width + E + col]) * T2_FXP);
    if (g == 0) {
        return;
    }
    for (int h = 0; h < heads; h++) {
        long at = (long)(head_off[h] + act[(long)row * heads + h]) * EA + col;
        atomicAdd((unsigned long long*)&act_grad_i[at], (unsigned long long)g);
    }
}

__global__ void t2_fxp_to_precision_kernel(precision_t* __restrict__ dst,
        const long long* __restrict__ src, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)((double)src[idx] * (1.0 / (double)T2_FXP)));
    }
}

// One minGRU layer step for three groups sharing lane indices:
//   group 0 (query, action a_t) from the lane state S_{t-1}, writes S_t;
//   group 1 (ctrl support, PAD) from S_{t-1}: the prefix through
//           (o_{t-1}, a_{t-1}) then (o_t, PAD), the literal control prefix;
//   group 2 (q1 support, PAD) from S_t: the prefix through (o_t, a_t) then
//           (o_{t+1}, PAD).
// state_prev keeps S_{t-1} for audits. Called once per layer; combined/x/out
// hold 3B rows, states hold this layer's (B, H) slice. Terminal resets are
// applied afterwards.
__global__ void t2_gate3_kernel(precision_t* __restrict__ out,
        precision_t* __restrict__ state, precision_t* __restrict__ state_prev,
        const precision_t* __restrict__ combined, const precision_t* __restrict__ x,
        int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H, h = idx % H;
    float h_cur = to_float(state[idx]);
    float h_in[3] = {h_cur, h_cur, 0.0f};
    float h_new = 0.0f;
    for (int g = 0; g < 3; g++) {
        int row = g * B + b;
        int cb = row * 3 * H;
        float hidden = to_float(combined[cb + h]);
        float gate = to_float(combined[cb + H + h]);
        float proj = to_float(combined[cb + 2 * H + h]);
        float xv = to_float(x[row * H + h]);
        float hi = g == 2 ? h_new : h_in[g];
        float z = sigmoid(gate);
        float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
        float ho = lerp(hi, h_tilde, z);
        float s = sigmoid(proj);
        out[row * H + h] = from_float(s * ho + (1.0f - s) * xv);
        if (g == 0) {
            // The stored state is rounded to precision_t, as in the train scan.
            h_new = to_float(from_float(ho));
        }
    }
    state_prev[idx] = from_float(h_cur);
    state[idx] = from_float(h_new);
}

// A terminal successor starts a new episode: both lane states restart from
// zero for one (B, H) layer slice.
__global__ void t2_reset_states_kernel(precision_t* __restrict__ state,
        precision_t* __restrict__ state_prev, const float* __restrict__ terminals,
        int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    if (terminals[idx / H] != 0.0f) {
        state[idx] = from_float(0.0f);
        state_prev[idx] = from_float(0.0f);
    }
}

// z = [query ; support + role] for the two scoring rows of every lane
// (rows [0, B): ctrl support, rows [B, 2B): q1 support).
__global__ void t2_score_z_kernel(precision_t* __restrict__ z,
        const precision_t* __restrict__ out, const precision_t* __restrict__ role,
        int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * B * 2 * H) {
        return;
    }
    int row = idx / (2 * H), col = idx % (2 * H);
    int b = row % B, which = row / B;  // 0 ctrl, 1 q1
    float v;
    if (col < H) {
        v = to_float(out[b * H + col]);
    } else {
        int h = col - H;
        v = to_float(out[((which + 1) * B + b) * H + h]) + to_float(role[h]);
    }
    z[idx] = from_float(v);
}

// Training head input: z[r] = [query_out[r, W-1] ; support + role] where the
// support is the self scan output (rows R..2R-1) or a cached peer summary.
__global__ void t2_train_z_kernel(precision_t* __restrict__ z,
        const precision_t* __restrict__ out, const precision_t* __restrict__ role,
        const precision_t* __restrict__ res_final, const int* __restrict__ mode,
        const int* __restrict__ partner, int R, int W, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * 2 * H) {
        return;
    }
    int r = idx / (2 * H), col = idx % (2 * H);
    float v;
    if (col < H) {
        v = to_float(out[((long)r * W + W - 1) * H + col]);
    } else {
        int h = col - H;
        if (mode[r]) {
            v = to_float(res_final[(long)partner[r] * H + h]);
        } else {
            v = to_float(out[((long)(R + r) * W + W - 1) * H + h]);
        }
        v += to_float(role[h]);
    }
    z[idx] = from_float(v);
}

// d_out (2R, W, H): only the last window position feeds the head.
__global__ void t2_train_dz_kernel(precision_t* __restrict__ d_out,
        precision_t* __restrict__ d_role_partials,
        const precision_t* __restrict__ d_z, const int* __restrict__ mode,
        int R, int W, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * 2 * H) {
        return;
    }
    int r = idx / (2 * H), col = idx % (2 * H);
    float g = to_float(d_z[idx]);
    if (col < H) {
        d_out[((long)r * W + W - 1) * H + col] = from_float(g);
    } else {
        int h = col - H;
        d_out[((long)(R + r) * W + W - 1) * H + h] = from_float(mode[r] ? 0.0f : g);
        d_role_partials[(long)r * H + h] = from_float(g);
    }
}

__global__ void t2_zero_prec_kernel(precision_t* p, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx] = from_float(0.0f);
    }
}

// hid[r, p] = gelu(u[r] + pos[p]); pre saved for the backward pass.
__global__ void t2_head_hidden_kernel(precision_t* __restrict__ hid,
        precision_t* __restrict__ pre, const precision_t* __restrict__ u,
        const precision_t* __restrict__ pos, int R, int L, int D) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)R * L * D) {
        return;
    }
    int d = idx % D;
    int p = (idx / D) % L;
    int r = idx / ((long)L * D);
    float v = to_float(u[(long)r * D + d]) + to_float(pos[(long)p * D + d]);
    if (pre) {
        pre[idx] = from_float(v);
    }
    hid[idx] = from_float(t2_gelu(v));
}

// One warp per (row, position): log-softmax over the byte vocabulary.
// nll_rp = lse - logit[target]; grad = (softmax - onehot) * gscale.
__global__ void t2_ce_kernel(float* __restrict__ nll_rp,
        precision_t* __restrict__ grad, const precision_t* __restrict__ logits,
        const unsigned char* __restrict__ target, long rows, float gscale) {
    long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp >= rows) {
        return;
    }
    const precision_t* lg = logits + warp * T2_VOCAB;
    float v[T2_VOCAB / 32];
    float m = -INFINITY;
    #pragma unroll
    for (int i = 0; i < T2_VOCAB / 32; i++) {
        v[i] = to_float(lg[lane + 32 * i]);
        m = fmaxf(m, v[i]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < T2_VOCAB / 32; i++) {
        s += expf(v[i] - m);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, off);
    }
    float lse = m + logf(s);
    int t = target[warp];  // warp-uniform
    float lt = 0.0f;
    #pragma unroll
    for (int i = 0; i < T2_VOCAB / 32; i++) {
        float cand = __shfl_sync(0xffffffff, v[i], t % 32);
        if (t / 32 == i) {
            lt = cand;
        }
    }
    if (lane == 0) {
        nll_rp[warp] = lse - lt;
    }
    if (grad) {
        #pragma unroll
        for (int i = 0; i < T2_VOCAB / 32; i++) {
            int c = lane + 32 * i;
            float p = expf(v[i] - lse);
            grad[warp * T2_VOCAB + c] = from_float((p - (c == t ? 1.0f : 0.0f)) * gscale);
        }
    }
}

__global__ void t2_sum_rows_kernel(float* __restrict__ nll,
        const float* __restrict__ nll_rp, int R, int L) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) {
        return;
    }
    float s = 0.0f;
    for (int p = 0; p < L; p++) {
        s += nll_rp[(long)r * L + p];
    }
    nll[r] = s;
}

// d_pre = d_hid * gelu'(pre); d_u[r] = sum_p d_pre; d_pos[p] = sum_r d_pre.
__global__ void t2_head_backward_kernel(precision_t* __restrict__ d_pre,
        const precision_t* __restrict__ d_hid, const precision_t* __restrict__ pre,
        long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_pre[idx] = from_float(to_float(d_hid[idx]) * t2_gelu_grad(to_float(pre[idx])));
    }
}

__global__ void t2_reduce_positions_kernel(precision_t* __restrict__ d_u,
        const precision_t* __restrict__ d_pre, int R, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * D) {
        return;
    }
    int r = idx / D, d = idx % D;
    float s = 0.0f;
    for (int p = 0; p < L; p++) {
        s += to_float(d_pre[((long)r * L + p) * D + d]);
    }
    d_u[idx] = from_float(s);
}

__global__ void t2_reduce_rows_kernel(precision_t* __restrict__ d_pos,
        const precision_t* __restrict__ d_pre, int R, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * D) {
        return;
    }
    int p = idx / D, d = idx % D;
    float s = 0.0f;
    for (int r = 0; r < R; r++) {
        s += to_float(d_pre[((long)r * L + p) * D + d]);
    }
    d_pos[idx] = from_float(s);
}

__global__ void t2_reduce_role_kernel(precision_t* __restrict__ d_role,
        const precision_t* __restrict__ partials, int R, int H) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    float s = 0.0f;
    for (int r = 0; r < R; r++) {
        s += to_float(partials[(long)r * H + h]);
    }
    d_role[h] = from_float(s);
}

// Reward for the transition into o_{t+1}, written on that rollout row, plus
// running statistics. Rows whose successor starts a new episode get no bonus.
__global__ void t2_reward_kernel(precision_t* __restrict__ reward_row,
        float* __restrict__ stats, const float* __restrict__ nll,
        const float* __restrict__ terminals, int B, float scale, float cap) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    float gain = (nll[b] - nll[B + b]) / T2_LN2;
    float r = fminf(fmaxf(gain, 0.0f), cap);
    int valid = terminals[b] == 0.0f;
    if (reward_row) {
        reward_row[b] = from_float(valid ? scale * r : 0.0f);
    }
    if (valid) {
        atomicAdd(&stats[T2_STAT_REWARD], scale * r);
        atomicAdd(&stats[T2_STAT_NLL_CTRL], nll[b]);
        atomicAdd(&stats[T2_STAT_NLL_1], nll[B + b]);
        atomicAdd(&stats[T2_STAT_CAPPED], gain >= cap ? 1.0f : 0.0f);
        atomicAdd(&stats[T2_STAT_VALID], 1.0f);
    }
}

// Append (tok_t, a_t, state_t) to the lane record; a terminal successor ends
// the episode (host mirrors the same counters from the env terminals).
__global__ void t2_record_kernel(unsigned char* __restrict__ ep_tok,
        int* __restrict__ ep_act, precision_t* __restrict__ ep_state,
        int* __restrict__ ep_len, precision_t* __restrict__ out_last,
        const unsigned char* __restrict__ tok, const int* __restrict__ act,
        const precision_t* __restrict__ state, const precision_t* __restrict__ out,
        const float* __restrict__ terminals, int agent0, int B, int A,
        int L, int M, int Lg, int H, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per_lane = L + Lg * H;
    if (idx >= B * per_lane) {
        return;
    }
    int b = idx / per_lane, k = idx % per_lane;
    int a = agent0 + b;
    int len = ep_len[a];
    if (len < M) {
        if (k < L) {
            ep_tok[((long)a * M + len) * L + k] = tok[(long)b * L + k];
        } else {
            int l = (k - L) / H, h = (k - L) % H;
            // state is (Lg, A, H) for the whole vec; this buffer's lanes are agent0+b.
            ep_state[(((long)a * M + len) * Lg + l) * H + h] =
                state[((long)l * A + a) * H + h];
            if (l == Lg - 1) {
                out_last[(long)a * H + h] = out[(long)b * H + h];
            }
        }
        if (k < heads) {
            ep_act[((long)a * M + len) * heads + k] = act[(long)b * heads + k];
        }
    }
}

// Kernel boundary: all record writers must finish reading the old length
// before any thread publishes the next length (warps/blocks are independent).
__global__ void t2_advance_record_lengths(int* ep_len,const float* terminals,
        int start,int B,int M) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;
    int a=start+b,len=ep_len[a];
    ep_len[a]=terminals[b]!=0.0f?0:min(len+1,M);
}

// Rollout actions (B, heads) as ints; heads a verb does not consume read as
// PAD so unused arguments do not condition the world model.
__global__ void t2_actions_kernel(int* __restrict__ dst, const float* __restrict__ src,
        const int* __restrict__ act_sizes, int B, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * heads) {
        return;
    }
    int b = idx / heads, h = idx % heads;
    int a = (int)src[idx];
#ifdef PUFFER_NETHACK
    if (!nethack_head_used((int)src[(long)b * heads], h)) {
        a = act_sizes[h];
    }
#endif
    dst[idx] = a;
}

// Gather training windows. Rows [0, R): query windows ending at (slot, t);
// rows [R, 2R): the same windows with PAD at the last step (self support).
// Positions before the episode start are padding with a reset flag at the
// first real step; otherwise the window starts from the recorded state.
__global__ void t2_gather_kernel(unsigned char* __restrict__ tok_seq,
        int* __restrict__ act_seq, precision_t* __restrict__ term,
        unsigned char* __restrict__ target, precision_t* __restrict__ init_state,
        const int* __restrict__ slot, const int* __restrict__ t_at,
        const unsigned char* __restrict__ res_tok, const int* __restrict__ res_act,
        const precision_t* __restrict__ res_state, const int* __restrict__ act_sizes,
        int R, int W, int L, int M, int Lg, int H, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * R * W) {
        return;
    }
    int row = idx / W, w = idx % W;
    int r = row % R;
    int j = slot[r], t = t_at[r];
    int src = t - W + 1 + w;
    unsigned char* dst = tok_seq + (long)idx * L;
    if (src < 0) {
        for (int p = 0; p < L; p++) {
            dst[p] = 0;
        }
        for (int h = 0; h < heads; h++) {
            act_seq[(long)idx * heads + h] = act_sizes[h];
        }
        term[idx] = from_float(0.0f);
    } else {
        const unsigned char* s = res_tok + ((long)j * M + src) * L;
        for (int p = 0; p < L; p++) {
            dst[p] = s[p];
        }
        int pad = row >= R && w == W - 1;
        for (int h = 0; h < heads; h++) {
            act_seq[(long)idx * heads + h] = pad ? act_sizes[h]
                : res_act[((long)j * M + src) * heads + h];
        }
        term[idx] = from_float(src == 0 ? 1.0f : 0.0f);
    }
    if (w == 0) {
        int start = t - W;  // state after step t-W seeds the window
        for (int l = 0; l < Lg; l++) {
            for (int h = 0; h < H; h++) {
                float v = 0.0f;
                if (start >= 0) {
                    v = to_float(res_state[(((long)j * M + start) * Lg + l) * H + h]);
                }
                init_state[((long)l * 2 * R + row) * H + h] = from_float(v);
            }
        }
        if (row < R) {
            const unsigned char* s = res_tok + ((long)j * M + t + 1) * L;
            for (int p = 0; p < L; p++) {
                target[(long)r * L + p] = s[p];
            }
        }
    }
}

// Adam on fp32 master weights with global-norm clipping; grads are Prec.
__global__ void t2_adam_kernel(float* __restrict__ w, float* __restrict__ m,
        float* __restrict__ v, const precision_t* __restrict__ g,
        const float* __restrict__ sum_sq, float max_norm, float lr,
        float b1, float b2, float eps, float bc1, float bc2, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float clip = fminf(max_norm / (sqrtf(*sum_sq) + 1e-6f), 1.0f);
    float gi = to_float(g[idx]) * clip;
    float mi = b1 * m[idx] + (1.0f - b1) * gi;
    float vi = b2 * v[idx] + (1.0f - b2) * gi * gi;
    m[idx] = mi;
    v[idx] = vi;
    w[idx] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

// Add (or substitute) the intrinsic rewards into the (B, T) train view.
__global__ void t2_apply_rewards_kernel(precision_t* __restrict__ rewards,
        const precision_t* __restrict__ intrinsic, int B, int T, int replace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) {
        return;
    }
    int b = idx / T, t = idx % T;
    float r = to_float(intrinsic[(long)t * B + b]);
    rewards[idx] = from_float(replace ? r : to_float(rewards[idx]) + r);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

static void t2_encode_tokens(T2* t2, Env* envs, int env_start, int env_count) {
#if T2_GENERIC_CODEC
    for (int i = env_start; i < env_start + env_count; i++) {
        obs_t* obs = envs[i].agents[0].observations;
        unsigned char* out = t2->host_tok + (long)i * t2->L;
        for (int k = 0; k < t2->L; k++) {
            float v = (float)obs[k];
            if (sizeof(obs_t) != 1) {
                v = v * 255.0f + 0.5f;
            }
            out[k] = (unsigned char)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }
    }
#else
    for (int i = env_start; i < env_start + env_count; i++) {
        puf_t2_tokens(&envs[i], t2->host_tok + (long)i * t2->L);
    }
#endif
}

static unsigned int t2_rand(T2* t2) {
    return (unsigned int)(puf_t2_mix(++t2->rng) >> 33);
}

// Called after actions reach the CPU and before the environment can mutate
// them or replace the current observation. Each worker owns disjoint lanes.
static void t2_capture_overflow(T2* t2, const float* actions, int start, int count) {
    if (!t2->reencode) return;
#ifdef PUFFER_NETHACK
    int verbs = 0, heads = 0;
    const signed char* consumed = env_head_consume_map(&verbs, &heads);
    assert(!consumed || (heads == t2->heads && verbs == t2->act_sizes[0]));
#endif
    for (int a = start; a < start + count; a++) {
        if (t2->host_len[a] < t2->M) continue;
        T2Overflow* tail = &t2->overflow[a];
        if (tail->len == tail->capacity) {
            size_t capacity = tail->capacity ? 2 * tail->capacity : (size_t)t2->M;
            assert(capacity > tail->capacity
                && capacity <= SIZE_MAX / (size_t)t2->L
                && capacity <= SIZE_MAX / ((size_t)t2->heads * sizeof(int))
                && "T2 active history capacity overflow");
            unsigned char* tok = (unsigned char*)realloc(tail->tok, capacity * t2->L);
            int* act = (int*)realloc(tail->act, capacity * t2->heads * sizeof(int));
            assert(tok && act && "T2 active history allocation failed");
            tail->tok = tok;
            tail->act = act;
            tail->capacity = capacity;
        }
        memcpy(tail->tok + tail->len * t2->L,
            t2->host_tok + (long)a * t2->L, t2->L);
        for (int h = 0; h < t2->heads; h++) {
            int action = (int)actions[(long)a * t2->heads + h];
#ifdef PUFFER_NETHACK
            int verb = (int)actions[(long)a * t2->heads];
            if (h != 0 && consumed && !consumed[verb * heads + h]) {
                action = t2->act_sizes[h];
            }
#endif
            tail->act[tail->len * t2->heads + h] = action;
        }
        tail->len++;
    }
}

static double t2_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Head forward over R rows of z: returns per-row summed NLL in nll (R,).
static void t2_head_forward(T2* t2, Prec z, Prec u, Prec hid, Prec pre, Prec logits,
        Float nll_rp, Float nll, const unsigned char* target, Prec grad_logits,
        int R, float gscale, cudaStream_t stream) {
    int L = t2->L, D = t2->D;
    puf_mm(&z, &t2->w.w_h, &u, stream);
    long n = (long)R * L * D;
    t2_head_hidden_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        hid.data, pre.data, u.data, t2->w.pos_embed.data, R, L, D);
    puf_mm(&hid, &t2->w.w_out, &logits, stream);
    long rows = (long)R * L;
    t2_ce_kernel<<<grid_size(rows * 32), BLOCK_SIZE, 0, stream>>>(
        nll_rp.data, grad_logits.data, logits.data, target, rows, gscale);
    t2_sum_rows_kernel<<<grid_size(R), BLOCK_SIZE, 0, stream>>>(
        nll.data, nll_rp.data, R, L);
}

// Embed tokens/actions and project: x = [bag(tok); act] @ w_in^T.
static void t2_input_forward(T2* t2, Prec x_in, Prec x, const unsigned char* tok,
        const int* act, int rows, cudaStream_t stream) {
    t2_embed_kernel<<<grid_size(rows * (t2->E + t2->EA)), BLOCK_SIZE, 0, stream>>>(
        x_in.data, tok, act, t2->w.tok_embed.data, t2->w.act_embed.data,
        t2->head_off_dev, rows, t2->L, t2->E, t2->EA, t2->heads);
    puf_mm(&x_in, &t2->w.w_in, &x, stream);
}

// Per-step scoring for one vec buffer: query GRU step, two support steps,
// two likelihoods, reward on the row of o_{t+1}, and the episode record.
void t2_rollout_step(PuffeRL* p, int buf, int t, cudaStream_t stream) {
    T2* t2 = p->t2;
    T2Rollout* r = &t2->roll[buf];
    int B = t2->A / t2->num_buffers;
    int agent0 = buf * B;
    int H = t2->H, Lg = t2->Lg;
    int T = t2->horizon;
    RolloutBuf rollouts = p->rollouts;
    if (p->hypers.async) {
        rollouts = rollout_time_view(&p->rollouts, p->write_slot * T, T);
    }
    // Tokens of o_{t+1} (just uploaded by the worker) and a_t as ints.
    cudaMemcpyAsync(r->tok_next.data, t2->host_tok + (long)agent0 * t2->L,
        (size_t)B * t2->L, cudaMemcpyHostToDevice, stream);
    int heads = t2->heads;
    const float* act_t = rollouts.actions.data + ((long)t * t2->A + agent0) * heads;
    t2_actions_kernel<<<grid_size(B * heads), BLOCK_SIZE, 0, stream>>>(
        r->act.data, act_t, t2->act_sizes_dev, B, heads);
    // Embedding rows: [query: (tok_t, a_t)] [ctrl: (tok_t, PAD)] [q1: (tok_{t+1}, PAD)].
    int width = t2->E + t2->EA;
    const unsigned char* toks[3] = {r->tok_cur.data, r->tok_cur.data, r->tok_next.data};
    const int* acts[3] = {r->act.data, r->pad.data, r->pad.data};
    for (int g = 0; g < 3; g++) {
        t2_embed_kernel<<<grid_size(B * width), BLOCK_SIZE, 0, stream>>>(
            r->x_in.data + (long)g * B * width, toks[g], acts[g], t2->w.tok_embed.data,
            t2->w.act_embed.data, t2->head_off_dev, B, t2->L, t2->E, t2->EA, heads);
    }
    puf_mm(&r->x_in, &t2->w.w_in, &r->x, stream);
    const float* term = p->env.terminals.data + agent0;
    Prec x = r->x;
    for (int l = 0; l < Lg; l++) {
        puf_mm(&x, &t2->w.gru[l], &r->combined, stream);
        precision_t* st = t2->state.data + ((long)l * t2->A + agent0) * H;
        precision_t* stp = t2->state_prev.data + ((long)l * t2->A + agent0) * H;
        t2_gate3_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            r->out.data, st, stp, r->combined.data, x.data, B, H);
        if (l + 1 < Lg) {
            // Next layer reads this layer's output as its input.
            puf_copy(&r->x, &r->out, stream);
        }
    }
    // Record (tok_t, a_t, state after (o_t, a_t)) before the terminal reset.
    t2_record_kernel<<<grid_size(B * (t2->L + Lg * H)), BLOCK_SIZE, 0, stream>>>(
        t2->ep_tok.data, t2->ep_act.data, t2->ep_state.data, t2->ep_len.data,
        t2->out_last.data, r->tok_cur.data, r->act.data, t2->state.data, r->out.data,
        term, agent0, B, t2->A, t2->L, t2->M, Lg, H, heads);
    t2_advance_record_lengths<<<grid_size(B),BLOCK_SIZE,0,stream>>>(t2->ep_len.data,term,agent0,B,t2->M);
    for (int l = 0; l < Lg; l++) {
        t2_reset_states_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            t2->state.data + ((long)l * t2->A + agent0) * H,
            t2->state_prev.data + ((long)l * t2->A + agent0) * H, term, B, H);
    }
    // Score: rows [0,B) ctrl, [B,2B) q1, both targeting tok_{t+1}.
    t2_score_z_kernel<<<grid_size(2 * B * 2 * H), BLOCK_SIZE, 0, stream>>>(
        r->z.data, r->out.data, t2->w.role.data, B, H);
    cudaMemcpyAsync(r->target2.data, r->tok_next.data, (size_t)B * t2->L,
        cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(r->target2.data + (long)B * t2->L, r->tok_next.data,
        (size_t)B * t2->L, cudaMemcpyDeviceToDevice, stream);
    Prec no_pre = {};
    Prec no_grad = {};
    t2_head_forward(t2, r->z, r->u, r->hid, no_pre, r->logits, r->nll_rp, r->nll,
        r->target2.data, no_grad, 2 * B, 0.0f, stream);
    precision_t* reward_row = NULL;
    if (t + 1 < T) {
        int row = (p->hypers.async ? p->write_slot * T : 0) + t + 1;
        reward_row = t2->rewards.data + (long)row * t2->A + agent0;
    }
    t2_reward_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        reward_row, t2->stats, r->nll.data, term, B, t2->reward_scale, t2->reward_cap);
    // o_{t+1} becomes the current observation.
    Byte tmp = r->tok_cur;
    r->tok_cur = r->tok_next;
    r->tok_next = tmp;
}

// Host bookkeeping after the env step: complete episodes enter the reservoir.
// Called by the worker after t2_rollout_step on the same stream.
void t2_episode_bookkeeping(PuffeRL* p, int buf, cudaStream_t stream) {
    T2* t2 = p->t2;
    VecEnv* vec = p->vec;
    int B = t2->A / t2->num_buffers;
    int agent0 = buf * B;
    bool terminal=false;
    for(int a=agent0;a<agent0+B;a++)terminal|=vec->terminals[a]!=0.0f;
    if(!terminal) {
        // Lane counters are disjoint across workers; ordinary steps need no lock.
        for(int a=agent0;a<agent0+B;a++)t2->host_len[a]=min(t2->host_len[a]+1,t2->M);
        return;
    }
    pthread_mutex_lock(&t2->reservoir_mutex);
    for (int a = agent0; a < agent0 + B; a++) {
        int len = t2->host_len[a] < t2->M ? t2->host_len[a] + 1 : t2->M;
        if (vec->terminals[a] == 0.0f) {
            t2->host_len[a] = len;
            continue;
        }
        int episode = t2->host_episode[a]++;
        t2->host_len[a] = 0;
        free(t2->overflow[a].tok);
        free(t2->overflow[a].act);
        t2->overflow[a] = (T2Overflow){};
        t2->episodes_complete++;
        if (len < 2) {
            continue;  // needs at least one successor
        }
        int pair = a >> 1;
        int dest = -1;
        for (int j = 0; j < t2->C; j += 2) {
            if (t2->res_len[j] + t2->res_len[j + 1] > 0
                    && t2->res_pair[j] == pair && t2->res_episode[j] == episode) {
                dest = j;
                break;
            }
        }
        if (dest < 0) {
            long seen = t2->pairs_seen++;
            long half = t2->C / 2;
            long pick = seen < half ? seen : (long)(t2_rand(t2) % (unsigned)(seen + 1));
            if (pick >= half) {
                continue;
            }
            dest = 2 * (int)pick;
            t2->res_len[dest] = 0;
            t2->res_len[dest + 1] = 0;
            t2->res_pair[dest] = pair;
            t2->res_episode[dest] = episode;
            t2->res_pair[dest + 1] = pair;
            t2->res_episode[dest + 1] = episode;
        }
        int j = dest + (a & 1);
        t2->res_len[j] = len;
        cudaMemcpyAsync(t2->res_tok.data + (long)j * t2->M * t2->L,
            t2->ep_tok.data + (long)a * t2->M * t2->L, (size_t)len * t2->L,
            cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(t2->res_act.data + (long)j * t2->M * t2->heads,
            t2->ep_act.data + (long)a * t2->M * t2->heads,
            (size_t)len * t2->heads * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(t2->res_state.data + (long)j * t2->M * t2->Lg * t2->H,
            t2->ep_state.data + (long)a * t2->M * t2->Lg * t2->H,
            (size_t)len * t2->Lg * t2->H * sizeof(precision_t),
            cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(t2->res_final.data + (long)j * t2->H,
            t2->out_last.data + (long)a * t2->H, (size_t)t2->H * sizeof(precision_t),
            cudaMemcpyDeviceToDevice, stream);
    }
    // Publish metadata and device contents together. A different worker must
    // not recycle the destination while this stream still copies its records.
    cudaStreamSynchronize(stream);
    pthread_mutex_unlock(&t2->reservoir_mutex);
}

void t2_apply_rewards(PuffeRL* p, RolloutBuf* train_view, int slot,
        cudaStream_t stream) {
    T2* t2 = p->t2;
    int T = t2->horizon, B = t2->A;
    Prec view = puf_time_view(t2->rewards, slot * T, T);
    t2_apply_rewards_kernel<<<grid_size(B * T), BLOCK_SIZE, 0, stream>>>(
        train_view->rewards.data, view.data, B, T, t2->intrinsic_only);
}

// Sample R training rows (slot, t, mode, partner) from resident complete
// episodes into the device sample buffers. Returns the number of peer rows,
// or -1 while the reservoir is still empty.
static int t2_sample(T2* t2, cudaStream_t stream) {
    T2Train* tr = &t2->tr;
    int R = t2->R;
    int candidates = 0;
    for (int j = 0; j < t2->C; j++) {
        candidates += t2->res_len[j] >= 2;
    }
    if (candidates == 0) {
        return -1;
    }
    int* slot = (int*)malloc(R * sizeof(int));
    int* t_at = (int*)malloc(R * sizeof(int));
    int* mode = (int*)malloc(R * sizeof(int));
    int* partner = (int*)malloc(R * sizeof(int));
    int peer_rows = 0;
    for (int r = 0; r < R; r++) {
        int j;
        do {
            j = t2_rand(t2) % t2->C;
        } while (t2->res_len[j] < 2);
        int len = t2->res_len[j];
        slot[r] = j;
        t_at[r] = t2_rand(t2) % (len - 1);
        int sib = j ^ 1;
        int has_peer = T2_PAIRED && t2->res_len[sib] > 0
            && t2->res_pair[sib] == t2->res_pair[j]
            && t2->res_episode[sib] == t2->res_episode[j];
        unsigned coin = t2_rand(t2) % 1000000;
        mode[r] = has_peer && coin < (unsigned)(t2->peer_prob * 1e6f);
        partner[r] = mode[r] ? sib : 0;
        peer_rows += mode[r];
    }
    cudaMemcpyAsync(tr->sample_slot.data, slot, R * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tr->sample_t.data, t_at, R * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tr->sample_mode.data, mode, R * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(tr->sample_partner.data, partner, R * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    free(slot);
    free(t_at);
    free(mode);
    free(partner);
    return peer_rows;
}

// Forward and backward over the sampled rows; gradients land in t2->grad.
// Returns the mean per-row NLL (nats, summed over token positions).
static float t2_forward_backward(T2* t2, cudaStream_t stream) {
    T2Train* tr = &t2->tr;
    int R = t2->R, W = t2->W, H = t2->H, Lg = t2->Lg, L = t2->L;
    int rows = 2 * R * W;
    t2_gather_kernel<<<grid_size(2 * R * W), BLOCK_SIZE, 0, stream>>>(
        tr->tok_seq.data, tr->act_seq.data, tr->term.data, tr->target.data,
        tr->init_state.data, tr->sample_slot.data, tr->sample_t.data,
        t2->res_tok.data, t2->res_act.data, t2->res_state.data, t2->act_sizes_dev,
        R, W, L, t2->M, Lg, H, t2->heads);

    // Forward: embed -> w_in -> minGRU scans (2R sequences of W) -> head.
    t2_input_forward(t2, tr->x_in, tr->x[0], tr->tok_seq.data, tr->act_seq.data,
        rows, stream);
    for (int l = 0; l < Lg; l++) {
        puf_mm(&tr->x[l], &t2->w.gru[l], &tr->combined[l], stream);
        PrefixScan& scan = tr->scan[l];
        scan.combined_ptr = tr->combined[l].data;
        scan.state_ptr = tr->init_state.data + (long)l * 2 * R * H;
        scan.input_ptr = tr->x[l].data;
        scan.terminals_ptr = tr->term.data;
        mingru_scan_forward<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream>>>(scan);
        if (l + 1 < Lg) {
            puf_copy(&tr->x[l + 1], &scan.out, stream);
        }
    }
    Prec last_out = tr->scan[Lg - 1].out;
    t2_train_z_kernel<<<grid_size(R * 2 * H), BLOCK_SIZE, 0, stream>>>(
        tr->z.data, last_out.data, t2->w.role.data, t2->res_final.data,
        tr->sample_mode.data, tr->sample_partner.data, R, W, H);
    float gscale = 1.0f / ((float)R * (float)L);
    t2_head_forward(t2, tr->z, tr->u, tr->hid, tr->pre, tr->logits, tr->nll_rp,
        tr->nll, tr->target.data, tr->logits, R, gscale, stream);
    // The CE kernel wrote softmax gradients into logits (in place).

    // Backward through the head.
    puf_mm_tn(&tr->logits, &tr->hid, &t2->g.w_out, stream);
    puf_mm_nn(&tr->logits, &t2->w.w_out, &tr->d_hid, stream);
    long n_hid = (long)R * L * t2->D;
    t2_head_backward_kernel<<<grid_size(n_hid), BLOCK_SIZE, 0, stream>>>(
        tr->d_hid.data, tr->d_hid.data, tr->pre.data, n_hid);
    t2_reduce_positions_kernel<<<grid_size(R * t2->D), BLOCK_SIZE, 0, stream>>>(
        tr->d_u.data, tr->d_hid.data, R, L, t2->D);
    t2_reduce_rows_kernel<<<grid_size(L * t2->D), BLOCK_SIZE, 0, stream>>>(
        t2->g.pos_embed.data, tr->d_hid.data, R, L, t2->D);
    puf_mm_tn(&tr->d_u, &tr->z, &t2->g.w_h, stream);
    puf_mm_nn(&tr->d_u, &t2->w.w_h, &tr->d_z, stream);
    // Route d_z to the last window position of query/support scans and role.
    long n_out = (long)2 * R * W * H;
    t2_zero_prec_kernel<<<grid_size(n_out), BLOCK_SIZE, 0, stream>>>(tr->d_out.data, n_out);
    // Role partials reuse d_hid scratch (R*H fits in R*L*D).
    t2_train_dz_kernel<<<grid_size(R * 2 * H), BLOCK_SIZE, 0, stream>>>(
        tr->d_out.data, tr->d_hid.data, tr->d_z.data, tr->sample_mode.data, R, W, H);
    t2_reduce_role_kernel<<<grid_size(H), BLOCK_SIZE, 0, stream>>>(
        t2->g.role.data, tr->d_hid.data, R, H);
    // Backward through the scans (last layer first).
    Prec grad = tr->d_out;
    t2_zero_prec_kernel<<<grid_size((long)2 * R * H), BLOCK_SIZE, 0, stream>>>(
        tr->grad_next_state.data, (long)2 * R * H);
    for (int l = Lg - 1; l >= 0; l--) {
        PrefixScan& scan = tr->scan[l];
        mingru_scan_backward<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream>>>(
            scan, grad.data, tr->grad_next_state.data);
        puf_mm_tn(&scan.grad_combined, &tr->x[l], &t2->g.gru[l], stream);
        puf_mm_nn(&scan.grad_combined, &t2->w.gru[l], &tr->d_x, stream);
        long n = numel(scan.grad_input.shape);
        add_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            tr->d_x.data, scan.grad_input.data, n);
        grad = tr->d_x;
    }
    // Input projection and embeddings.
    puf_mm_tn(&tr->d_x, &tr->x_in, &t2->g.w_in, stream);
    puf_mm_nn(&tr->d_x, &t2->w.w_in, &tr->d_x_in, stream);
    long n_tok = numel(t2->g.tok_embed.shape);
    long n_act = numel(t2->g.act_embed.shape);
    cudaMemsetAsync(tr->tok_embed_grad_i.data, 0, n_tok * sizeof(long), stream);
    cudaMemsetAsync(tr->act_embed_grad_i.data, 0, n_act * sizeof(long), stream);
    dim3 embed_grid(L, T2_EMBED_CHUNKS, t2->E / T2_EMBED_COLS);
    t2_tok_embed_backward_kernel<<<embed_grid, T2_EMBED_COLS, 0, stream>>>(
        (long long*)tr->tok_embed_grad_i.data, tr->d_x_in.data, tr->tok_seq.data,
        rows, L, t2->E, t2->EA);
    t2_act_embed_backward_kernel<<<grid_size(rows * t2->EA), BLOCK_SIZE, 0, stream>>>(
        (long long*)tr->act_embed_grad_i.data, tr->d_x_in.data, tr->act_seq.data,
        t2->head_off_dev, rows, t2->E, t2->EA, t2->heads);
    t2_fxp_to_precision_kernel<<<grid_size(n_tok), BLOCK_SIZE, 0, stream>>>(
        t2->g.tok_embed.data, (long long*)tr->tok_embed_grad_i.data, n_tok);
    t2_fxp_to_precision_kernel<<<grid_size(n_act), BLOCK_SIZE, 0, stream>>>(
        t2->g.act_embed.data, (long long*)tr->act_embed_grad_i.data, n_act);

    float* h_nll = (float*)malloc(R * sizeof(float));
    cudaMemcpyAsync(h_nll, tr->nll.data, R * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double total = 0.0;
    for (int r = 0; r < R; r++) {
        total += h_nll[r];
    }
    free(h_nll);
    return (float)(total / R);
}

// Adam over the flat gradient with global-norm clipping.
static void t2_adam(T2* t2, cudaStream_t stream) {
    T2Train* tr = &t2->tr;
    long n = numel(t2->grad.shape);
    int blocks = min((int)grid_size(n), 256);
    muon_sum_sq_partials<<<blocks, 256, 0, stream>>>(tr->partials.data, t2->grad.data, n);
    muon_sum_sq_reduce<<<1, 256, 0, stream>>>(tr->norm, tr->partials.data, blocks);
    t2->adam_step++;
    float bc1 = 1.0f - powf(t2->beta1, (float)t2->adam_step);
    float bc2 = 1.0f - powf(t2->beta2, (float)t2->adam_step);
    t2_adam_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        t2->master.data, t2->adam_m.data, t2->adam_v.data, t2->grad.data, tr->norm,
        t2->max_grad_norm, t2->lr, t2->beta1, t2->beta2, t2->eps, bc1, bc2, n);
    if (USE_BF16) {
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(t2->param.data, t2->master.data, n);
    }
}

static void t2_train_step(T2* t2, cudaStream_t stream) {
    int peer_rows = t2_sample(t2, stream);
    if (peer_rows < 0) {
        return;
    }
    float loss = t2_forward_backward(t2, stream);
    t2_adam(t2, stream);
    float add[T2_STAT_N] = {0};
    add[T2_STAT_WM_LOSS] = loss;
    add[T2_STAT_WM_STEPS] = 1.0f;
    add[T2_STAT_PEER_ROWS] = (float)peer_rows;
    add[T2_STAT_ROWS] = (float)t2->R;
    float cur[T2_STAT_N];
    cudaMemcpy(cur, t2->stats, sizeof(cur), cudaMemcpyDeviceToHost);
    for (int i = 0; i < T2_STAT_N; i++) {
        cur[i] += add[i];
    }
    cudaMemcpy(t2->stats, cur, sizeof(cur), cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// Re-encoding under the current weights: after WM updates, streaming lane
// states and reservoir states are stale continuations of older weights.
// Records are replayed right-aligned in windows of M steps (reset flag at
// the first real step), in chunks that reuse the training scan buffers, so
// every stored state is again a literal encoding of its prefix.
// ---------------------------------------------------------------------------

__global__ void t2_reencode_gather_kernel(unsigned char* __restrict__ tok_seq,
        int* __restrict__ act_seq, precision_t* __restrict__ term,
        const unsigned char* __restrict__ src_tok, const int* __restrict__ src_act,
        const int* __restrict__ ids, const int* __restrict__ lens,
        const int* __restrict__ act_sizes, int G, int M, int L, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * M) {
        return;
    }
    int g = idx / M, w = idx % M;
    int len = lens[g], pos = w - (M - len);
    unsigned char* dst = tok_seq + (long)idx * L;
    if (pos < 0) {
        for (int p = 0; p < L; p++) {
            dst[p] = 0;
        }
        for (int h = 0; h < heads; h++) {
            act_seq[(long)idx * heads + h] = act_sizes[h];
        }
        term[idx] = from_float(0.0f);
        return;
    }
    long rec = (long)ids[g] * M + pos;
    const unsigned char* s = src_tok + rec * L;
    for (int p = 0; p < L; p++) {
        dst[p] = s[p];
    }
    for (int h = 0; h < heads; h++) {
        act_seq[(long)idx * heads + h] = src_act[rec * heads + h];
    }
    term[idx] = from_float(pos == 0 ? 1.0f : 0.0f);
}

// Lane states from a chunk: state = last post-step state, state_prev = the
// state before the last step, out_last = last output (last layer only).
__global__ void t2_reencode_lanes_kernel(precision_t* __restrict__ state,
        precision_t* __restrict__ state_prev, precision_t* __restrict__ out_last,
        const precision_t* __restrict__ scan_h, const precision_t* __restrict__ next_state,
        const precision_t* __restrict__ out, const int* __restrict__ ids,
        int layer, int last_layer, int G, int M, int A, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * H) {
        return;
    }
    int g = idx / H, h = idx % H;
    long a = ids[g];
    state[((long)layer * A + a) * H + h] = next_state[(long)g * H + h];
    state_prev[((long)layer * A + a) * H + h] = scan_h[((long)g * M + M - 1) * H + h];
    if (last_layer) {
        out_last[a * H + h] = out[((long)g * M + M - 1) * H + h];
    }
}

// Recorded states from a chunk: record[j, s] = state after step s.
// Used for active episode histories as well as completed replay histories.
__global__ void t2_reencode_records_kernel(precision_t* __restrict__ res_state,
        precision_t* __restrict__ res_final, const precision_t* __restrict__ scan_h,
        const precision_t* __restrict__ next_state, const precision_t* __restrict__ out,
        const int* __restrict__ ids, const int* __restrict__ lens,
        int layer, int last_layer, int G, int M, int Lg, int H) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)G * M * H) {
        return;
    }
    int h = idx % H, s = (idx / H) % M, g = idx / ((long)M * H);
    int len = lens[g];
    if (s >= len) {
        return;
    }
    long j = ids[g];
    float v = s + 1 < len ? to_float(scan_h[((long)g * M + M - len + s + 1) * H + h])
        : to_float(next_state[(long)g * H + h]);
    res_state[((j * M + s) * Lg + layer) * H + h] = from_float(v);
    if (last_layer && s == 0) {
        res_final[j * H + h] = out[((long)g * M + M - 1) * H + h];
    }
}

// Encode G records (ids/lens on device) through the scan; outputs stay in the
// training scan buffers viewed as (G, M).
static void t2_reencode_chunk(T2* t2, const unsigned char* src_tok, const int* src_act,
        const int* ids, const int* lens, int G, cudaStream_t stream) {
    T2Train* tr = &t2->tr;
    int M = t2->M, H = t2->H, L = t2->L, width = t2->E + t2->EA;
    long rows = (long)G * M;
    t2_reencode_gather_kernel<<<grid_size(rows), BLOCK_SIZE, 0, stream>>>(
        tr->tok_seq.data, tr->act_seq.data, tr->term.data, src_tok, src_act, ids, lens,
        t2->act_sizes_dev, G, M, L, t2->heads);
    Prec x_in = {.data = tr->x_in.data, .shape = {rows, width}};
    Prec x = {.data = tr->x[0].data, .shape = {rows, H}};
    t2_input_forward(t2, x_in, x, tr->tok_seq.data, tr->act_seq.data, (int)rows, stream);
    cudaMemsetAsync(tr->init_state.data, 0,
        (size_t)t2->Lg * G * H * sizeof(precision_t), stream);
    for (int l = 0; l < t2->Lg; l++) {
        Prec xl = {.data = tr->x[l].data, .shape = {rows, H}};
        Prec comb = {.data = tr->combined[l].data, .shape = {rows, 3 * H}};
        puf_mm(&xl, &t2->w.gru[l], &comb, stream);
        PrefixScan scan = tr->scan[l];
        scan.B = G;
        scan.T = M;
        scan.combined_ptr = comb.data;
        scan.state_ptr = tr->init_state.data + (long)l * G * H;
        scan.input_ptr = xl.data;
        scan.terminals_ptr = tr->term.data;
        mingru_scan_forward<<<grid_size(G * H), BLOCK_SIZE, 0, stream>>>(scan);
        if (l + 1 < t2->Lg) {
            cudaMemcpyAsync(tr->x[l + 1].data, scan.out.data,
                rows * H * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
        }
    }
}

// Continue each refreshed active prefix through its CPU suffix. Scratch stays
// bounded by the existing M-step scan. No reset or padding is inserted between
// chunks. out_last retains the GPU replay-prefix summary, matching res_len=M.
static void t2_reencode_overflow(T2* t2, cudaStream_t stream) {
    T2Train* tr = &t2->tr;
    int H = t2->H;
    for (int a = 0; a < t2->A; a++) {
        T2Overflow* tail = &t2->overflow[a];
        if (!tail->len) continue;
        assert(t2->host_len[a] == t2->M && "T2 suffix requires its complete prefix");
        cudaMemcpyAsync(tr->sample_slot.data, &a, sizeof(int), cudaMemcpyHostToDevice, stream);
        for (size_t start = 0; start < tail->len; start += t2->M) {
            int steps = (int)((tail->len - start) < (size_t)t2->M
                ? tail->len - start : (size_t)t2->M);
            cudaMemcpyAsync(tr->tok_seq.data, tail->tok + start * t2->L,
                (size_t)steps * t2->L, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(tr->act_seq.data, tail->act + start * t2->heads,
                (size_t)steps * t2->heads * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemsetAsync(tr->term.data, 0, steps * sizeof(precision_t), stream);
            Prec input = {.data = tr->x_in.data, .shape = {steps, t2->E + t2->EA}};
            Prec x = {.data = tr->x[0].data, .shape = {steps, H}};
            t2_input_forward(t2, input, x, tr->tok_seq.data, tr->act_seq.data, steps, stream);
            for (int l = 0; l < t2->Lg; l++) {
                cudaMemcpyAsync(tr->init_state.data + (long)l * H,
                    t2->state.data + ((long)l * t2->A + a) * H,
                    H * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
                Prec xl = {.data = tr->x[l].data, .shape = {steps, H}};
                Prec combined = {.data = tr->combined[l].data, .shape = {steps, 3 * H}};
                puf_mm(&xl, &t2->w.gru[l], &combined, stream);
                PrefixScan scan = tr->scan[l];
                scan.B = 1;
                scan.T = steps;
                scan.combined_ptr = combined.data;
                scan.state_ptr = tr->init_state.data + (long)l * H;
                scan.input_ptr = xl.data;
                scan.terminals_ptr = tr->term.data;
                mingru_scan_forward<<<grid_size(H), BLOCK_SIZE, 0, stream>>>(scan);
                if (l + 1 < t2->Lg) {
                    cudaMemcpyAsync(tr->x[l + 1].data, scan.out.data,
                        (size_t)steps * H * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
                }
                t2_reencode_lanes_kernel<<<grid_size(H), BLOCK_SIZE, 0, stream>>>(
                    t2->state.data, t2->state_prev.data, t2->out_last.data,
                    scan.scan_h.data, scan.next_state.data, scan.out.data,
                    tr->sample_slot.data, l, 0, 1, steps, t2->A, H);
            }
        }
        // The next lane reuses the host ID and the scratch buffers.
        cudaStreamSynchronize(stream);
    }
}

void t2_reencode(T2* t2, cudaStream_t stream) {
    T2Train* tr = &t2->tr;
    int M = t2->M, H = t2->H, Lg = t2->Lg;
    int per_chunk = (int)(((long)2 * t2->R * t2->W) / M);
    int n = t2->A > t2->C ? t2->A : t2->C;
    int* ids = (int*)malloc(n * sizeof(int));
    int* lens = (int*)malloc(n * sizeof(int));
    int* ids_dev = tr->sample_slot.data;    // scratch: sample buffers are idle here
    int* lens_dev = tr->sample_t.data;
    assert(per_chunk >= 1 && t2->R >= per_chunk && "re-encode chunk exceeds sample scratch");
    // Refresh every active GPU prefix, including exactly-full records. CPU
    // suffixes continue these states below, before any new reward is scored.
    int count = 0;
    for (int a = 0; a < t2->A; a++) {
        if (t2->host_len[a] > 0 && t2->host_len[a] <= M) {
            ids[count] = a;
            lens[count] = t2->host_len[a];
            count++;
        }
    }
    for (int start = 0; start < count; start += per_chunk) {
        int G = count - start < per_chunk ? count - start : per_chunk;
        cudaMemcpyAsync(ids_dev, ids + start, G * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(lens_dev, lens + start, G * sizeof(int), cudaMemcpyHostToDevice, stream);
        t2_reencode_chunk(t2, t2->ep_tok.data, t2->ep_act.data, ids_dev, lens_dev, G, stream);
        for (int l = 0; l < Lg; l++) {
            PrefixScan* sc = &tr->scan[l];
            t2_reencode_lanes_kernel<<<grid_size(G * H), BLOCK_SIZE, 0, stream>>>(
                t2->state.data, t2->state_prev.data, t2->out_last.data, sc->scan_h.data,
                sc->next_state.data, tr->scan[Lg - 1].out.data, ids_dev, l, l == Lg - 1,
                G, M, t2->A, H);
            // Episode completion copies ep_state into replay. Refresh every
            // recorded prefix now, not only the live carry at its end.
            // out_last was already refreshed by the lane kernel above.
            t2_reencode_records_kernel<<<grid_size((long)G * M * H), BLOCK_SIZE, 0, stream>>>(
                t2->ep_state.data, t2->out_last.data, sc->scan_h.data,
                sc->next_state.data, tr->scan[Lg - 1].out.data,
                ids_dev, lens_dev, l, 0, G, M, Lg, H);
        }
        cudaStreamSynchronize(stream);
    }
    t2_reencode_overflow(t2, stream);
    count = 0;
    for (int j = 0; j < t2->C; j++) {
        if (t2->res_len[j] > 0) {
            ids[count] = j;
            lens[count] = t2->res_len[j];
            count++;
        }
    }
    for (int start = 0; start < count; start += per_chunk) {
        int G = count - start < per_chunk ? count - start : per_chunk;
        cudaMemcpyAsync(ids_dev, ids + start, G * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(lens_dev, lens + start, G * sizeof(int), cudaMemcpyHostToDevice, stream);
        t2_reencode_chunk(t2, t2->res_tok.data, t2->res_act.data, ids_dev, lens_dev, G, stream);
        for (int l = 0; l < Lg; l++) {
            PrefixScan* sc = &tr->scan[l];
            t2_reencode_records_kernel<<<grid_size((long)G * M * H), BLOCK_SIZE, 0, stream>>>(
                t2->res_state.data, t2->res_final.data, sc->scan_h.data, sc->next_state.data,
                tr->scan[Lg - 1].out.data, ids_dev, lens_dev, l, l == Lg - 1, G, M, Lg, H);
        }
        cudaStreamSynchronize(stream);
    }
    free(ids);
    free(lens);
}

void t2_train(PuffeRL* p) {
    T2* t2 = p->t2;
    cudaStream_t stream = p->train_stream;
    double t0 = t2_now();
    for (int s = 0; s < t2->steps; s++) {
        t2_train_step(t2, stream);
    }
    if (t2->reencode) {
        t2_reencode(t2, stream);
    }
    cudaStreamSynchronize(stream);
    assert(cudaGetLastError() == cudaSuccess && "t2 train kernel failed");
    t2->train_ms += (float)((t2_now() - t0) * 1000.0);
}

void t2_log(PuffeRL* p, Dict* out) {
    T2* t2 = p->t2;
    float s[T2_STAT_N];
    cudaMemcpy(s, t2->stats, sizeof(s), cudaMemcpyDeviceToHost);
    cudaMemset(t2->stats, 0, sizeof(s));
    float inv_valid = s[T2_STAT_VALID] > 0 ? 1.0f / s[T2_STAT_VALID] : 0.0f;
    dict_set(out, "env/t2_reward", s[T2_STAT_REWARD] * inv_valid);
    dict_set(out, "env/t2_nll_ctrl", s[T2_STAT_NLL_CTRL] * inv_valid);
    dict_set(out, "env/t2_nll_1", s[T2_STAT_NLL_1] * inv_valid);
    dict_set(out, "env/t2_cap_frac", s[T2_STAT_CAPPED] * inv_valid);
    dict_set(out, "env/t2_wm_loss", s[T2_STAT_WM_STEPS] > 0
        ? s[T2_STAT_WM_LOSS] / s[T2_STAT_WM_STEPS] : 0.0f);
    dict_set(out, "env/t2_peer_frac", s[T2_STAT_ROWS] > 0
        ? s[T2_STAT_PEER_ROWS] / s[T2_STAT_ROWS] : 0.0f);
    int resident = 0;
    for (int j = 0; j < t2->C; j++) {
        resident += t2->res_len[j] > 0;
    }
    dict_set(out, "env/t2_reservoir", resident);
    dict_set(out, "env/t2_episodes", (double)t2->episodes_complete);
    float rollout_ms = 0.0f;
    for (int b = 0; b < t2->num_buffers; b++) {
        rollout_ms += t2->rollout_ms[b];
        t2->rollout_ms[b] = 0.0f;
    }
    dict_set(out, "perf/t2_rollout", rollout_ms / t2->num_buffers / 1000.0f);
    dict_set(out, "perf/t2_train", t2->train_ms / 1000.0f);
    t2->train_ms = 0.0f;
}

// Checkpoint: fp32 master weights followed by the Adam moments and step.
void t2_save(PuffeRL* p, const char* path) {
    T2* t2 = p->t2;
    long n = numel(t2->master.shape);
    float* buf = (float*)malloc(3 * n * sizeof(float));
    cudaMemcpy(buf, t2->master.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(buf + n, t2->adam_m.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(buf + 2 * n, t2->adam_v.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "wb");
    assert(fp && "failed to open t2 weights for writing");
    assert(fwrite(&n, sizeof(n), 1, fp) == 1);
    assert(fwrite(&t2->adam_step, sizeof(t2->adam_step), 1, fp) == 1);
    assert(fwrite(buf, sizeof(float), 3 * n, fp) == (size_t)(3 * n));
    fclose(fp);
    free(buf);
    assert(rename(tmp, path) == 0 && "failed to publish t2 weights");
}

void t2_load(PuffeRL* p, const char* path) {
    T2* t2 = p->t2;
    long n = numel(t2->master.shape);
    FILE* fp = fopen(path, "rb");
    assert(fp && "failed to open t2 weights");
    long n_file = 0;
    assert(fread(&n_file, sizeof(n_file), 1, fp) == 1 && n_file == n
        && "t2 checkpoint size mismatch");
    assert(fread(&t2->adam_step, sizeof(t2->adam_step), 1, fp) == 1);
    float* buf = (float*)malloc(3 * n * sizeof(float));
    assert(fread(buf, sizeof(float), 3 * n, fp) == (size_t)(3 * n));
    fclose(fp);
    cudaMemcpy(t2->master.data, buf, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t2->adam_m.data, buf + n, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t2->adam_v.data, buf + 2 * n, n * sizeof(float), cudaMemcpyHostToDevice);
    free(buf);
    if (USE_BF16) {
        cast<<<grid_size(n), BLOCK_SIZE>>>(t2->param.data, t2->master.data, n);
    }
    cudaDeviceSynchronize();
}

static void t2_reg_train(T2* t2) {
    T2Train* tr = &t2->tr;
    Allocator* a = &t2->buf_alloc;
    int R = t2->R, W = t2->W, H = t2->H, L = t2->L, D = t2->D;
    int width = t2->E + t2->EA;
    long rows = (long)2 * R * W;
    *tr = (T2Train){};
    tr->sample_slot = {.shape = {R}};
    tr->sample_t = {.shape = {R}};
    tr->sample_mode = {.shape = {R}};
    tr->sample_partner = {.shape = {R}};
    tr->tok_seq = {.shape = {rows, L}};
    tr->act_seq = {.shape = {rows, t2->heads}};
    tr->term = {.shape = {2 * R, W}};
    tr->target = {.shape = {R, L}};
    tr->init_state = {.shape = {t2->Lg, 2 * R, H}};
    tr->x_in = {.shape = {rows, width}};
    tr->z = {.shape = {R, 2 * H}};
    tr->u = {.shape = {R, D}};
    tr->pre = {.shape = {(long)R * L, D}};
    tr->hid = {.shape = {(long)R * L, D}};
    tr->logits = {.shape = {(long)R * L, T2_VOCAB}};
    tr->nll_rp = {.shape = {R, L}};
    tr->nll = {.shape = {R}};
    tr->d_hid = {.shape = {(long)R * L, D}};
    tr->d_u = {.shape = {R, D}};
    tr->d_z = {.shape = {R, 2 * H}};
    tr->d_out = {.shape = {2 * R, W, H}};
    tr->d_x = {.shape = {rows, H}};
    tr->d_x_in = {.shape = {rows, width}};
    tr->grad_next_state = {.shape = {2 * R, H}};
    tr->tok_embed_grad_i = {.shape = {(long)T2_TOKENS * T2_VOCAB * t2->E}};
    tr->act_embed_grad_i = {.shape = {(long)t2->act_rows * t2->EA}};
    tr->partials = {.shape = {256}};
    alloc_register(a, &tr->sample_slot);
    alloc_register(a, &tr->sample_t);
    alloc_register(a, &tr->sample_mode);
    alloc_register(a, &tr->sample_partner);
    alloc_register(a, &tr->tok_seq);
    alloc_register(a, &tr->act_seq);
    alloc_register(a, &tr->term);
    alloc_register(a, &tr->target);
    alloc_register(a, &tr->init_state);
    alloc_register(a, &tr->x_in);
    alloc_register(a, &tr->z);
    alloc_register(a, &tr->u);
    alloc_register(a, &tr->pre);
    alloc_register(a, &tr->hid);
    alloc_register(a, &tr->logits);
    alloc_register(a, &tr->nll_rp);
    alloc_register(a, &tr->nll);
    alloc_register(a, &tr->d_hid);
    alloc_register(a, &tr->d_u);
    alloc_register(a, &tr->d_z);
    alloc_register(a, &tr->d_out);
    alloc_register(a, &tr->d_x);
    alloc_register(a, &tr->d_x_in);
    alloc_register(a, &tr->grad_next_state);
    alloc_register(a, &tr->tok_embed_grad_i);
    alloc_register(a, &tr->act_embed_grad_i);
    alloc_register(a, &tr->partials);
    for (int l = 0; l < t2->Lg; l++) {
        tr->x[l] = {.shape = {rows, H}};
        tr->combined[l] = {.shape = {rows, 3 * H}};
        tr->scan[l] = {
            .B = 2 * R, .T = W, .H = H,
            .scan_h =        {.shape = {2 * R, W, H}},
            .out =           {.shape = {2 * R, W, H}},
            .next_state =    {.shape = {2 * R, 1, H}},
            .grad_combined = {.shape = {2 * R, W, 3 * H}},
            .grad_state =    {.shape = {2 * R, 1, H}},
            .grad_input =    {.shape = {2 * R, W, H}},
        };
        alloc_register(a, &tr->x[l]);
        alloc_register(a, &tr->combined[l]);
        alloc_register(a, &tr->scan[l].scan_h);
        alloc_register(a, &tr->scan[l].out);
        alloc_register(a, &tr->scan[l].next_state);
        alloc_register(a, &tr->scan[l].grad_combined);
        alloc_register(a, &tr->scan[l].grad_state);
        alloc_register(a, &tr->scan[l].grad_input);
    }
    cudaMalloc((void**)&tr->norm, sizeof(float));
}

static void t2_reg_rollout(T2* t2, T2Rollout* r, int B) {
    Allocator* a = &t2->buf_alloc;
    int H = t2->H, L = t2->L, D = t2->D;
    int width = t2->E + t2->EA;
    *r = (T2Rollout){};
    r->tok_cur = {.shape = {B, L}};
    r->tok_next = {.shape = {B, L}};
    r->target2 = {.shape = {2 * B, L}};
    r->act = {.shape = {B, t2->heads}};
    r->pad = {.shape = {B, t2->heads}};
    r->x_in = {.shape = {3 * B, width}};
    r->x = {.shape = {3 * B, H}};
    r->combined = {.shape = {3 * B, 3 * H}};
    r->out = {.shape = {3 * B, H}};
    r->z = {.shape = {2 * B, 2 * H}};
    r->u = {.shape = {2 * B, D}};
    r->hid = {.shape = {(long)2 * B * L, D}};
    r->logits = {.shape = {(long)2 * B * L, T2_VOCAB}};
    r->nll_rp = {.shape = {2 * B, L}};
    r->nll = {.shape = {2 * B}};
    alloc_register(a, &r->tok_cur);
    alloc_register(a, &r->tok_next);
    alloc_register(a, &r->target2);
    alloc_register(a, &r->act);
    alloc_register(a, &r->pad);
    alloc_register(a, &r->x_in);
    alloc_register(a, &r->x);
    alloc_register(a, &r->combined);
    alloc_register(a, &r->out);
    alloc_register(a, &r->z);
    alloc_register(a, &r->u);
    alloc_register(a, &r->hid);
    alloc_register(a, &r->logits);
    alloc_register(a, &r->nll_rp);
    alloc_register(a, &r->nll);
}

static void t2_reg_params(T2* t2) {
    Allocator* a = &t2->params_alloc;
    Allocator* g = &t2->grads_alloc;
    int H = t2->H, D = t2->D;
    int width = t2->E + t2->EA;
    t2->w.tok_embed = {.shape = {(long)T2_TOKENS * T2_VOCAB, t2->E}};
    t2->w.act_embed = {.shape = {t2->act_rows, t2->EA}};
    t2->w.w_in = {.shape = {H, width}};
    for (int l = 0; l < t2->Lg; l++) {
        t2->w.gru[l] = {.shape = {3 * H, H}};
    }
    t2->w.role = {.shape = {1, H}};
    t2->w.w_h = {.shape = {D, 2 * H}};
    t2->w.pos_embed = {.shape = {T2_TOKENS, D}};
    t2->w.w_out = {.shape = {T2_VOCAB, D}};
    t2->g = (T2Grads){};
    Prec* ws[] = {&t2->w.tok_embed, &t2->w.act_embed, &t2->w.w_in, NULL};
    Prec* gs[] = {&t2->g.tok_embed, &t2->g.act_embed, &t2->g.w_in, NULL};
    for (int i = 0; ws[i]; i++) {
        alloc_register(a, ws[i]);
        *gs[i] = {.shape = {ws[i]->shape[0], ws[i]->shape[1]}};
        alloc_register(g, gs[i]);
    }
    for (int l = 0; l < t2->Lg; l++) {
        alloc_register(a, &t2->w.gru[l]);
        t2->g.gru[l] = {.shape = {3 * H, H}};
        alloc_register(g, &t2->g.gru[l]);
    }
    Prec* ws2[] = {&t2->w.role, &t2->w.w_h, &t2->w.pos_embed, &t2->w.w_out, NULL};
    Prec* gs2[] = {&t2->g.role, &t2->g.w_h, &t2->g.pos_embed, &t2->g.w_out, NULL};
    for (int i = 0; ws2[i]; i++) {
        alloc_register(a, ws2[i]);
        *gs2[i] = {.shape = {ws2[i]->shape[0], ws2[i]->shape[1]}};
        alloc_register(g, gs2[i]);
    }
}

static void t2_init_weights(T2* t2, ulong seed, cudaStream_t stream) {
    // Embedding bags sum T2_TOKENS rows: keep the sum O(1).
    puf_normal_init(&t2->w.tok_embed, 1.0f / sqrtf((float)T2_TOKENS), seed++, stream);
    puf_normal_init(&t2->w.act_embed, 1.0f, seed++, stream);
    puf_kaiming_init(&t2->w.w_in, 1.0f, seed++, stream);
    for (int l = 0; l < t2->Lg; l++) {
        puf_kaiming_init(&t2->w.gru[l], 1.0f, seed++, stream);
    }
    puf_kaiming_init(&t2->w.w_h, 1.0f, seed++, stream);
    puf_normal_init(&t2->w.pos_embed, 0.1f, seed++, stream);
    puf_kaiming_init(&t2->w.w_out, 1.0f, seed++, stream);
    // role starts at zero (alloc_create memsets).
}

int t2_enabled(Ini* ini) {
    return puf_ini_get(ini, "t2", "enabled") != 0;
}

T2* t2_create(PuffeRL* p, Ini* ini) {
    assert(PUF_BACKEND == PUF_CPU && "T2 supports CPU env backends");
    assert(p->num_policies == 1 && "T2 requires a single trainable policy");
    assert(!p->is_continuous && "T2 requires discrete action heads");
    VecEnv* vec = p->vec;
    assert(vec->size == vec->total_agents && "T2 requires one agent per env");
    int act_sizes[] = ACT_SIZES;
    T2* t2 = (T2*)calloc(1, sizeof(T2));
    assert(pthread_mutex_init(&t2->reservoir_mutex,NULL)==0);
    t2->heads = NUM_ATNS;
    t2->act_rows = 0;
    for (int h = 0; h < NUM_ATNS; h++) {
        t2->act_sizes[h] = act_sizes[h];
        t2->head_off[h] = t2->act_rows;
        t2->act_rows += act_sizes[h] + 1;
    }
    cudaMalloc((void**)&t2->act_sizes_dev, NUM_ATNS * sizeof(int));
    cudaMalloc((void**)&t2->head_off_dev, NUM_ATNS * sizeof(int));
    cudaMemcpy(t2->act_sizes_dev, t2->act_sizes, NUM_ATNS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(t2->head_off_dev, t2->head_off, NUM_ATNS * sizeof(int), cudaMemcpyHostToDevice);
    t2->reencode = puf_ini_get(ini, "t2", "reencode") != 0;
    t2->A = vec->total_agents;
    t2->L = T2_TOKENS;
    t2->E = puf_ini_get(ini, "t2", "embed_dim");
    t2->EA = 16;
    t2->H = puf_ini_get(ini, "t2", "hidden_size");
    t2->Lg = puf_ini_get(ini, "t2", "num_layers");
    t2->D = puf_ini_get(ini, "t2", "head_dim");
    t2->M = puf_ini_get(ini, "t2", "max_episode");
    t2->C = puf_ini_get(ini, "t2", "reservoir");
    t2->R = puf_ini_get(ini, "t2", "wm_batch");
    t2->W = puf_ini_get(ini, "t2", "bptt_window");
    t2->steps = puf_ini_get(ini, "t2", "wm_steps");
    t2->horizon = p->hypers.horizon;
    t2->num_buffers = p->hypers.num_buffers;
    t2->slots = p->async_num_slots;
    t2->reward_scale = puf_ini_get(ini, "t2", "reward_scale");
    t2->reward_cap = puf_ini_get(ini, "t2", "reward_cap");
    t2->peer_prob = puf_ini_get(ini, "t2", "peer_prob");
    t2->lr = puf_ini_get(ini, "t2", "learning_rate");
    t2->beta1 = puf_ini_get(ini, "t2", "adam_beta1");
    t2->beta2 = puf_ini_get(ini, "t2", "adam_beta2");
    t2->eps = puf_ini_get(ini, "t2", "adam_eps");
    t2->max_grad_norm = puf_ini_get(ini, "t2", "max_grad_norm");
    t2->intrinsic_only = puf_ini_get(ini, "t2", "intrinsic_only") != 0;
    t2->rng = (unsigned int)p->seed * 2654435761u + 12345u;
    assert(t2->Lg >= 1 && t2->Lg <= T2_MAX_LAYERS && "t2.num_layers in 1..4");
    assert(t2->E % T2_EMBED_COLS == 0 && "t2.embed_dim must be a multiple of 32");
    assert(t2->C >= 2 && t2->C % 2 == 0 && "t2.reservoir must be even");
    assert(t2->A % t2->num_buffers == 0 && t2->A % 2 == 0
        && (t2->A / t2->num_buffers) % 2 == 0 && "paired lanes must not straddle buffers");
    assert(t2->M >= 2 && t2->W >= 1 && t2->R >= 1);
    if (!T2_PAIRED) {
        fprintf(stderr, "t2: env has no PUF_T2_PAIRED reset; peer support disabled\n");
        t2->peer_prob = 0.0f;
    }
    if (T2_GENERIC_CODEC) {
        fprintf(stderr, "t2: env has no PUF_T2_TOKENS codec; quantizing %d values\n", t2->L);
    }

    t2_reg_params(t2);
    alloc_create(&t2->params_alloc);
    alloc_create(&t2->grads_alloc);
    t2->param = {.data = (precision_t*)t2->params_alloc.mem,
        .shape = {t2->params_alloc.total_elems}};
    t2->grad = {.data = (precision_t*)t2->grads_alloc.mem,
        .shape = {t2->grads_alloc.total_elems}};
    long n = t2->params_alloc.total_elems;
    // fp32 master weights: a separate copy in bf16 builds, an alias in float builds.
    t2->master = {.shape = {n}};
    if (USE_BF16) {
        cudaMalloc((void**)&t2->master.data, n * sizeof(float));
    } else {
        t2->master.data = (float*)t2->param.data;
    }
    t2->adam_m = {.shape = {n}};
    t2->adam_v = {.shape = {n}};
    cudaMalloc((void**)&t2->adam_m.data, n * sizeof(float));
    cudaMalloc((void**)&t2->adam_v.data, n * sizeof(float));
    cudaMemset(t2->adam_m.data, 0, n * sizeof(float));
    cudaMemset(t2->adam_v.data, 0, n * sizeof(float));
    t2_init_weights(t2, p->seed * 7919 + 17, p->default_stream);
    if (USE_BF16) {
        cast<<<grid_size(n), BLOCK_SIZE, 0, p->default_stream>>>(
            t2->master.data, t2->param.data, n);
    }

    // Lane, record, reservoir, rollout and training buffers.
    Allocator* a = &t2->buf_alloc;
    int A = t2->A, H = t2->H, Lg = t2->Lg, L = t2->L, M = t2->M, C = t2->C;
    t2->state = {.shape = {Lg, A, H}};
    t2->state_prev = {.shape = {Lg, A, H}};
    t2->out_last = {.shape = {A, H}};
    t2->ep_tok = {.shape = {A, M, L}};
    t2->ep_act = {.shape = {A, M, t2->heads}};
    t2->ep_state = {.shape = {A, M, Lg, H}};
    t2->ep_len = {.shape = {A}};
    t2->rewards = {.shape = {t2->slots * t2->horizon, A}};
    t2->res_tok = {.shape = {C, M, L}};
    t2->res_act = {.shape = {C, M, t2->heads}};
    t2->res_state = {.shape = {C, M, Lg, H}};
    t2->res_final = {.shape = {C, H}};
    alloc_register(a, &t2->state);
    alloc_register(a, &t2->state_prev);
    alloc_register(a, &t2->out_last);
    alloc_register(a, &t2->ep_tok);
    alloc_register(a, &t2->ep_act);
    alloc_register(a, &t2->ep_state);
    alloc_register(a, &t2->ep_len);
    alloc_register(a, &t2->rewards);
    alloc_register(a, &t2->res_tok);
    alloc_register(a, &t2->res_act);
    alloc_register(a, &t2->res_state);
    alloc_register(a, &t2->res_final);
    t2->roll = (T2Rollout*)calloc(t2->num_buffers, sizeof(T2Rollout));
    for (int b = 0; b < t2->num_buffers; b++) {
        t2_reg_rollout(t2, &t2->roll[b], A / t2->num_buffers);
    }
    t2_reg_train(t2);
    alloc_create(a);
    fprintf(stderr, "t2: params %.2fM, buffers %.2f GB, tokens %d, lanes %d, reservoir %d x %d\n",
        n / 1e6, a->total_bytes / 1e9, L, A, C, M);

    t2->host_len = (int*)calloc(A, sizeof(int));
    t2->overflow = (T2Overflow*)calloc(A, sizeof(T2Overflow));
    assert(t2->overflow && "T2 active history index allocation failed");
    t2->host_episode = (int*)calloc(A, sizeof(int));
    t2->res_len = (int*)calloc(C, sizeof(int));
    t2->res_pair = (int*)calloc(C, sizeof(int));
    t2->res_episode = (int*)calloc(C, sizeof(int));
    t2->rollout_ms = (float*)calloc(t2->num_buffers, sizeof(float));
    cudaHostAlloc((void**)&t2->host_tok, (size_t)A * L, cudaHostAllocPortable);
    cudaMalloc((void**)&t2->stats, T2_STAT_N * sizeof(float));
    cudaMemset(t2->stats, 0, T2_STAT_N * sizeof(float));
    for (int b = 0; b < t2->num_buffers; b++) {
        cudaEventCreate(&t2->roll[b].start);
        cudaEventCreate(&t2->roll[b].end);
    }

    // Initial observations were produced by env_start; tokenize and stage them.
    t2_encode_tokens(t2, vec->envs, 0, vec->size);
    int B = A / t2->num_buffers;
    int* pad = (int*)malloc((size_t)B * t2->heads * sizeof(int));
    for (int i = 0; i < B; i++) {
        for (int h = 0; h < t2->heads; h++) {
            pad[i * t2->heads + h] = t2->act_sizes[h];
        }
    }
    for (int b = 0; b < t2->num_buffers; b++) {
        cudaMemcpy(t2->roll[b].tok_cur.data, t2->host_tok + (long)b * B * L,
            (size_t)B * L, cudaMemcpyHostToDevice);
        cudaMemcpy(t2->roll[b].pad.data, pad, (size_t)B * t2->heads * sizeof(int),
            cudaMemcpyHostToDevice);
    }
    free(pad);
    if (t2->reencode) {
        assert((long)2 * t2->R * t2->W >= t2->M
            && "t2.reencode needs 2 * wm_batch * bptt_window >= max_episode");
    }
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess && "t2 create failed");
    return t2;
}

// Worker hook after puf_step + cpu_upload for this buffer at rollout step t.
// The worker synchronizes its stream before the next env step, so the
// previous step's events are complete by the time we read them here.
void t2_worker_step(PuffeRL* p, int buf, int t, cudaStream_t stream) {
    T2* t2 = p->t2;
    T2Rollout* r = &t2->roll[buf];
    VecEnv* vec = p->vec;
    if (r->timed) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, r->start, r->end);
        t2->rollout_ms[buf] += ms;
        r->timed = 0;
    }
    int env_start = vec->env_starts[buf];
    int env_count = vec->env_counts[buf];
    t2_encode_tokens(t2, vec->envs, env_start, env_count);
    cudaEventRecord(r->start, stream);
    t2_rollout_step(p, buf, t, stream);
    t2_episode_bookkeeping(p, buf, stream);
    cudaEventRecord(r->end, stream);
    r->timed = 1;
}
