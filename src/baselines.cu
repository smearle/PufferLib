// Intrinsic-reward baselines in the native trainer: E3B, ICM, RND and
// E3B x RND, following the released MiniHack code paths used by the E3B
// paper (Henaff et al. 2022; Henaff, Jiang & Raileanu 2023) as encoded in
// infogain-world-models' jax_e3b / jax_rnd contracts. Included by intrinsic.cu.
//
//   e3b      b_t = phi(s_t)^T C^-1 phi(s_t) with C the episodic Gram matrix of
//            phi (ridge lambda I at every reset), Sherman-Morrison updates,
//            bonus zeroed on the episode's first step, C reset when the
//            transition terminates; phi from an inverse-dynamics encoder.
//   icm      b_t = || phi(s_{t+1}) - f(phi(s_t), a_t) ||_2, forward and inverse
//            losses on the same encoder, no reward normalization.
//   rnd      b_t = || predictor(s_{t+1}) - target(s_{t+1}) ||_2.
//   e3b_rnd  b_t = e3b_t * rnd_t, then one running standard deviation.
// Bonuses land on the rollout row of s_{t+1}, the row PufferLib credits to
// a_t, scaled by the method coefficient, and go through the trainer's
// [-1, 1] reward clamp exactly like TorchBeast's reward clipping.
//
// Reward normalization follows the released learner: "torchbeast" scans the
// (T, B) bonus tensor in virtual learner batches of eight lanes and advances
// the running count by T per batch (the released quirk); "running" is a
// plain Welford estimate over every value; "none" leaves the bonus raw.
// Losses reproduce the TorchBeast reductions: batch/feature means summed
// over the unroll, so gradients carry a factor of the horizon.

#define IR_NORM_NONE 0
#define IR_NORM_TORCHBEAST 1
#define IR_NORM_RUNNING 2
#define IR_OPT_ADAM 0
#define IR_OPT_RMSPROP 1
#define IR_LEARNER_LANES 8

// Trunk + linear head (+ optional LayerNorm): phi(obs) in R^out. The trunk is
// PufferLib's own encoder for this env (linear by default, the env's custom
// CUDA encoder when it has one), a separate instance with its own weights.
struct FeatNet {
    Encoder enc;
    void* weights;            // trunk weights
    Prec w_out;               // (out, hidden)
    Prec ln_gamma, ln_beta;   // (1, out) when layernorm
    int hidden, out, layernorm;
};

struct FeatActs {
    void* trunk;              // Encoder activations (rollout or train)
    Prec h;                   // (rows, hidden) trunk output view
    Prec pre;                 // (rows, out) before LayerNorm
    Prec f;                   // (rows, out)
    Float mean, rstd;         // (rows,) LayerNorm statistics
    Prec d_pre;               // (rows, out)
    Prec d_h;                 // (rows, hidden)
    int rows, train;
};

struct FeatGrads {
    Prec w_out, ln_gamma, ln_beta;
};

// Two-layer MLP without biases: relu(x W1^T) W2^T.
struct Mlp {
    Prec w1, w2;              // (hidden, in), (out, hidden)
    int in, hidden, out;
};

struct MlpActs {
    Prec x;                   // (rows, in) saved input
    Prec h;                   // (rows, hidden) post-relu
    Prec y;                   // (rows, out)
    Prec d_h, d_x;
};

struct MlpGrads {
    Prec w1, w2;
};

// One trainable parameter group: flat params, fp32 master, optimizer state.
struct ParamGroup {
    Allocator params, grads;
    Prec param, grad;
    Float master, m, v;
    Float partials;
    float* norm;
    int optimizer;
    float lr, beta1, beta2, eps, alpha, max_norm;
    long step;
};

struct Baseline {
    int method;               // IR_E3B, IR_ICM, IR_RND, IR_E3B_RND
    int A, num_buffers, horizon, slots, obs_size, hidden, d, act_n, num_heads;
    int batch_rows, steps_per_epoch, mask_resets, intrinsic_only;
    int act_sizes[PUF_MAX_DIMS * 4];
    int act_offsets[PUF_MAX_DIMS * 4];
    float coef, ridge, forward_coef, inverse_coef;
    int e3b_norm, rnd_norm, icm_norm;
    unsigned int rng;

    // Inverse-dynamics feature learner (e3b, icm, e3b_rnd)
    FeatNet feat;
    Mlp inverse, forward;
    ParamGroup idm;           // feat + inverse (+ forward) parameters
    FeatGrads feat_g;
    MlpGrads inverse_g, forward_g;
    // RND (rnd, e3b_rnd)
    FeatNet target, predictor;
    ParamGroup rnd;
    FeatGrads pred_g;
    Allocator target_params;
    Prec target_param;

    // Per-buffer rollout activations (rows = 2B: [obs_t ; obs_{t+1}])
    Allocator buf_alloc;
    Prec* x2;                 // [num_buffers] (2B, obs)
    FeatActs* feat_roll;      // [num_buffers]
    FeatActs* target_roll;
    FeatActs* pred_roll;
    MlpActs* forward_roll;    // [num_buffers] rows B
    Prec* fwd_in_roll;        // [num_buffers] (B, d + act_n)
    Int* act_roll;            // [num_buffers] (B, num_heads)
    Float* err_roll;          // [num_buffers] (B,) scratch

    // Lane state
    Float cinv;               // (A, d, d) episodic inverse covariance
    Int ep_step;              // (A,)
    Float bonus_e3b, bonus_rnd, bonus_icm;  // (slots * T, A) raw bonuses

    Float combined;           // (T, A) scratch
    Float normalized;         // (T, A)

    // Training buffers (rows = batch_rows)
    Int tr_rows;              // (batch_rows, 2): (t, lane)
    Prec tr_x2;               // (2R, obs)
    Int tr_act;               // (R, num_heads)
    Float tr_valid;           // (R,)
    FeatActs feat_tr, target_tr, pred_tr;
    MlpActs inverse_tr, forward_tr;
    Prec inv_in;              // (R, 2d)
    Prec fwd_in;              // (R, d + act_n)
    Prec d_f;                 // (2R, d) gradient into features
    Prec d_fwd_in;            // (R, d + act_n)
    Float loss_inv, loss_fwd, loss_rnd;  // (R,)
    Prec tr_grad_y;           // (R, d) forward-model output gradient
    Prec tr_grad_rnd;         // (R, rnd out) predictor output gradient
    double* norm_state;       // device (sum, m2, count)
    float* stats;             // device (IR_STAT_N,)
    float last_inv, last_fwd, last_rnd;  // mean per-row losses of the last step
    float train_ms;
    float* rollout_ms;
    cudaEvent_t* ev_start;
    cudaEvent_t* ev_end;
    int* timed;
};

enum {
    IR_STAT_BONUS, IR_STAT_NORMALIZED, IR_STAT_ROWS, IR_STAT_INV_LOSS,
    IR_STAT_FWD_LOSS, IR_STAT_RND_LOSS, IR_STAT_STEPS, IR_STAT_ZERO, IR_STAT_N,
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Trainer observations are precision_t; env buffers are obs_t.
__global__ void ir_cast_obs_kernel(precision_t* __restrict__ dst,
        const obs_t* __restrict__ src, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

// One block per row; blockDim >= out is not required (strided loops).
__global__ void ir_layernorm_kernel(precision_t* __restrict__ f,
        float* __restrict__ mean_out, float* __restrict__ rstd_out,
        const precision_t* __restrict__ pre, const precision_t* __restrict__ gamma,
        const precision_t* __restrict__ beta, int out) {
    __shared__ float red[256];
    int row = blockIdx.x, tid = threadIdx.x;
    const precision_t* x = pre + (long)row * out;
    float s = 0.0f;
    for (int i = tid; i < out; i += blockDim.x) {
        s += to_float(x[i]);
    }
    red[tid] = s;
    block_reduce_sum(red, red, tid, blockDim.x, 1);
    float mean = red[0] / out;
    __syncthreads();
    float q = 0.0f;
    for (int i = tid; i < out; i += blockDim.x) {
        float d = to_float(x[i]) - mean;
        q += d * d;
    }
    red[tid] = q;
    block_reduce_sum(red, red, tid, blockDim.x, 1);
    float rstd = rsqrtf(red[0] / out + 1e-5f);
    for (int i = tid; i < out; i += blockDim.x) {
        float xn = (to_float(x[i]) - mean) * rstd;
        f[(long)row * out + i] = from_float(xn * to_float(gamma[i]) + to_float(beta[i]));
    }
    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
}

// d_pre from d_f; gamma/beta grads accumulate in fixed point (deterministic).
__global__ void ir_layernorm_backward_kernel(precision_t* __restrict__ d_pre,
        long long* __restrict__ dgamma_i, long long* __restrict__ dbeta_i,
        const precision_t* __restrict__ d_f, const precision_t* __restrict__ pre,
        const precision_t* __restrict__ gamma, const float* __restrict__ mean,
        const float* __restrict__ rstd, int out) {
    __shared__ float red[512];
    int row = blockIdx.x, tid = threadIdx.x;
    const precision_t* x = pre + (long)row * out;
    const precision_t* g = d_f + (long)row * out;
    float m = mean[row], r = rstd[row];
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = tid; i < out; i += blockDim.x) {
        float xn = (to_float(x[i]) - m) * r;
        float dxn = to_float(g[i]) * to_float(gamma[i]);
        s1 += dxn;
        s2 += dxn * xn;
        atomicAdd((unsigned long long*)&dgamma_i[i],
            (unsigned long long)__float2ll_rn(to_float(g[i]) * xn * T2_FXP));
        atomicAdd((unsigned long long*)&dbeta_i[i],
            (unsigned long long)__float2ll_rn(to_float(g[i]) * T2_FXP));
    }
    red[tid] = s1;
    red[256 + tid] = s2;
    block_reduce_sum(red, red, tid, blockDim.x, 2);
    float m1 = red[0] / out, m2 = red[256] / out;
    for (int i = tid; i < out; i += blockDim.x) {
        float xn = (to_float(x[i]) - m) * r;
        float dxn = to_float(g[i]) * to_float(gamma[i]);
        d_pre[(long)row * out + i] = from_float(r * (dxn - m1 - xn * m2));
    }
}

__global__ void ir_relu_kernel(precision_t* __restrict__ h, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        h[idx] = from_float(fmaxf(0.0f, to_float(h[idx])));
    }
}

__global__ void ir_relu_backward_kernel(precision_t* __restrict__ d_h,
        const precision_t* __restrict__ h, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && to_float(h[idx]) <= 0.0f) {
        d_h[idx] = from_float(0.0f);
    }
}

// Per-row L2 distance and its gradient: err = ||a - b||, d(a) = scale (a-b)/err.
__global__ void ir_l2_kernel(float* __restrict__ err, precision_t* __restrict__ grad,
        const precision_t* __restrict__ a, const precision_t* __restrict__ b,
        const float* __restrict__ valid, int rows, int d, float scale) {
    __shared__ float red[256];
    int row = blockIdx.x, tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float e = to_float(a[(long)row * d + i]) - to_float(b[(long)row * d + i]);
        s += e * e;
    }
    red[tid] = s;
    block_reduce_sum(red, red, tid, blockDim.x, 1);
    float n = sqrtf(red[0]);
    float w = valid ? valid[row] : 1.0f;
    if (tid == 0) {
        err[row] = n * w;
    }
    if (grad) {
        float inv = n > 1e-8f ? scale * w / n : 0.0f;
        for (int i = tid; i < d; i += blockDim.x) {
            float e = to_float(a[(long)row * d + i]) - to_float(b[(long)row * d + i]);
            grad[(long)row * d + i] = from_float(e * inv);
        }
    }
}

// E3B step for one lane per block: u = C^-1 phi, raw = phi^T u,
// C^-1 -= u u^T / (1 + raw); bonus is zero on the episode's first step; a
// terminated transition resets C^-1 = I / ridge and the step counter.
__global__ void ir_e3b_kernel(float* __restrict__ bonus, float* __restrict__ cinv,
        int* __restrict__ ep_step, const precision_t* __restrict__ phi,
        const float* __restrict__ done, int agent0, int d, float ridge) {
    extern __shared__ float sm[];
    float* p = sm;             // phi (d)
    float* u = sm + d;         // C^-1 phi (d)
    float* red = sm + 2 * d;   // reduction (blockDim)
    int b = blockIdx.x, tid = threadIdx.x;
    int a = agent0 + b;
    float* C = cinv + (long)a * d * d;
    for (int i = tid; i < d; i += blockDim.x) {
        p[i] = to_float(phi[(long)b * d + i]);
    }
    __syncthreads();
    for (int i = tid; i < d; i += blockDim.x) {
        float s = 0.0f;
        const float* row = C + (long)i * d;
        for (int j = 0; j < d; j++) {
            s += row[j] * p[j];
        }
        u[i] = s;
    }
    __syncthreads();
    float s = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        s += p[i] * u[i];
    }
    red[tid] = s;
    block_reduce_sum(red, red, tid, blockDim.x, 1);
    float raw = red[0];
    int step = ep_step[a];
    int reset = done[b] != 0.0f;
    float scale = 1.0f / (1.0f + raw);
    for (int i = tid; i < d; i += blockDim.x) {
        float* row = C + (long)i * d;
        if (reset) {
            for (int j = 0; j < d; j++) {
                row[j] = i == j ? 1.0f / ridge : 0.0f;
            }
        } else {
            float ui = u[i] * scale;
            for (int j = 0; j < d; j++) {
                row[j] -= ui * u[j];
            }
        }
    }
    if (tid == 0) {
        bonus[b] = step == 0 ? 0.0f : raw;
        ep_step[a] = reset ? 0 : step + 1;
    }
}

__global__ void ir_reset_cinv_kernel(float* __restrict__ cinv, int A, int d, float ridge) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)A * d * d) {
        return;
    }
    int i = (idx / d) % d, j = idx % d;
    cinv[idx] = i == j ? 1.0f / ridge : 0.0f;
}

// Forward-model input [phi_t ; onehot(a_t)] with unused action heads zeroed.
__global__ void ir_forward_input_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ phi, const int* __restrict__ act,
        const int* __restrict__ act_sizes, const int* __restrict__ act_offsets,
        int rows, int d, int act_n, int heads) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int width = d + act_n;
    if (idx >= (long)rows * width) {
        return;
    }
    int row = idx / width, col = idx % width;
    float v;
    if (col < d) {
        v = to_float(phi[(long)row * d + col]);
    } else {
        int c = col - d;
        v = 0.0f;
        for (int h = 0; h < heads; h++) {
            if (c >= act_offsets[h] && c < act_offsets[h] + act_sizes[h]) {
                int a = act[(long)row * heads + h];
                int used = 1;
#ifdef PUFFER_NETHACK
                used = nethack_head_used(act[(long)row * heads], h);
#endif
                v = (used && c - act_offsets[h] == a) ? 1.0f : 0.0f;
            }
        }
    }
    dst[idx] = from_float(v);
}

// Inverse-dynamics loss: per-head cross-entropy over the joint logits row,
// masked by head usage; grad = (softmax - onehot) * scale.
__global__ void ir_inverse_ce_kernel(float* __restrict__ loss,
        precision_t* __restrict__ grad, const precision_t* __restrict__ logits,
        const int* __restrict__ act, const int* __restrict__ act_sizes,
        const int* __restrict__ act_offsets, const float* __restrict__ valid,
        int rows, int act_n, int heads, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const precision_t* lg = logits + (long)row * act_n;
    precision_t* g = grad + (long)row * act_n;
    float w = valid[row];
    float total = 0.0f;
    for (int h = 0; h < heads; h++) {
        int off = act_offsets[h], n = act_sizes[h];
        int a = act[(long)row * heads + h];
        int used = 1;
#ifdef PUFFER_NETHACK
        used = nethack_head_used(act[(long)row * heads], h);
#endif
        float m = -INFINITY;
        for (int c = 0; c < n; c++) {
            m = fmaxf(m, to_float(lg[off + c]));
        }
        float s = 0.0f;
        for (int c = 0; c < n; c++) {
            s += expf(to_float(lg[off + c]) - m);
        }
        float lse = m + logf(s);
        float wh = used ? w : 0.0f;
        total += wh * (lse - to_float(lg[off + a]));
        for (int c = 0; c < n; c++) {
            float p = expf(to_float(lg[off + c]) - lse);
            g[off + c] = from_float(wh * scale * (p - (c == a ? 1.0f : 0.0f)));
        }
    }
    loss[row] = total;
}

// Gather training rows from the (T, B) rollout slot: x2 = [obs_t ; obs_{t+1}],
// actions of step t, valid = 1 unless the successor starts a new episode.
__global__ void ir_gather_kernel(precision_t* __restrict__ x2, int* __restrict__ act,
        float* __restrict__ valid, const int* __restrict__ rows_tb,
        const precision_t* __restrict__ obs, const float* __restrict__ actions,
        const precision_t* __restrict__ terminals, int R, int A, int obs_size,
        int heads, int mask_resets) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)2 * R * obs_size) {
        return;
    }
    int row = idx / obs_size, k = idx % obs_size;
    int r = row % R, next = row / R;
    int t = rows_tb[2 * r], b = rows_tb[2 * r + 1];
    x2[idx] = obs[((long)(t + next) * A + b) * obs_size + k];
    if (k == 0 && next == 0) {
        for (int h = 0; h < heads; h++) {
            act[(long)r * heads + h] = (int)actions[((long)t * A + b) * heads + h];
        }
        int reset = to_float(terminals[(long)(t + 1) * A + b]) != 0.0f;
        valid[r] = (mask_resets && reset) ? 0.0f : 1.0f;
    }
}

__global__ void ir_actions_kernel(int* __restrict__ dst, const float* __restrict__ src,
        int B, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * heads) {
        dst[idx] = (int)src[idx];
    }
}

__global__ void ir_concat_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ a, const precision_t* __restrict__ b,
        int rows, int da, int db) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int width = da + db;
    if (idx >= (long)rows * width) {
        return;
    }
    int row = idx / width, col = idx % width;
    dst[idx] = col < da ? a[(long)row * da + col] : b[(long)row * db + col - da];
}

__global__ void ir_split_kernel(precision_t* __restrict__ a, precision_t* __restrict__ b,
        const precision_t* __restrict__ src, int rows, int da, int db) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int width = da + db;
    if (idx >= (long)rows * width) {
        return;
    }
    int row = idx / width, col = idx % width;
    if (col < da) {
        a[(long)row * da + col] = src[idx];
    } else if (b) {
        b[(long)row * db + col - da] = src[idx];
    }
}

// Sequential normalizer over the (T, B) tensor in lane chunks. Torchbeast:
// chunk = 8 lanes, count += T per chunk. Running: one chunk, count += T*B.
// One block; rows of the chunk are reduced in parallel, chunks in order.
__global__ void ir_normalize_kernel(float* __restrict__ out, double* __restrict__ stats,
        const float* __restrict__ raw, int T, int B, int chunk_lanes, int mode) {
    __shared__ float red[256];
    __shared__ float mean_s;
    int tid = threadIdx.x;
    double sum = stats[0], m2 = stats[1], count = stats[2];
    int chunks = (B + chunk_lanes - 1) / chunk_lanes;
    for (int c = 0; c < chunks; c++) {
        int b0 = c * chunk_lanes, b1 = min(B, b0 + chunk_lanes);
        int n = (b1 - b0) * T;
        float s = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) {
            s += raw[(long)(i / (b1 - b0)) * B + b0 + i % (b1 - b0)];
        }
        red[tid] = s;
        block_reduce_sum(red, red, tid, blockDim.x, 1);
        float batch_sum = red[0];
        // Released quirk: the batch count is the unroll length, not T * lanes.
        double batch_count = mode == IR_NORM_TORCHBEAST ? (double)T : (double)n;
        double batch_mean = batch_sum / batch_count;
        if (tid == 0) {
            mean_s = (float)batch_mean;
        }
        __syncthreads();
        float q = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) {
            float d = raw[(long)(i / (b1 - b0)) * B + b0 + i % (b1 - b0)] - mean_s;
            q += d * d;
        }
        red[tid] = q;
        block_reduce_sum(red, red, tid, blockDim.x, 1);
        double batch_m2 = red[0];
        double old_mean = count > 0 ? sum / count : 0.0;
        double total = count + batch_count;
        double new_m2 = m2 + batch_m2
            + count * batch_count / total * (batch_mean - old_mean) * (batch_mean - old_mean);
        double var = new_m2 / total;
        float inv = (float)(1.0 / sqrt(var + 1e-8));
        for (int i = tid; i < n; i += blockDim.x) {
            long at = (long)(i / (b1 - b0)) * B + b0 + i % (b1 - b0);
            out[at] = raw[at] * inv;
        }
        sum += batch_sum;
        m2 = new_m2;
        count = total;
        __syncthreads();
    }
    if (tid == 0) {
        stats[0] = sum;
        stats[1] = m2;
        stats[2] = count;
    }
}

__global__ void ir_combine_kernel(float* __restrict__ out, const float* __restrict__ a,
        const float* __restrict__ b, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = b ? a[idx] * b[idx] : a[idx];
    }
}

__global__ void ir_apply_kernel(precision_t* __restrict__ rewards,
        float* __restrict__ stats, const float* __restrict__ normalized,
        const float* __restrict__ raw, int B, int T, float coef, int replace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) {
        return;
    }
    int b = idx / T, t = idx % T;
    float r = coef * normalized[(long)t * B + b];
    rewards[idx] = from_float(replace ? r : to_float(rewards[idx]) + r);
    if (t > 0) {
        atomicAdd(&stats[IR_STAT_BONUS], raw[(long)t * B + b]);
        atomicAdd(&stats[IR_STAT_NORMALIZED], r);
        atomicAdd(&stats[IR_STAT_ROWS], 1.0f);
    }
}

// RMSProp (torch semantics: eps outside the root) and Adam on fp32 master
// weights after global-norm clipping.
__global__ void ir_rmsprop_kernel(float* __restrict__ w, float* __restrict__ sq,
        const precision_t* __restrict__ g, const float* __restrict__ sum_sq,
        float max_norm, float lr, float alpha, float eps, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float clip = fminf(max_norm / (sqrtf(*sum_sq) + 1e-6f), 1.0f);
    float gi = to_float(g[idx]) * clip;
    float s = alpha * sq[idx] + (1.0f - alpha) * gi * gi;
    sq[idx] = s;
    w[idx] -= lr * gi / (sqrtf(s) + eps);
}

// ---------------------------------------------------------------------------
// Parameter groups, feature nets and MLPs
// ---------------------------------------------------------------------------

static float* to_host_float(const precision_t* dev, long n) {
    float* out = (float*)malloc(n * sizeof(float));
    precision_t* tmp = (precision_t*)malloc(n * sizeof(precision_t));
    cudaMemcpy(tmp, dev, n * sizeof(precision_t), cudaMemcpyDeviceToHost);
    for (long i = 0; i < n; i++) {
        out[i] = to_float(tmp[i]);
    }
    free(tmp);
    return out;
}

static Encoder ir_trunk(int in_dim, int out_dim) {
    Encoder e = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .in_dim = in_dim, .out_dim = out_dim,
        .activation_size = sizeof(EncoderActivations),
    };
    create_custom_encoder(&e);
    return e;
}

static void featnet_reg_params(FeatNet* f, Allocator* params) {
    f->weights = f->enc.create_weights(&f->enc);
    f->enc.reg_params(f->weights, params);
    f->w_out = {.shape = {f->out, f->hidden}};
    alloc_register(params, &f->w_out);
    if (f->layernorm) {
        f->ln_gamma = {.shape = {1, f->out}};
        f->ln_beta = {.shape = {1, f->out}};
        alloc_register(params, &f->ln_gamma);
        alloc_register(params, &f->ln_beta);
    }
}

static void featnet_reg_grads(FeatNet* f, FeatGrads* g, Allocator* grads) {
    g->w_out = {.shape = {f->out, f->hidden}};
    alloc_register(grads, &g->w_out);
    if (f->layernorm) {
        g->ln_gamma = {.shape = {1, f->out}};
        g->ln_beta = {.shape = {1, f->out}};
        alloc_register(grads, &g->ln_gamma);
        alloc_register(grads, &g->ln_beta);
    }
}

// Rollout activations (no gradients) or train activations (trunk grads land
// in grads, registered right after the trunk params so the flat layouts match).
static void featnet_reg_acts(FeatNet* f, FeatActs* a, FeatGrads* g, Allocator* acts,
        Allocator* grads, int rows, int train) {
    *a = (FeatActs){};
    a->rows = rows;
    a->train = train;
    a->trunk = calloc(1, f->enc.activation_size);
    if (train) {
        f->enc.reg_train(f->weights, a->trunk, acts, grads, rows);
        featnet_reg_grads(f, g, grads);
    } else {
        f->enc.reg_rollout(f->weights, a->trunk, acts, rows);
    }
    a->pre = {.shape = {rows, f->out}};
    a->f = {.shape = {rows, f->out}};
    a->mean = {.shape = {rows}};
    a->rstd = {.shape = {rows}};
    alloc_register(acts, &a->pre);
    alloc_register(acts, &a->f);
    alloc_register(acts, &a->mean);
    alloc_register(acts, &a->rstd);
    if (train) {
        a->d_pre = {.shape = {rows, f->out}};
        a->d_h = {.shape = {rows, f->hidden}};
        alloc_register(acts, &a->d_pre);
        alloc_register(acts, &a->d_h);
    }
}

static Prec featnet_forward(FeatNet* f, FeatActs* a, Prec x, cudaStream_t stream) {
    a->h = f->enc.forward(f->weights, a->trunk, x, stream);
    if (!f->layernorm) {
        puf_mm(&a->h, &f->w_out, &a->f, stream);
        return a->f;
    }
    puf_mm(&a->h, &f->w_out, &a->pre, stream);
    ir_layernorm_kernel<<<a->rows, 256, 0, stream>>>(a->f.data, a->mean.data,
        a->rstd.data, a->pre.data, f->ln_gamma.data, f->ln_beta.data, f->out);
    return a->f;
}

// d_f (rows, out) -> trunk and head gradients. dgamma/dbeta use fixed-point
// scratch borrowed from d_h (cleared first; the trunk backward runs after).
static void featnet_backward(FeatNet* f, FeatActs* a, FeatGrads* g, Prec d_f,
        cudaStream_t stream) {
    Prec d_pre = d_f;
    if (f->layernorm) {
        long long* scratch = (long long*)a->d_h.data;
        cudaMemsetAsync(scratch, 0, 2 * f->out * sizeof(long long), stream);
        ir_layernorm_backward_kernel<<<a->rows, 256, 0, stream>>>(a->d_pre.data,
            scratch, scratch + f->out, d_f.data, a->pre.data, f->ln_gamma.data,
            a->mean.data, a->rstd.data, f->out);
        t2_fxp_to_precision_kernel<<<grid_size(f->out), BLOCK_SIZE, 0, stream>>>(
            g->ln_gamma.data, scratch, f->out);
        t2_fxp_to_precision_kernel<<<grid_size(f->out), BLOCK_SIZE, 0, stream>>>(
            g->ln_beta.data, scratch + f->out, f->out);
        d_pre = a->d_pre;
    }
    puf_mm_tn(&d_pre, &a->h, &g->w_out, stream);
    puf_mm_nn(&d_pre, &f->w_out, &a->d_h, stream);
    f->enc.backward(f->weights, a->trunk, a->d_h, stream);
}

static void mlp_reg_params(Mlp* m, Allocator* params) {
    m->w1 = {.shape = {m->hidden, m->in}};
    m->w2 = {.shape = {m->out, m->hidden}};
    alloc_register(params, &m->w1);
    alloc_register(params, &m->w2);
}

static void mlp_reg_grads(Mlp* m, MlpGrads* g, Allocator* grads) {
    g->w1 = {.shape = {m->hidden, m->in}};
    g->w2 = {.shape = {m->out, m->hidden}};
    alloc_register(grads, &g->w1);
    alloc_register(grads, &g->w2);
}

static void mlp_reg_acts(Mlp* m, MlpActs* a, Allocator* acts, int rows, int train) {
    *a = (MlpActs){};
    a->h = {.shape = {rows, m->hidden}};
    a->y = {.shape = {rows, m->out}};
    alloc_register(acts, &a->h);
    alloc_register(acts, &a->y);
    if (train) {
        a->d_h = {.shape = {rows, m->hidden}};
        a->d_x = {.shape = {rows, m->in}};
        alloc_register(acts, &a->d_h);
        alloc_register(acts, &a->d_x);
    }
}

static Prec mlp_forward(Mlp* m, MlpActs* a, Prec x, cudaStream_t stream) {
    a->x = x;
    puf_mm(&x, &m->w1, &a->h, stream);
    long n = numel(a->h.shape);
    ir_relu_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->h.data, n);
    puf_mm(&a->h, &m->w2, &a->y, stream);
    return a->y;
}

// Returns the input gradient in a->d_x.
static Prec mlp_backward(Mlp* m, MlpActs* a, MlpGrads* g, Prec d_y, cudaStream_t stream) {
    puf_mm_tn(&d_y, &a->h, &g->w2, stream);
    puf_mm_nn(&d_y, &m->w2, &a->d_h, stream);
    long n = numel(a->h.shape);
    ir_relu_backward_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->d_h.data, a->h.data, n);
    puf_mm_tn(&a->d_h, &a->x, &g->w1, stream);
    puf_mm_nn(&a->d_h, &m->w1, &a->d_x, stream);
    return a->d_x;
}

static void param_group_create(ParamGroup* pg, Allocator* acts) {
    alloc_create(&pg->params);
    alloc_create(&pg->grads);
    long n = pg->params.total_elems;
    assert(pg->grads.total_elems == n && "parameter and gradient layouts differ");
    pg->param = {.data = (precision_t*)pg->params.mem, .shape = {n}};
    pg->grad = {.data = (precision_t*)pg->grads.mem, .shape = {n}};
    pg->master = {.shape = {n}};
    if (USE_BF16) {
        cudaMalloc((void**)&pg->master.data, n * sizeof(float));
    } else {
        pg->master.data = (float*)pg->param.data;
    }
    pg->m = {.shape = {n}};
    pg->v = {.shape = {n}};
    cudaMalloc((void**)&pg->m.data, n * sizeof(float));
    cudaMalloc((void**)&pg->v.data, n * sizeof(float));
    cudaMemset(pg->m.data, 0, n * sizeof(float));
    cudaMemset(pg->v.data, 0, n * sizeof(float));
    pg->partials = {.shape = {256}};
    alloc_register(acts, &pg->partials);
    cudaMalloc((void**)&pg->norm, sizeof(float));
}

static void param_group_sync_master(ParamGroup* pg, cudaStream_t stream) {
    long n = numel(pg->param.shape);
    if (USE_BF16) {
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(pg->master.data, pg->param.data, n);
    }
}

static void param_group_step(ParamGroup* pg, cudaStream_t stream) {
    long n = numel(pg->grad.shape);
    int blocks = min((int)grid_size(n), 256);
    muon_sum_sq_partials<<<blocks, 256, 0, stream>>>(pg->partials.data, pg->grad.data, n);
    muon_sum_sq_reduce<<<1, 256, 0, stream>>>(pg->norm, pg->partials.data, blocks);
    pg->step++;
    if (pg->optimizer == IR_OPT_ADAM) {
        float bc1 = 1.0f - powf(pg->beta1, (float)pg->step);
        float bc2 = 1.0f - powf(pg->beta2, (float)pg->step);
        t2_adam_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(pg->master.data,
            pg->m.data, pg->v.data, pg->grad.data, pg->norm, pg->max_norm, pg->lr,
            pg->beta1, pg->beta2, pg->eps, bc1, bc2, n);
    } else {
        ir_rmsprop_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(pg->master.data,
            pg->v.data, pg->grad.data, pg->norm, pg->max_norm, pg->lr, pg->alpha,
            pg->eps, n);
    }
    if (USE_BF16) {
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(pg->param.data, pg->master.data, n);
    }
}

static void param_group_config(ParamGroup* pg, Ini* ini, const char* section) {
    const char* opt = puf_ini_get_str(ini, section, "optimizer");
    if (strcmp(opt, "adam") == 0) {
        pg->optimizer = IR_OPT_ADAM;
    } else if (strcmp(opt, "rmsprop") == 0) {
        pg->optimizer = IR_OPT_RMSPROP;
    } else {
        fprintf(stderr, "[%s] optimizer must be adam or rmsprop, got %s\n", section, opt);
        exit(1);
    }
    pg->lr = puf_ini_get(ini, section, "learning_rate");
    pg->beta1 = puf_ini_get(ini, section, "adam_beta1");
    pg->beta2 = puf_ini_get(ini, section, "adam_beta2");
    pg->eps = puf_ini_get(ini, section, "eps");
    pg->alpha = puf_ini_get(ini, section, "rmsprop_alpha");
    pg->max_norm = puf_ini_get(ini, "intrinsic", "max_grad_norm");
}

static int ir_norm_mode(Ini* ini, const char* section) {
    const char* s = puf_ini_get_str(ini, section, "normalize");
    if (strcmp(s, "none") == 0) {
        return IR_NORM_NONE;
    }
    if (strcmp(s, "torchbeast") == 0) {
        return IR_NORM_TORCHBEAST;
    }
    if (strcmp(s, "running") == 0) {
        return IR_NORM_RUNNING;
    }
    fprintf(stderr, "[%s] normalize must be none, torchbeast or running\n", section);
    exit(1);
}

static void ir_init_featnet(FeatNet* f, ulong* seed, cudaStream_t stream) {
    f->enc.init_weights(f->weights, seed, stream);
    puf_kaiming_init(&f->w_out, 1.0f, (*seed)++, stream);
    if (f->layernorm) {
        // gamma = 1, beta = 0
        float* ones = (float*)malloc(f->out * sizeof(float));
        for (int i = 0; i < f->out; i++) {
            ones[i] = 1.0f;
        }
        float* dev;
        cudaMalloc((void**)&dev, f->out * sizeof(float));
        cudaMemcpy(dev, ones, f->out * sizeof(float), cudaMemcpyHostToDevice);
        cast<<<grid_size(f->out), BLOCK_SIZE, 0, stream>>>(f->ln_gamma.data, dev, f->out);
        cudaStreamSynchronize(stream);
        cudaFree(dev);
        free(ones);
    }
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

static int ir_has_idm(int method) {
    return method == IR_E3B || method == IR_ICM || method == IR_E3B_RND;
}

static int ir_has_rnd(int method) {
    return method == IR_RND || method == IR_E3B_RND;
}

Baseline* baseline_create(PuffeRL* p, Ini* ini, int method) {
    assert(PUF_BACKEND == PUF_CPU && "intrinsic baselines support CPU env backends");
    assert(p->num_policies == 1 && !p->is_continuous
        && "intrinsic baselines require one trainable discrete policy");
    VecEnv* vec = p->vec;
    assert(vec->size == vec->total_agents && "intrinsic baselines require one agent per env");
    Baseline* bl = (Baseline*)calloc(1, sizeof(Baseline));
    bl->method = method;
    bl->A = vec->total_agents;
    bl->num_buffers = p->hypers.num_buffers;
    bl->horizon = p->hypers.horizon;
    bl->slots = p->async_num_slots;
    bl->obs_size = OBS_SIZE;
    bl->hidden = p->hypers.hidden_size;
    bl->d = puf_ini_get(ini, "intrinsic", "feature_dim");
    bl->batch_rows = puf_ini_get(ini, "intrinsic", "batch_rows");
    bl->steps_per_epoch = puf_ini_get(ini, "intrinsic", "steps_per_epoch");
    bl->mask_resets = puf_ini_get(ini, "intrinsic", "mask_resets") != 0;
    bl->intrinsic_only = puf_ini_get(ini, "intrinsic", "intrinsic_only") != 0;
    bl->rng = (unsigned int)p->seed * 2246822519u + 77u;
    int sizes[] = ACT_SIZES;
    bl->num_heads = NUM_ATNS;
    bl->act_n = 0;
    for (int h = 0; h < NUM_ATNS; h++) {
        bl->act_sizes[h] = sizes[h];
        bl->act_offsets[h] = bl->act_n;
        bl->act_n += sizes[h];
    }
    assert(bl->A % bl->num_buffers == 0);
    assert(bl->batch_rows >= 1 && bl->d >= 1 && bl->d <= 1024);
    int B = bl->A / bl->num_buffers;
    int R = bl->batch_rows;
    int hidden_mlp = puf_ini_get(ini, "intrinsic", "hidden_dim");
    int layernorm = puf_ini_get(ini, "intrinsic", "layernorm") != 0;
    Allocator* acts = &bl->buf_alloc;
    ulong seed = p->seed * 104729 + 31;

    if (ir_has_idm(method)) {
        const char* sec = method == IR_ICM ? "icm" : "e3b";
        bl->feat = (FeatNet){.enc = ir_trunk(bl->obs_size, bl->hidden),
            .hidden = bl->hidden, .out = bl->d, .layernorm = layernorm};
        bl->inverse = (Mlp){.in = 2 * bl->d, .hidden = hidden_mlp, .out = bl->act_n};
        bl->forward = (Mlp){.in = bl->d + bl->act_n, .hidden = hidden_mlp, .out = bl->d};
        featnet_reg_params(&bl->feat, &bl->idm.params);
        mlp_reg_params(&bl->inverse, &bl->idm.params);
        if (method == IR_ICM) {
            mlp_reg_params(&bl->forward, &bl->idm.params);
        }
        // Train activations register the trunk grads first, matching params.
        featnet_reg_acts(&bl->feat, &bl->feat_tr, &bl->feat_g, acts, &bl->idm.grads, 2 * R, 1);
        mlp_reg_grads(&bl->inverse, &bl->inverse_g, &bl->idm.grads);
        if (method == IR_ICM) {
            mlp_reg_grads(&bl->forward, &bl->forward_g, &bl->idm.grads);
        }
        mlp_reg_acts(&bl->inverse, &bl->inverse_tr, acts, R, 1);
        mlp_reg_acts(&bl->forward, &bl->forward_tr, acts, R, 1);
        param_group_config(&bl->idm, ini, sec);
        param_group_create(&bl->idm, acts);
        bl->ridge = puf_ini_get(ini, "e3b", "ridge");
        bl->e3b_norm = ir_norm_mode(ini, "e3b");
        bl->icm_norm = ir_norm_mode(ini, "icm");
        bl->forward_coef = puf_ini_get(ini, "icm", "forward_coef");
        bl->inverse_coef = puf_ini_get(ini, "icm", "inverse_coef");
        bl->coef = puf_ini_get(ini, sec, "coef");
        if (method != IR_ICM) {
            bl->inverse_coef = 1.0f;
        }
    }
    if (ir_has_rnd(method)) {
        bl->target = (FeatNet){.enc = ir_trunk(bl->obs_size, bl->hidden),
            .hidden = bl->hidden, .out = (int)puf_ini_get(ini, "rnd", "output_dim"),
            .layernorm = 0};
        bl->predictor = bl->target;
        featnet_reg_params(&bl->target, &bl->target_params);
        featnet_reg_params(&bl->predictor, &bl->rnd.params);
        featnet_reg_acts(&bl->predictor, &bl->pred_tr, &bl->pred_g, acts, &bl->rnd.grads, R, 1);
        FeatGrads unused = {};
        featnet_reg_acts(&bl->target, &bl->target_tr, &unused, acts, NULL, R, 0);
        param_group_config(&bl->rnd, ini, "rnd");
        param_group_create(&bl->rnd, acts);
        alloc_create(&bl->target_params);
        bl->target_param = {.data = (precision_t*)bl->target_params.mem,
            .shape = {bl->target_params.total_elems}};
        bl->rnd_norm = ir_norm_mode(ini, "rnd");
        if (method == IR_RND) {
            bl->coef = puf_ini_get(ini, "rnd", "coef");
        }
    }

    // Rollout activations per buffer.
    bl->x2 = (Prec*)calloc(bl->num_buffers, sizeof(Prec));
    bl->feat_roll = (FeatActs*)calloc(bl->num_buffers, sizeof(FeatActs));
    bl->target_roll = (FeatActs*)calloc(bl->num_buffers, sizeof(FeatActs));
    bl->pred_roll = (FeatActs*)calloc(bl->num_buffers, sizeof(FeatActs));
    bl->forward_roll = (MlpActs*)calloc(bl->num_buffers, sizeof(MlpActs));
    bl->fwd_in_roll = (Prec*)calloc(bl->num_buffers, sizeof(Prec));
    bl->act_roll = (Int*)calloc(bl->num_buffers, sizeof(Int));
    bl->err_roll = (Float*)calloc(bl->num_buffers, sizeof(Float));
    FeatGrads none = {};
    for (int b = 0; b < bl->num_buffers; b++) {
        bl->x2[b] = {.shape = {2 * B, bl->obs_size}};
        alloc_register(acts, &bl->x2[b]);
        bl->act_roll[b] = {.shape = {B, bl->num_heads}};
        alloc_register(acts, &bl->act_roll[b]);
        bl->err_roll[b] = {.shape = {2 * B}};
        alloc_register(acts, &bl->err_roll[b]);
        if (ir_has_idm(method)) {
            featnet_reg_acts(&bl->feat, &bl->feat_roll[b], &none, acts, NULL, 2 * B, 0);
            mlp_reg_acts(&bl->forward, &bl->forward_roll[b], acts, B, 0);
            bl->fwd_in_roll[b] = {.shape = {B, bl->d + bl->act_n}};
            alloc_register(acts, &bl->fwd_in_roll[b]);
        }
        if (ir_has_rnd(method)) {
            featnet_reg_acts(&bl->target, &bl->target_roll[b], &none, acts, NULL, B, 0);
            featnet_reg_acts(&bl->predictor, &bl->pred_roll[b], &none, acts, NULL, B, 0);
        }
    }
    // Lane and slot buffers.
    long rows = (long)bl->slots * bl->horizon;
    bl->bonus_e3b = {.shape = {rows, bl->A}};
    bl->bonus_rnd = {.shape = {rows, bl->A}};
    bl->bonus_icm = {.shape = {rows, bl->A}};
    bl->combined = {.shape = {bl->horizon, bl->A}};
    bl->normalized = {.shape = {bl->horizon, bl->A}};
    bl->ep_step = {.shape = {bl->A}};
    alloc_register(acts, &bl->bonus_e3b);
    alloc_register(acts, &bl->bonus_rnd);
    alloc_register(acts, &bl->bonus_icm);
    alloc_register(acts, &bl->combined);
    alloc_register(acts, &bl->normalized);
    alloc_register(acts, &bl->ep_step);
    cudaMalloc((void**)&bl->norm_state, 3 * sizeof(double));
    cudaMemset(bl->norm_state, 0, 3 * sizeof(double));
    if (method == IR_E3B || method == IR_E3B_RND) {
        bl->cinv = {.shape = {bl->A, bl->d, bl->d}};
        alloc_register(acts, &bl->cinv);
    }
    // Training buffers.
    bl->tr_rows = {.shape = {R, 2}};
    bl->tr_x2 = {.shape = {2 * R, bl->obs_size}};
    bl->tr_act = {.shape = {R, bl->num_heads}};
    bl->tr_valid = {.shape = {R}};
    bl->inv_in = {.shape = {R, 2 * bl->d}};
    bl->fwd_in = {.shape = {R, bl->d + bl->act_n}};
    bl->d_f = {.shape = {2 * R, bl->d}};
    bl->d_fwd_in = {.shape = {R, bl->d + bl->act_n}};
    bl->loss_inv = {.shape = {R}};
    bl->loss_fwd = {.shape = {R}};
    bl->loss_rnd = {.shape = {R}};
    bl->tr_grad_y = {.shape = {R, bl->d}};
    bl->tr_grad_rnd = {.shape = {R, ir_has_rnd(method) ? bl->target.out : 1}};
    alloc_register(acts, &bl->loss_inv);
    alloc_register(acts, &bl->loss_fwd);
    alloc_register(acts, &bl->loss_rnd);
    alloc_register(acts, &bl->tr_grad_y);
    alloc_register(acts, &bl->tr_grad_rnd);
    alloc_register(acts, &bl->tr_rows);
    alloc_register(acts, &bl->tr_x2);
    alloc_register(acts, &bl->tr_act);
    alloc_register(acts, &bl->tr_valid);
    alloc_register(acts, &bl->inv_in);
    alloc_register(acts, &bl->fwd_in);
    alloc_register(acts, &bl->d_f);
    alloc_register(acts, &bl->d_fwd_in);
    alloc_create(acts);

    cudaStream_t stream = p->default_stream;
    if (ir_has_idm(method)) {
        ir_init_featnet(&bl->feat, &seed, stream);
        puf_kaiming_init(&bl->inverse.w1, sqrtf(2.0f), seed++, stream);
        puf_kaiming_init(&bl->inverse.w2, 1.0f, seed++, stream);
        if (method == IR_ICM) {
            puf_kaiming_init(&bl->forward.w1, sqrtf(2.0f), seed++, stream);
            puf_kaiming_init(&bl->forward.w2, 1.0f, seed++, stream);
        }
        param_group_sync_master(&bl->idm, stream);
        if (bl->cinv.data) {
            ir_reset_cinv_kernel<<<grid_size((long)bl->A * bl->d * bl->d), BLOCK_SIZE, 0,
                stream>>>(bl->cinv.data, bl->A, bl->d, bl->ridge);
        }
    }
    if (ir_has_rnd(method)) {
        ir_init_featnet(&bl->target, &seed, stream);
        ir_init_featnet(&bl->predictor, &seed, stream);
        param_group_sync_master(&bl->rnd, stream);
    }
    int* act_sizes_dev;
    int* act_offsets_dev;
    cudaMalloc((void**)&act_sizes_dev, NUM_ATNS * sizeof(int));
    cudaMalloc((void**)&act_offsets_dev, NUM_ATNS * sizeof(int));
    cudaMemcpy(act_sizes_dev, bl->act_sizes, NUM_ATNS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(act_offsets_dev, bl->act_offsets, NUM_ATNS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&bl->stats, IR_STAT_N * sizeof(float));
    cudaMemset(bl->stats, 0, IR_STAT_N * sizeof(float));
    bl->rollout_ms = (float*)calloc(bl->num_buffers, sizeof(float));
    bl->ev_start = (cudaEvent_t*)calloc(bl->num_buffers, sizeof(cudaEvent_t));
    bl->ev_end = (cudaEvent_t*)calloc(bl->num_buffers, sizeof(cudaEvent_t));
    bl->timed = (int*)calloc(bl->num_buffers, sizeof(int));
    for (int b = 0; b < bl->num_buffers; b++) {
        cudaEventCreate(&bl->ev_start[b]);
        cudaEventCreate(&bl->ev_end[b]);
    }
    // Device action layout tables live in static globals below.
    ir_act_sizes_dev = act_sizes_dev;
    ir_act_offsets_dev = act_offsets_dev;
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "intrinsic create: %s\n", cudaGetErrorString(err));
    }
    assert(err == cudaSuccess && "baseline create failed");
    const char* names[] = {"none", "t2", "e3b", "icm", "rnd", "e3b_rnd"};
    fprintf(stderr, "intrinsic: %s, features %d, lanes %d, buffers %.2f GB, coef %g\n",
        names[method], bl->d, bl->A, acts->total_bytes / 1e9, bl->coef);
    return bl;
}

// ---------------------------------------------------------------------------
// Rollout scoring, reward application, training, logging, checkpoints
// ---------------------------------------------------------------------------

__global__ void ir_axpy_kernel(precision_t* __restrict__ dst, const precision_t* __restrict__ src,
        float alpha, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) + alpha * to_float(src[idx]));
    }
}

// dst (rows, d) += src[:, :d] of a (rows, width) tensor.
__global__ void ir_add_slice_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int rows, int d, int width) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * d) {
        return;
    }
    int row = idx / d, col = idx % d;
    dst[idx] = from_float(to_float(dst[idx]) + to_float(src[(long)row * width + col]));
}

static Float ir_slot_view(Float x, int slot, int T) {
    long A = x.shape[1];
    return {.data = x.data + (long)slot * T * A, .shape = {T, A}};
}

// Per-step scoring for one vec buffer, after the env step uploaded o_{t+1}:
// features of [o_t ; o_{t+1}], the E3B bonus of the pre-action state (then
// its covariance update and terminal reset), the ICM forward error and the
// RND error of the successor, each stored on the rollout row of o_{t+1}.
void baseline_worker_step(PuffeRL* p, int buf, int t, cudaStream_t stream) {
    Baseline* bl = p->ir->bl;
    int B = bl->A / bl->num_buffers, agent0 = buf * B, T = bl->horizon;
    int obs = bl->obs_size, d = bl->d, heads = bl->num_heads;
    if (bl->timed[buf]) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, bl->ev_start[buf], bl->ev_end[buf]);
        bl->rollout_ms[buf] += ms;
        bl->timed[buf] = 0;
    }
    cudaEventRecord(bl->ev_start[buf], stream);
    RolloutBuf rollouts = p->rollouts;
    if (p->hypers.async) {
        rollouts = rollout_time_view(&p->rollouts, p->write_slot * T, T);
    }
    Prec x2 = bl->x2[buf];
    const precision_t* obs_t = rollouts.observations.data + ((long)t * bl->A + agent0) * obs;
    cudaMemcpyAsync(x2.data, obs_t, (size_t)B * obs * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
    ir_cast_obs_kernel<<<grid_size((long)B * obs), BLOCK_SIZE, 0, stream>>>(
        x2.data + (long)B * obs, p->env.obs.data + (long)agent0 * obs, (long)B * obs);
    const float* act_t = rollouts.actions.data + ((long)t * bl->A + agent0) * heads;
    ir_actions_kernel<<<grid_size(B * heads), BLOCK_SIZE, 0, stream>>>(
        bl->act_roll[buf].data, act_t, B, heads);
    const float* done = p->env.terminals.data + agent0;
    int row = (p->hypers.async ? p->write_slot * T : 0) + t + 1;
    int store = t + 1 < T;
    float* err = bl->err_roll[buf].data;
    if (ir_has_idm(bl->method)) {
        Prec phi = featnet_forward(&bl->feat, &bl->feat_roll[buf], x2, stream);
        Prec phi_next = {.data = phi.data + (long)B * d, .shape = {B, d}};
        if (bl->cinv.data) {
            size_t smem = (2 * d + 256) * sizeof(float);
            ir_e3b_kernel<<<B, 256, smem, stream>>>(err, bl->cinv.data, bl->ep_step.data,
                phi.data, done, agent0, d, bl->ridge);
            if (store) {
                cudaMemcpyAsync(bl->bonus_e3b.data + (long)row * bl->A + agent0, err,
                    B * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
        }
        if (bl->method == IR_ICM) {
            Prec fwd_in = bl->fwd_in_roll[buf];
            ir_forward_input_kernel<<<grid_size((long)B * (d + bl->act_n)), BLOCK_SIZE, 0,
                stream>>>(fwd_in.data, phi.data, bl->act_roll[buf].data, ir_act_sizes_dev,
                ir_act_offsets_dev, B, d, bl->act_n, heads);
            Prec y = mlp_forward(&bl->forward, &bl->forward_roll[buf], fwd_in, stream);
            ir_l2_kernel<<<B, 256, 0, stream>>>(err + B, NULL, y.data, phi_next.data,
                NULL, B, d, 0.0f);
            if (store) {
                cudaMemcpyAsync(bl->bonus_icm.data + (long)row * bl->A + agent0, err + B,
                    B * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
    if (ir_has_rnd(bl->method)) {
        Prec x_next = {.data = x2.data + (long)B * obs, .shape = {B, obs}};
        Prec ft = featnet_forward(&bl->target, &bl->target_roll[buf], x_next, stream);
        Prec fp = featnet_forward(&bl->predictor, &bl->pred_roll[buf], x_next, stream);
        ir_l2_kernel<<<B, 256, 0, stream>>>(err, NULL, fp.data, ft.data, NULL, B,
            bl->target.out, 0.0f);
        if (store) {
            cudaMemcpyAsync(bl->bonus_rnd.data + (long)row * bl->A + agent0, err,
                B * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
    }
    cudaEventRecord(bl->ev_end[buf], stream);
    bl->timed[buf] = 1;
}

// Combine, normalize and add the slot's bonuses into the (B, T) train view.
void baseline_apply_rewards(PuffeRL* p, RolloutBuf* train_view, int slot,
        cudaStream_t stream) {
    Baseline* bl = p->ir->bl;
    int T = bl->horizon, A = bl->A;
    long n = (long)T * A;
    Float e3b = ir_slot_view(bl->bonus_e3b, slot, T);
    Float rnd = ir_slot_view(bl->bonus_rnd, slot, T);
    Float icm = ir_slot_view(bl->bonus_icm, slot, T);
    const float* a = e3b.data;
    const float* b = NULL;
    int mode = bl->e3b_norm;
    if (bl->method == IR_ICM) {
        a = icm.data;
        mode = bl->icm_norm;
    } else if (bl->method == IR_RND) {
        a = rnd.data;
        mode = bl->rnd_norm;
    } else if (bl->method == IR_E3B_RND) {
        b = rnd.data;
    }
    ir_combine_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(bl->combined.data, a, b, n);
    if (mode == IR_NORM_NONE) {
        cudaMemcpyAsync(bl->normalized.data, bl->combined.data, n * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
    } else {
        int chunk = mode == IR_NORM_TORCHBEAST ? IR_LEARNER_LANES : A;
        ir_normalize_kernel<<<1, 256, 0, stream>>>(bl->normalized.data, bl->norm_state,
            bl->combined.data, T, A, chunk, mode);
    }
    ir_apply_kernel<<<grid_size(A * T), BLOCK_SIZE, 0, stream>>>(train_view->rewards.data,
        bl->stats, bl->normalized.data, bl->combined.data, A, T, bl->coef,
        bl->intrinsic_only);
}

static unsigned int ir_rand(Baseline* bl) {
    return (unsigned int)(puf_t2_mix(++bl->rng) >> 33);
}

// Sample batch_rows transitions (t < T-1, lane) into the device row list.
static void baseline_sample_rows(Baseline* bl, cudaStream_t stream) {
    int R = bl->batch_rows, T = bl->horizon, A = bl->A;
    int* rows = (int*)malloc(2 * R * sizeof(int));
    for (int r = 0; r < R; r++) {
        rows[2 * r] = ir_rand(bl) % (T - 1);
        rows[2 * r + 1] = ir_rand(bl) % A;
    }
    cudaMemcpyAsync(bl->tr_rows.data, rows, 2 * R * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    free(rows);
}

// Forward and backward over the sampled rows; gradients land in the groups'
// flat buffers. Returns the TorchBeast-scaled total loss on the host.
static float baseline_forward_backward(Baseline* bl, RolloutBuf* src, cudaStream_t stream) {
    int R = bl->batch_rows, T = bl->horizon, A = bl->A, d = bl->d, heads = bl->num_heads;
    ir_gather_kernel<<<grid_size((long)2 * R * bl->obs_size), BLOCK_SIZE, 0, stream>>>(
        bl->tr_x2.data, bl->tr_act.data, bl->tr_valid.data, bl->tr_rows.data,
        src->observations.data, src->actions.data, src->terminals.data, R, A,
        bl->obs_size, heads, bl->mask_resets);
    float inv_scale = bl->inverse_coef * (float)T / R;
    float fwd_scale = bl->forward_coef * (float)T / R;
    float rnd_scale = (float)T / R;
    if (ir_has_idm(bl->method)) {
        Prec phi = featnet_forward(&bl->feat, &bl->feat_tr, bl->tr_x2, stream);
        Prec phi_t = {.data = phi.data, .shape = {R, d}};
        Prec phi_n = {.data = phi.data + (long)R * d, .shape = {R, d}};
        ir_concat_kernel<<<grid_size((long)R * 2 * d), BLOCK_SIZE, 0, stream>>>(
            bl->inv_in.data, phi_t.data, phi_n.data, R, d, d);
        Prec logits = mlp_forward(&bl->inverse, &bl->inverse_tr, bl->inv_in, stream);
        ir_inverse_ce_kernel<<<grid_size(R), BLOCK_SIZE, 0, stream>>>(bl->loss_inv.data,
            logits.data, logits.data, bl->tr_act.data, ir_act_sizes_dev, ir_act_offsets_dev,
            bl->tr_valid.data, R, bl->act_n, heads, inv_scale);
        Prec d_inv_in = mlp_backward(&bl->inverse, &bl->inverse_tr, &bl->inverse_g, logits, stream);
        Prec d_f_t = {.data = bl->d_f.data, .shape = {R, d}};
        Prec d_f_n = {.data = bl->d_f.data + (long)R * d, .shape = {R, d}};
        ir_split_kernel<<<grid_size((long)R * 2 * d), BLOCK_SIZE, 0, stream>>>(
            d_f_t.data, d_f_n.data, d_inv_in.data, R, d, d);
        if (bl->method == IR_ICM) {
            ir_forward_input_kernel<<<grid_size((long)R * (d + bl->act_n)), BLOCK_SIZE, 0,
                stream>>>(bl->fwd_in.data, phi_t.data, bl->tr_act.data, ir_act_sizes_dev,
                ir_act_offsets_dev, R, d, bl->act_n, heads);
            Prec y = mlp_forward(&bl->forward, &bl->forward_tr, bl->fwd_in, stream);
            ir_l2_kernel<<<R, 256, 0, stream>>>(bl->loss_fwd.data, bl->tr_grad_y.data,
                y.data, phi_n.data, bl->tr_valid.data, R, d, fwd_scale);
            // The target phi(s_{t+1}) keeps its gradient (released code).
            ir_axpy_kernel<<<grid_size((long)R * d), BLOCK_SIZE, 0, stream>>>(
                d_f_n.data, bl->tr_grad_y.data, -1.0f, (long)R * d);
            Prec d_fwd_in = mlp_backward(&bl->forward, &bl->forward_tr, &bl->forward_g,
                bl->tr_grad_y, stream);
            ir_add_slice_kernel<<<grid_size((long)R * d), BLOCK_SIZE, 0, stream>>>(
                d_f_t.data, d_fwd_in.data, R, d, d + bl->act_n);
        }
        featnet_backward(&bl->feat, &bl->feat_tr, &bl->feat_g, bl->d_f, stream);
    }
    if (ir_has_rnd(bl->method)) {
        Prec x_n = {.data = bl->tr_x2.data + (long)R * bl->obs_size, .shape = {R, bl->obs_size}};
        Prec ft = featnet_forward(&bl->target, &bl->target_tr, x_n, stream);
        Prec fp = featnet_forward(&bl->predictor, &bl->pred_tr, x_n, stream);
        // Episode-boundary targets are not masked (released learner).
        ir_l2_kernel<<<R, 256, 0, stream>>>(bl->loss_rnd.data, bl->tr_grad_rnd.data,
            fp.data, ft.data, NULL, R, bl->target.out, rnd_scale);
        featnet_backward(&bl->predictor, &bl->pred_tr, &bl->pred_g, bl->tr_grad_rnd, stream);
    }
    float* h = (float*)malloc(3 * R * sizeof(float));
    cudaMemcpyAsync(h, bl->loss_inv.data, R * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h + R, bl->loss_fwd.data, R * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h + 2 * R, bl->loss_rnd.data, R * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double inv = 0.0, fwd = 0.0, rnd = 0.0;
    for (int r = 0; r < R; r++) {
        inv += h[r];
        fwd += h[R + r];
        rnd += h[2 * R + r];
    }
    free(h);
    bl->last_inv = (float)(inv / R);
    bl->last_fwd = (float)(fwd / R);
    bl->last_rnd = (float)(rnd / R);
    float total = 0.0f;
    if (ir_has_idm(bl->method)) {
        total += inv_scale * (float)inv;
    }
    if (bl->method == IR_ICM) {
        total += fwd_scale * (float)fwd;
    }
    if (ir_has_rnd(bl->method)) {
        total += rnd_scale * (float)rnd;
    }
    return total;
}

static void baseline_train_step(Baseline* bl, RolloutBuf* src, cudaStream_t stream) {
    baseline_sample_rows(bl, stream);
    baseline_forward_backward(bl, src, stream);
    if (ir_has_idm(bl->method)) {
        param_group_step(&bl->idm, stream);
    }
    if (ir_has_rnd(bl->method)) {
        param_group_step(&bl->rnd, stream);
    }
}

void baseline_train(PuffeRL* p) {
    Baseline* bl = p->ir->bl;
    cudaStream_t stream = p->train_stream;
    double t0 = t2_now();
    int slot = p->hypers.async ? p->async_ready_slot : 0;
    RolloutBuf src = rollout_time_view(&p->rollouts, slot * bl->horizon, bl->horizon);
    int steps = bl->steps_per_epoch;
    if (steps <= 0) {
        steps = ((long)bl->A * (bl->horizon - 1) + bl->batch_rows - 1) / bl->batch_rows;
    }
    double sum_inv = 0.0, sum_fwd = 0.0, sum_rnd = 0.0;
    for (int s = 0; s < steps; s++) {
        baseline_train_step(bl, &src, stream);
        sum_inv += bl->last_inv;
        sum_fwd += bl->last_fwd;
        sum_rnd += bl->last_rnd;
    }
    cudaStreamSynchronize(stream);
    assert(cudaGetLastError() == cudaSuccess && "intrinsic baseline train kernel failed");
    float cur[IR_STAT_N];
    cudaMemcpy(cur, bl->stats, sizeof(cur), cudaMemcpyDeviceToHost);
    cur[IR_STAT_INV_LOSS] += (float)(sum_inv / steps);
    cur[IR_STAT_FWD_LOSS] += (float)(sum_fwd / steps);
    cur[IR_STAT_RND_LOSS] += (float)(sum_rnd / steps);
    cur[IR_STAT_STEPS] += 1.0f;
    cudaMemcpy(bl->stats, cur, sizeof(cur), cudaMemcpyHostToDevice);
    bl->train_ms += (float)((t2_now() - t0) * 1000.0);
}

void baseline_log(PuffeRL* p, Dict* out) {
    Baseline* bl = p->ir->bl;
    float s[IR_STAT_N];
    cudaMemcpy(s, bl->stats, sizeof(s), cudaMemcpyDeviceToHost);
    cudaMemset(bl->stats, 0, sizeof(s));
    float inv_rows = s[IR_STAT_ROWS] > 0 ? 1.0f / s[IR_STAT_ROWS] : 0.0f;
    float inv_steps = s[IR_STAT_STEPS] > 0 ? 1.0f / s[IR_STAT_STEPS] : 0.0f;
    dict_set(out, "env/ir_bonus", s[IR_STAT_BONUS] * inv_rows);
    dict_set(out, "env/ir_reward", s[IR_STAT_NORMALIZED] * inv_rows);
    dict_set(out, "env/ir_inverse_loss", s[IR_STAT_INV_LOSS] * inv_steps);
    dict_set(out, "env/ir_forward_loss", s[IR_STAT_FWD_LOSS] * inv_steps);
    dict_set(out, "env/ir_rnd_loss", s[IR_STAT_RND_LOSS] * inv_steps);
    double norm_state[3];
    cudaMemcpy(norm_state, bl->norm_state, sizeof(norm_state), cudaMemcpyDeviceToHost);
    dict_set(out, "env/ir_norm_std", norm_state[2] > 0
        ? sqrt(norm_state[1] / norm_state[2] + 1e-8) : 0.0);
    float rollout_ms = 0.0f;
    for (int b = 0; b < bl->num_buffers; b++) {
        rollout_ms += bl->rollout_ms[b];
        bl->rollout_ms[b] = 0.0f;
    }
    dict_set(out, "perf/ir_rollout", rollout_ms / bl->num_buffers / 1000.0f);
    dict_set(out, "perf/ir_train", bl->train_ms / 1000.0f);
    bl->train_ms = 0.0f;
}

static void ir_write_group(FILE* fp, ParamGroup* pg) {
    long n = numel(pg->master.shape);
    float* buf = (float*)malloc(3 * n * sizeof(float));
    cudaMemcpy(buf, pg->master.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(buf + n, pg->m.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(buf + 2 * n, pg->v.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    assert(fwrite(&n, sizeof(n), 1, fp) == 1);
    assert(fwrite(&pg->step, sizeof(pg->step), 1, fp) == 1);
    assert(fwrite(buf, sizeof(float), 3 * n, fp) == (size_t)(3 * n));
    free(buf);
}

// Checkpoint: trainable groups (fp32 master + optimizer state), the frozen RND
// target and the reward normalizer state. Episodic E3B state is not saved.
void baseline_save(PuffeRL* p, const char* path) {
    Baseline* bl = p->ir->bl;
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "wb");
    assert(fp && "failed to open intrinsic weights for writing");
    assert(fwrite(&bl->method, sizeof(int), 1, fp) == 1);
    if (ir_has_idm(bl->method)) {
        ir_write_group(fp, &bl->idm);
    }
    if (ir_has_rnd(bl->method)) {
        ir_write_group(fp, &bl->rnd);
        long n = numel(bl->target_param.shape);
        float* buf = to_host_float(bl->target_param.data, n);
        assert(fwrite(&n, sizeof(n), 1, fp) == 1);
        assert(fwrite(buf, sizeof(float), n, fp) == (size_t)n);
        free(buf);
    }
    double norm_state[3];
    cudaMemcpy(norm_state, bl->norm_state, sizeof(norm_state), cudaMemcpyDeviceToHost);
    assert(fwrite(norm_state, sizeof(double), 3, fp) == 3);
    fclose(fp);
    assert(rename(tmp, path) == 0 && "failed to publish intrinsic weights");
}
