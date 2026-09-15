// Intrinsic reward tests: T2 (codec/pairing contract, kernels against
// references, literal control prefix, re-encoding, gradients, reservoir,
// checkpoints) and the paper-faithful baselines (E3B Sherman-Morrison against
// an explicit inverse, the TorchBeast normalizer against its formula, feature
// learner gradients, method smokes). Runs on the real trainer substrate with
// small craftax_classic configurations.
//   ./build.sh craftax_classic build/test_intrinsic --t2-test --float && ./build/test_intrinsic
// Gradient and likelihood checks need the float build; a bf16 build skips them.
#include "pufferl.cu"

static int checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        exit(1); \
    } \
} while (0)

static unsigned rng_state = 12345;
static unsigned lcg(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static float* to_host(const void* dev, long n, int elem) {
    float* out = (float*)malloc(n * sizeof(float));
    if (elem == sizeof(float)) {
        cudaMemcpy(out, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        precision_t* tmp = (precision_t*)malloc(n * sizeof(precision_t));
        cudaMemcpy(tmp, dev, n * sizeof(precision_t), cudaMemcpyDeviceToHost);
        for (long i = 0; i < n; i++) {
            out[i] = to_float(tmp[i]);
        }
        free(tmp);
    }
    return out;
}

static float* prec_host(Prec p) {
    return to_host(p.data, numel(p.shape), sizeof(precision_t));
}

static void prec_upload(precision_t* dev, const float* src, long n) {
    precision_t* tmp = (precision_t*)malloc(n * sizeof(precision_t));
    for (long i = 0; i < n; i++) {
        tmp[i] = from_float(src[i]);
    }
    cudaMemcpy(dev, tmp, n * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);
}

static float urand(float lo, float hi) {
    return lo + (hi - lo) * (float)(lcg() % 100000) / 100000.0f;
}

static void env_alloc(Env* env, unsigned lane, Dict* kwargs) {
    memset(env, 0, sizeof(Env));
    env->rng = lane;
    env->agents[0].observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    env->agents[0].actions = (float*)calloc(NUM_ATNS, sizeof(float));
    env->agents[0].rewards = (float*)calloc(1, sizeof(float));
    env->agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_init(env, kwargs);
}

#ifdef PUFFER_CRAFTAX_CLASSIC
// Every token except the light byte reproduces the float observation exactly.
static void test_codec(void) {
    Dict kwargs = {0};
    Env env;
    env_alloc(&env, 0, &kwargs);
    puf_reset(&env);
    unsigned char tok[PUF_T2_TOKENS];
    int steps = 0, terminals = 0;
    for (int step = 0; step < 400; step++) {
        puf_t2_tokens(&env, tok);
        float* obs = env.agents[0].observations;
        for (int i = 0; i < 63; i++) {
            int blk = tok[i];
            for (int b = 0; b < NUM_BLOCK_TYPES; b++) {
                CHECK(obs[i * 21 + b] == (b == blk ? 1.0f : 0.0f), "block %d tile %d", b, i);
            }
            int m = (tok[63 + i / 2] >> (4 * (i % 2))) & 15;
            for (int f = 0; f < 4; f++) {
                CHECK(obs[i * 21 + 17 + f] == ((m >> f) & 1 ? 1.0f : 0.0f), "mob %d tile %d", f, i);
            }
        }
        for (int k = 0; k < NUM_INVENTORY; k++) {
            CHECK(obs[1323 + k] == (float)tok[95 + k] * 0.1f, "inventory %d", k);
        }
        for (int k = 0; k < 4; k++) {
            CHECK(obs[1335 + k] == (float)tok[107 + k] * 0.1f, "intrinsic %d", k);
        }
        for (int d = 1; d <= 4; d++) {
            CHECK(obs[1338 + d] == (tok[111] == d ? 1.0f : 0.0f), "direction %d", d);
        }
        CHECK(fabsf(obs[1343] - tok[112] / 255.0f) <= 0.5f / 255.0f + 1e-6f, "light");
        CHECK(obs[1344] == (tok[113] ? 1.0f : 0.0f), "sleeping");
        env.agents[0].actions[0] = (float)(lcg() % NUM_ACTIONS);
        puf_step(&env);
        steps++;
        terminals += env.agents[0].terminals[0] != 0.0f;
    }
    printf("PASS codec: %d steps, %d terminals, %d tokens\n", steps, terminals, PUF_T2_TOKENS);
}

// Partner lanes share each episode's world and differ in dynamics; other
// pairs and other episodes get other worlds.
static void test_pairing(void) {
    Dict kwargs = {0};
    dict_set(&kwargs, "t2_seed", 7);
    Env a, b, c;
    env_alloc(&a, 0, &kwargs);
    env_alloc(&b, 1, &kwargs);
    env_alloc(&c, 2, &kwargs);
    puf_reset(&a);
    puf_reset(&b);
    puf_reset(&c);
    CHECK(memcmp(a.map_packed, b.map_packed, sizeof(a.map_packed)) == 0, "pair world");
    CHECK(memcmp(a.agents[0].observations, b.agents[0].observations,
        OBS_SIZE * sizeof(obs_t)) == 0, "pair initial observation");
    CHECK(memcmp(a.map_packed, c.map_packed, sizeof(a.map_packed)) != 0, "other pair");
    unsigned char first_world[MAP_PACKED_SIZE];
    memcpy(first_world, a.map_packed, sizeof(first_world));
    int differ = 0, done_a = 0, done_b = 0;
    for (int step = 0; step < 2000 && !(done_a && done_b); step++) {
        float act = (float)(lcg() % NUM_ACTIONS);
        if (!done_a) {
            a.agents[0].actions[0] = act;
            puf_step(&a);
            done_a = a.agents[0].terminals[0] != 0.0f;
        }
        if (!done_b) {
            b.agents[0].actions[0] = act;
            puf_step(&b);
            done_b = b.agents[0].terminals[0] != 0.0f;
        }
        if (!done_a && !done_b) {
            differ += memcmp(a.agents[0].observations, b.agents[0].observations,
                OBS_SIZE * sizeof(obs_t)) != 0;
        }
    }
    CHECK(done_a && done_b, "episodes end within 2000 steps");
    CHECK(differ > 0, "independent dynamics diverge");
    CHECK(a.t2_episode == 2 && b.t2_episode == 2, "episode counters");
    CHECK(memcmp(a.map_packed, b.map_packed, sizeof(a.map_packed)) == 0,
        "second episode shares the world after unsynchronized resets");
    CHECK(memcmp(a.map_packed, first_world, sizeof(first_world)) != 0,
        "second episode is a new world");
    Env d;
    env_alloc(&d, 0, &kwargs);
    puf_reset(&d);
    CHECK(memcmp(d.map_packed, first_world, sizeof(first_world)) == 0, "replay world");
    printf("PASS pairing: diverged on %d steps before the first terminal\n", differ);
}
#endif

#ifdef PUFFER_NETHACK
// Partner lanes boot the same character and dungeon from shared seeds and
// diverge afterwards; the codec's crop, status and message bytes reproduce
// the engine state they are cut from.
static void test_nethack_pairing_and_codec(Ini* ini) {
    Dict* kwargs = puf_ini_section(ini, "env", 0);
    dict_set(kwargs, "t2_seed", 11);
    Env a, b, c;
    env_alloc(&a, 0, kwargs);
    env_alloc(&b, 1, kwargs);
    env_alloc(&c, 2, kwargs);
    Env* envs[3] = {&a, &b, &c};
    for (int i = 0; i < 3; i++) {
        puf_reset(envs[i]);
        for (int h = 0; h < NUM_ATNS; h++) {
            envs[i]->agents[0].actions[h] = 0.0f;
        }
        envs[i]->agents[0].actions[0] = (float)NETHACK_ACT_SEARCH;
        puf_step(envs[i]);
    }
    int same_ab = 0, same_ac = 0;
    for (int cell = 0; cell < NH_GRID; cell++) {
        same_ab += a.glyphs[cell] == b.glyphs[cell];
        same_ac += a.glyphs[cell] == c.glyphs[cell];
    }
    CHECK(same_ab >= NH_GRID * 99 / 100, "pair shares the first level (%d/%d cells)", same_ab, NH_GRID);
    CHECK(same_ac < NH_GRID * 95 / 100, "other pair differs (%d/%d cells)", same_ac, NH_GRID);
    CHECK(a.role_idx == b.role_idx && a.race_idx == b.race_idx && a.gend_idx == b.gend_idx,
        "pair shares the character");
    CHECK(a.t2_episode == 1 && b.t2_episode == 1, "episode counters");
    unsigned char tok[PUF_T2_TOKENS];
    puf_t2_tokens(&a, tok);
    long hx = a.blstats[NLE_BL_X], hy = a.blstats[NLE_BL_Y];
    int k = 0, half = NETHACK_T2_CROP / 2;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            long x = hx + dx, y = hy + dy;
            int g = (x >= 0 && x < NH_COLS && y >= 0 && y < NH_ROWS)
                ? a.glyphs[y * NH_COLS + x] : NETHACK_PAD_GLYPH;
            CHECK((tok[k] | (tok[k + 1] << 8)) == g, "crop cell %d", k / 2);
            k += 2;
        }
    }
    CHECK(tok[k] == hx && tok[k + 1] == hy, "hero position bytes");
    CHECK(memcmp(tok + k + 16, a.message, NETHACK_T2_MSG) == 0, "message bytes");
    // Same actions, lane-specific dynamics: partners diverge eventually.
    int differ = 0;
    for (int step = 0; step < 300 && !differ; step++) {
        int verb = lcg() % 2 ? NETHACK_ACT_MOVE : NETHACK_ACT_SEARCH;
        int dir = lcg() % NETHACK_NUM_DIRS;
        for (int i = 0; i < 2; i++) {
            envs[i]->agents[0].actions[0] = (float)verb;
            envs[i]->agents[0].actions[13] = (float)dir;
            puf_step(envs[i]);
        }
        differ = memcmp(a.glyphs, b.glyphs, NH_GRID * sizeof(short)) != 0;
    }
    printf("PASS nethack pairing/codec: %d/%d shared cells, diverged=%d, %d tokens\n",
        same_ab, NH_GRID, differ, PUF_T2_TOKENS);
    for (int i = 0; i < 3; i++) {
        puf_close(envs[i]);
    }
}
#endif

static double host_gelu(double x) {
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
}

static PuffeRL* make_trainer(Ini* ini, const char* method,
        int buffers = 1, int layers = 1, int async = 0) {
    puf_ini_put(ini, "intrinsic.method", method);
    puf_ini_put(ini, "vec.total_agents", "32");
    char value[32];
    snprintf(value, sizeof(value), "%d", buffers);
    puf_ini_put(ini, "vec.num_buffers", value);
    puf_ini_put(ini, "vec.num_threads", "4");
    puf_ini_put(ini, "train.horizon", "16");
    puf_ini_put(ini, "train.minibatch_size", "512");
    snprintf(value, sizeof(value), "%d", async);
    puf_ini_put(ini, "base.async", value);
    puf_ini_put(ini, "base.cudagraphs", "-1");
    puf_ini_put(ini, "t2.embed_dim", "32");
    puf_ini_put(ini, "t2.hidden_size", "32");
    snprintf(value, sizeof(value), "%d", layers);
    puf_ini_put(ini, "t2.num_layers", value);
    puf_ini_put(ini, "t2.head_dim", "32");
    puf_ini_put(ini, "t2.max_episode", "64");
    puf_ini_put(ini, "t2.reservoir", "16");
    puf_ini_put(ini, "t2.wm_batch", "8");
    puf_ini_put(ini, "t2.bptt_window", "8");
    puf_ini_put(ini, "t2.wm_steps", "1");
    puf_ini_put(ini, "intrinsic.feature_dim", "32");
    puf_ini_put(ini, "intrinsic.hidden_dim", "32");
    puf_ini_put(ini, "intrinsic.batch_rows", "24");
    puf_ini_put(ini, "intrinsic.steps_per_epoch", "2");
    puf_ini_put(ini, "rnd.output_dim", "32");
    TrainContext ctx = {.world_size = 1, .artifact_owner = 0};
    return create_pufferl(ini, &ctx);
}

#include "test_t2_history.cuh"
#include "test_t2_record.cuh"
#include "test_t2_limits.cuh"

// ---------------------------------------------------------------------------
// T2
// ---------------------------------------------------------------------------

static void test_head_reference(PuffeRL* p) {
    T2* t2 = p->t2;
    T2Rollout* r = &t2->roll[0];
    int B = t2->A / t2->num_buffers, H = t2->H, D = t2->D, L = t2->L;
    float* z = prec_host(r->z);
    float* w_h = prec_host(t2->w.w_h);
    float* pos = prec_host(t2->w.pos_embed);
    float* w_out = prec_host(t2->w.w_out);
    float* nll = to_host(r->nll.data, 2 * B, sizeof(float));
    unsigned char* target = (unsigned char*)malloc((size_t)2 * B * L);
    cudaMemcpy(target, r->target2.data, (size_t)2 * B * L, cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    double* u = (double*)malloc(D * sizeof(double));
    double* logits = (double*)malloc(T2_VOCAB * sizeof(double));
    for (int row = 0; row < 2 * B; row++) {
        for (int d = 0; d < D; d++) {
            double s = 0.0;
            for (int k = 0; k < 2 * H; k++) {
                s += (double)z[row * 2 * H + k] * w_h[d * 2 * H + k];
            }
            u[d] = s;
        }
        double total = 0.0;
        for (int pp = 0; pp < L; pp++) {
            double m = -1e30;
            for (int v = 0; v < T2_VOCAB; v++) {
                double s = 0.0;
                for (int d = 0; d < D; d++) {
                    s += host_gelu(u[d] + pos[pp * D + d]) * w_out[v * D + d];
                }
                logits[v] = s;
                m = s > m ? s : m;
            }
            double lse = 0.0;
            for (int v = 0; v < T2_VOCAB; v++) {
                lse += exp(logits[v] - m);
            }
            lse = m + log(lse);
            total += lse - logits[target[row * L + pp]];
        }
        double err = fabs(total - nll[row]) / (1.0 + fabs(total));
        max_err = err > max_err ? err : max_err;
    }
    CHECK(max_err < 2e-4, "head reference mismatch %.3e", max_err);
    printf("PASS head reference: %d rows, max relative error %.2e\n", 2 * B, max_err);
    free(z); free(w_h); free(pos); free(w_out); free(nll); free(target); free(u); free(logits);
}

// The three-group gate kernel agrees with PufferLib's mingru_gate for each
// group's input state, and the control group starts from the pre-step state.
static void test_gate_consistency(PuffeRL* p) {
    T2* t2 = p->t2;
    int B = t2->A / t2->num_buffers, H = t2->H;
    long n3 = (long)3 * B * H;
    float* h_comb = (float*)malloc(3 * n3 * sizeof(float));
    float* h_x = (float*)malloc(n3 * sizeof(float));
    float* h_state = (float*)malloc(B * H * sizeof(float));
    for (long i = 0; i < 3 * n3; i++) h_comb[i] = urand(-2.0f, 2.0f);
    for (long i = 0; i < n3; i++) h_x[i] = urand(-2.0f, 2.0f);
    for (long i = 0; i < B * H; i++) h_state[i] = urand(-1.0f, 1.0f);
    precision_t *comb, *x, *state, *prev, *out, *ref_out, *ref_next;
    cudaMalloc((void**)&comb, 3 * n3 * sizeof(precision_t));
    cudaMalloc((void**)&x, n3 * sizeof(precision_t));
    cudaMalloc((void**)&state, B * H * sizeof(precision_t));
    cudaMalloc((void**)&prev, B * H * sizeof(precision_t));
    cudaMalloc((void**)&out, n3 * sizeof(precision_t));
    cudaMalloc((void**)&ref_out, B * H * sizeof(precision_t));
    cudaMalloc((void**)&ref_next, B * H * sizeof(precision_t));
    prec_upload(comb, h_comb, 3 * n3);
    prec_upload(x, h_x, n3);
    prec_upload(state, h_state, B * H);
    cudaMemset(prev, 0, B * H * sizeof(precision_t));
    t2_gate3_kernel<<<grid_size(B * H), BLOCK_SIZE>>>(out, state, prev, comb, x, B, H);
    cudaDeviceSynchronize();
    float* g_out = to_host(out, n3, sizeof(precision_t));
    float* g_state = to_host(state, B * H, sizeof(precision_t));
    float* g_prev = to_host(prev, B * H, sizeof(precision_t));
    // Query and control both step from the pre-step state; q1 from the new state.
    float* inputs[3] = {h_state, h_state, g_state};
    double max_err = 0.0;
    for (int g = 0; g < 3; g++) {
        prec_upload(state, inputs[g], B * H);
        mingru_gate<<<grid_size(B * H), BLOCK_SIZE>>>(ref_out, ref_next,
            comb + (long)g * B * 3 * H, state, x + (long)g * B * H, H, B);
        cudaDeviceSynchronize();
        float* ro = to_host(ref_out, B * H, sizeof(precision_t));
        float* rn = to_host(ref_next, B * H, sizeof(precision_t));
        for (long i = 0; i < B * H; i++) {
            double e = fabs(ro[i] - g_out[(long)g * B * H + i]);
            max_err = e > max_err ? e : max_err;
            if (g == 0) {
                double es = fabs(rn[i] - g_state[i]);
                max_err = es > max_err ? es : max_err;
            }
        }
        free(ro); free(rn);
    }
    for (long i = 0; i < B * H; i++) {
        CHECK(g_prev[i] == to_float(from_float(h_state[i])), "state_prev keeps the pre-step state");
    }
    CHECK(max_err < 1e-5, "gate3 vs mingru_gate mismatch %.3e", max_err);
    printf("PASS gate consistency: max abs error %.2e\n", max_err);
    free(h_comb); free(h_x); free(h_state); free(g_out); free(g_state); free(g_prev);
    cudaFree(comb); cudaFree(x); cudaFree(state); cudaFree(prev);
    cudaFree(out); cudaFree(ref_out); cudaFree(ref_next);
}

// Independent scalar recurrence for the control support rows of the last
// rollout step: from the saved pre-step state with the (o_t, PAD) inputs.
static void test_literal_control_prefix(PuffeRL* p) {
    T2* t2 = p->t2;
    T2Rollout* r = &t2->roll[0];
    int B = t2->A, H = t2->H;
    CHECK(t2->num_buffers == 1 && t2->Lg == 1, "audit geometry");
    float* before = prec_host(t2->state_prev);
    float* combined = prec_host(r->combined);
    float* x = prec_host(r->x);
    float* out = prec_host(r->out);
    float* terminals = to_host(p->env.terminals.data, B, sizeof(float));
    double maxerr = 0.0, reset_error = 0.0;
    int compared = 0, resets = 0;
    for (int b = 0; b < B; b++) {
        if (terminals[b] != 0.0f) resets++; else compared++;
        for (int h = 0; h < H; h++) {
            int row = B + b, cb = row * 3 * H;
            double hidden = combined[cb + h], gate = combined[cb + H + h];
            double proj = combined[cb + 2 * H + h];
            double z = 1.0 / (1.0 + exp(-gate));
            double candidate = hidden >= 0 ? hidden + 0.5 : 1.0 / (1.0 + exp(-hidden));
            double state = (1.0 - z) * before[b * H + h] + z * candidate;
            double s = 1.0 / (1.0 + exp(-proj));
            double expected = s * state + (1.0 - s) * x[row * H + h];
            double err = fabs(expected - out[row * H + h]);
            // Scoring preceded terminal reset; state_prev has since been
            // cleared. Those rows have no bonus and no surviving prefix to
            // compare. Their zero/reset behavior is checked by reward tests.
            if (terminals[b] != 0.0f) {
                CHECK(before[b * H + h] == 0.0f, "terminal clears state_prev");
                reset_error = fmax(reset_error, err);
            } else {
                maxerr = fmax(maxerr, err);
            }
        }
    }
    CHECK(compared > 0, "literal control test covers nonterminal rows");
    CHECK(maxerr < (USE_BF16 ? 1e-2 : 1e-5), "literal control prefix max error %.3e", maxerr);
    printf("PASS literal control prefix: max abs error %.2e; %d reset rows have obsolete-output error %.2e\n", maxerr, resets, reset_error);
    free(before); free(combined); free(x); free(out); free(terminals);
}

// Online rewards: zero on the boundary row and on rows whose observation
// starts a new episode, finite and capped elsewhere.
static void test_rollout_rewards(PuffeRL* p) {
    T2* t2 = p->t2;
    int T = t2->horizon, A = t2->A;
    float* rew = prec_host(t2->rewards);
    float* term = prec_host(p->rollouts.terminals);
    int nonzero = 0, masked = 0;
    for (int t = 0; t < T; t++) {
        for (int a = 0; a < A; a++) {
            float r = rew[t * A + a];
            CHECK(r == r && r >= 0.0f && r <= t2->reward_scale * t2->reward_cap + 1e-6f,
                "reward range t=%d a=%d r=%f", t, a, r);
            if (t == 0) {
                CHECK(r == 0.0f, "boundary row must stay zero");
            } else if (term[t * A + a] != 0.0f) {
                CHECK(r == 0.0f, "reset row t=%d a=%d has reward %f", t, a, r);
                masked++;
            }
            nonzero += r > 0.0f;
        }
    }
    CHECK(nonzero > 0, "some transitions earn intrinsic reward");
    printf("PASS rollout rewards: %d nonzero, %d reset rows masked\n", nonzero, masked);
    free(rew); free(term);
}

static void test_reservoir(PuffeRL* p) {
    T2* t2 = p->t2;
    int resident = 0, paired = 0, heads = t2->heads;
    long n = (long)t2->C * t2->M * heads;
    int* act = (int*)malloc(n * sizeof(int));
    cudaMemcpy(act, t2->res_act.data, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int j = 0; j < t2->C; j++) {
        int len = t2->res_len[j];
        CHECK(len >= 0 && len <= t2->M, "reservoir length");
        if (len == 0) {
            continue;
        }
        resident++;
        for (int s = 0; s < len; s++) {
            for (int h = 0; h < heads; h++) {
                int a = act[((long)j * t2->M + s) * heads + h];
                CHECK(a >= 0 && a <= t2->act_sizes[h], "recorded action out of range");
            }
        }
        int sib = j ^ 1;
        if (t2->res_len[sib] > 0) {
            CHECK(t2->res_pair[j] == t2->res_pair[sib]
                && t2->res_episode[j] == t2->res_episode[sib], "sibling key");
            paired++;
        }
    }
    CHECK(resident > 0 && t2->episodes_complete > 0, "episodes recorded");
    printf("PASS reservoir: %d resident, %d in complete pairs, %ld episodes\n",
        resident, paired, t2->episodes_complete);
    free(act);
}

// Gathered windows reproduce the reservoir rows they were cut from.
static void test_gather(PuffeRL* p) {
    T2* t2 = p->t2;
    T2Train* tr = &t2->tr;
    int R = t2->R, W = t2->W, L = t2->L, M = t2->M, heads = t2->heads;
    int* slot = (int*)malloc(R * sizeof(int));
    int* t_at = (int*)malloc(R * sizeof(int));
    cudaMemcpy(slot, tr->sample_slot.data, R * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(t_at, tr->sample_t.data, R * sizeof(int), cudaMemcpyDeviceToHost);
    unsigned char* seq = (unsigned char*)malloc((size_t)2 * R * W * L);
    unsigned char* res = (unsigned char*)malloc((size_t)t2->C * M * L);
    unsigned char* target = (unsigned char*)malloc((size_t)R * L);
    int* act_seq = (int*)malloc((size_t)2 * R * W * heads * sizeof(int));
    int* res_act = (int*)malloc((size_t)t2->C * M * heads * sizeof(int));
    float* term = prec_host(tr->term);
    cudaMemcpy(seq, tr->tok_seq.data, (size_t)2 * R * W * L, cudaMemcpyDeviceToHost);
    cudaMemcpy(res, t2->res_tok.data, (size_t)t2->C * M * L, cudaMemcpyDeviceToHost);
    cudaMemcpy(target, tr->target.data, (size_t)R * L, cudaMemcpyDeviceToHost);
    cudaMemcpy(act_seq, tr->act_seq.data, (size_t)2 * R * W * heads * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(res_act, t2->res_act.data, (size_t)t2->C * M * heads * sizeof(int),
        cudaMemcpyDeviceToHost);
    for (int r = 0; r < R; r++) {
        int j = slot[r], t = t_at[r];
        CHECK(t >= 0 && t + 1 < t2->res_len[j], "sampled step has a successor");
        CHECK(memcmp(target + (long)r * L, res + ((long)j * M + t + 1) * L, L) == 0, "target");
        for (int w = 0; w < W; w++) {
            int src = t - W + 1 + w;
            for (int half = 0; half < 2; half++) {
                long row = (long)(half * R + r) * W + w;
                for (int h = 0; h < heads; h++) {
                    int a = act_seq[row * heads + h];
                    if (src < 0) {
                        CHECK(term[row] == 0.0f && a == t2->act_sizes[h], "padding");
                    } else {
                        int expect = (half == 1 && w == W - 1)
                            ? t2->act_sizes[h] : res_act[((long)j * M + src) * heads + h];
                        CHECK(a == expect, "window action");
                    }
                }
                if (src >= 0) {
                    CHECK(memcmp(seq + row * L, res + ((long)j * M + src) * L, L) == 0,
                        "window tokens");
                    CHECK(term[row] == (src == 0 ? 1.0f : 0.0f), "reset flag");
                }
            }
        }
    }
    printf("PASS gather: %d rows, window %d\n", R, W);
    free(slot); free(t_at); free(seq); free(res); free(target); free(act_seq); free(res_act); free(term);
}

// Re-encoding under unchanged weights reproduces the streaming lane states
// and the recorded reservoir states exactly (float build) or closely (bf16).
static void test_reencode(PuffeRL* p) {
    T2* t2 = p->t2;
    float* before = prec_host(t2->state);
    float* prev_before = prec_host(t2->state_prev);
    float* res_before = prec_host(t2->res_state);
    t2_reencode(t2, p->train_stream);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError() == cudaSuccess, "re-encode kernels");
    float* after = prec_host(t2->state);
    float* prev_after = prec_host(t2->state_prev);
    float* res_after = prec_host(t2->res_state);
    double max_err = 0.0, max_prev = 0.0, max_res = 0.0;
    int lanes = 0, slots = 0;
    int H = t2->H;
    for (int a = 0; a < t2->A; a++) {
        if (t2->host_len[a] <= 0 || t2->host_len[a] >= t2->M) {
            continue;
        }
        lanes++;
        for (int l = 0; l < t2->Lg; l++) {
            for (int h = 0; h < H; h++) {
                long i = ((long)l * t2->A + a) * H + h;
                double e = fabs(before[i] - after[i]);
                max_err = e > max_err ? e : max_err;
                double ep = fabs(prev_before[i] - prev_after[i]);
                max_prev = ep > max_prev ? ep : max_prev;
            }
        }
    }
    for (int j = 0; j < t2->C; j++) {
        int len = t2->res_len[j];
        if (len <= 0) {
            continue;
        }
        slots++;
        long lo = (long)j * t2->M * t2->Lg * H, hi = ((long)j * t2->M + len) * t2->Lg * H;
        for (long i = lo; i < hi; i++) {
            double e = fabs(res_before[i] - res_after[i]);
            max_res = e > max_res ? e : max_res;
        }
    }
    double tol = USE_BF16 ? 3e-2 : 1e-6;
    CHECK(lanes > 0 && slots > 0, "re-encode coverage");
    CHECK(max_err <= tol && max_prev <= tol && max_res <= tol,
        "re-encode mismatch: state %.3e prev %.3e reservoir %.3e", max_err, max_prev, max_res);
    printf("PASS re-encode: %d lanes, %d slots, max errors %.2e / %.2e / %.2e\n",
        lanes, slots, max_err, max_prev, max_res);
    free(before); free(prev_before); free(res_before); free(after); free(prev_after); free(res_after);
}

// Central finite differences on the fp32 master weights against the analytic
// gradient of the mean per-token NLL. Float build only.
static void test_t2_gradients(PuffeRL* p) {
    T2* t2 = p->t2;
    if (USE_BF16) {
        printf("SKIP t2 gradients: bf16 build\n");
        return;
    }
    cudaStream_t stream = p->train_stream;
    int L = t2->L;
    float base = t2_forward_backward(t2, stream) / L;
    float* grad = prec_host(t2->grad);
    struct { const char* name; Prec* w; } params[] = {
        {"tok_embed", &t2->w.tok_embed}, {"act_embed", &t2->w.act_embed},
        {"w_in", &t2->w.w_in}, {"gru0", &t2->w.gru[0]}, {"role", &t2->w.role},
        {"w_h", &t2->w.w_h}, {"pos_embed", &t2->w.pos_embed}, {"w_out", &t2->w.w_out},
    };
    int nparams = (int)(sizeof(params) / sizeof(params[0]));
    int R = t2->R, W = t2->W;
    unsigned char* seq = (unsigned char*)malloc((size_t)2 * R * W * L);
    cudaMemcpy(seq, t2->tr.tok_seq.data, (size_t)2 * R * W * L, cudaMemcpyDeviceToHost);
    int* act_seq = (int*)malloc((size_t)2 * R * W * t2->heads * sizeof(int));
    cudaMemcpy(act_seq, t2->tr.act_seq.data, (size_t)2 * R * W * t2->heads * sizeof(int),
        cudaMemcpyDeviceToHost);
    double worst = 0.0;
    int compared = 0;
    for (int i = 0; i < nparams; i++) {
        long n = numel(params[i].w->shape);
        long offset = (long)(params[i].w->data - t2->param.data);
        for (int k = 0; k < 4; k++) {
            long idx;
            if (i == 0) {
                long row = lcg() % ((long)2 * R * W);
                int pp = lcg() % L;
                idx = ((long)pp * T2_VOCAB + seq[row * L + pp]) * t2->E + (lcg() % t2->E);
            } else if (i == 1) {
                long row = lcg() % ((long)2 * R * W);
                int h = lcg() % t2->heads;
                idx = (long)(t2->head_off[h] + act_seq[row * t2->heads + h]) * t2->EA
                    + (lcg() % t2->EA);
            } else {
                idx = lcg() % n;
            }
            long flat = offset + idx;
            float saved;
            cudaMemcpy(&saved, t2->master.data + flat, sizeof(float), cudaMemcpyDeviceToHost);
            float eps = 1e-2f;
            float plus = saved + eps, minus = saved - eps;
            cudaMemcpy(t2->master.data + flat, &plus, sizeof(float), cudaMemcpyHostToDevice);
            float lp = t2_forward_backward(t2, stream) / L;
            cudaMemcpy(t2->master.data + flat, &minus, sizeof(float), cudaMemcpyHostToDevice);
            float lm = t2_forward_backward(t2, stream) / L;
            cudaMemcpy(t2->master.data + flat, &saved, sizeof(float), cudaMemcpyHostToDevice);
            double fd = (double)(lp - lm) / (2.0 * eps);
            double an = grad[flat];
            double err = fabs(fd - an) / (1e-3 + fabs(fd) + fabs(an));
            worst = err > worst ? err : worst;
            compared++;
            CHECK(err < 0.1, "%s[%ld]: finite difference %.4e vs analytic %.4e (rel %.3f)",
                params[i].name, idx, fd, an, err);
        }
    }
    t2_forward_backward(t2, stream);
    printf("PASS t2 gradients: %d entries, base loss %.4f nats/token, worst rel err %.3f\n",
        compared, base, worst);
    free(grad); free(seq); free(act_seq);
}

static void run_t2_tests(Ini* ini) {
    PuffeRL* p = make_trainer(ini, "t2");
    CHECK(p->ir && p->t2, "T2 created");
    for (int epoch = 0; epoch < 60; epoch++) {
        rollouts(p);
        cudaDeviceSynchronize();
    }
    CHECK(cudaGetLastError() == cudaSuccess, "rollout kernels");
    test_rollout_rewards(p);
    test_reservoir(p);
    test_head_reference(p);
    test_gate_consistency(p);
    test_literal_control_prefix(p);
    test_reencode(p);
    CHECK(t2_sample(p->t2, p->train_stream) >= 0, "sampled a batch");
    t2_forward_backward(p->t2, p->train_stream);
    test_gather(p);
    test_t2_gradients(p);
    test_updated_history_records(p);
    t2_train(p);
    CHECK(cudaGetLastError() == cudaSuccess, "train kernels");
    t2_save(p, "build/test_t2_weights.bin");
    float* before = to_host(p->t2->master.data, numel(p->t2->master.shape), sizeof(float));
    cudaMemset(p->t2->master.data, 0, numel(p->t2->master.shape) * sizeof(float));
    t2_load(p, "build/test_t2_weights.bin");
    float* after = to_host(p->t2->master.data, numel(p->t2->master.shape), sizeof(float));
    CHECK(memcmp(before, after, numel(p->t2->master.shape) * sizeof(float)) == 0, "checkpoint");
    printf("PASS t2 checkpoint roundtrip\n");
    test_record_publication(p);
    free(before); free(after);
    close_pufferl(p);
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

// Gauss-Jordan inverse of a small symmetric positive definite matrix.
static void host_invert(double* a, int d) {
    double* inv = (double*)malloc(d * d * sizeof(double));
    for (int i = 0; i < d * d; i++) {
        inv[i] = (i / d == i % d) ? 1.0 : 0.0;
    }
    for (int c = 0; c < d; c++) {
        double piv = a[c * d + c];
        for (int j = 0; j < d; j++) {
            a[c * d + j] /= piv;
            inv[c * d + j] /= piv;
        }
        for (int r = 0; r < d; r++) {
            if (r == c) {
                continue;
            }
            double f = a[r * d + c];
            for (int j = 0; j < d; j++) {
                a[r * d + j] -= f * a[c * d + j];
                inv[r * d + j] -= f * inv[c * d + j];
            }
        }
    }
    memcpy(a, inv, d * d * sizeof(double));
    free(inv);
}

// One lane, a sequence of random features with a terminal in the middle:
// bonuses equal phi^T (lambda I + sum phi phi^T)^-1 phi from an explicit
// inverse, the first step of each episode reads zero, and the update after a
// terminal starts from a fresh ridge matrix.
static void test_e3b_kernel(void) {
    int d = 24, steps = 12;
    float ridge = 0.1f;
    float* cinv;
    int* ep_step;
    precision_t* phi;
    float* bonus;
    float* done;
    cudaMalloc((void**)&cinv, d * d * sizeof(float));
    cudaMalloc((void**)&ep_step, sizeof(int));
    cudaMalloc((void**)&phi, d * sizeof(precision_t));
    cudaMalloc((void**)&bonus, sizeof(float));
    cudaMalloc((void**)&done, sizeof(float));
    cudaMemset(ep_step, 0, sizeof(int));
    ir_reset_cinv_kernel<<<grid_size(d * d), BLOCK_SIZE>>>(cinv, 1, d, ridge);
    double* gram = (double*)malloc(d * d * sizeof(double));
    double* work = (double*)malloc(d * d * sizeof(double));
    for (int i = 0; i < d * d; i++) {
        gram[i] = (i / d == i % d) ? ridge : 0.0;
    }
    int episode_step = 0;
    double max_err = 0.0;
    for (int t = 0; t < steps; t++) {
        float f[64];
        for (int i = 0; i < d; i++) {
            f[i] = to_float(from_float(urand(-1.0f, 1.0f)));
        }
        prec_upload(phi, f, d);
        float dn = t == 5 ? 1.0f : 0.0f;
        cudaMemcpy(done, &dn, sizeof(float), cudaMemcpyHostToDevice);
        size_t smem = (2 * d + 256) * sizeof(float);
        ir_e3b_kernel<<<1, 256, smem>>>(bonus, cinv, ep_step, phi, done, 0, d, ridge);
        cudaDeviceSynchronize();
        float b;
        cudaMemcpy(&b, bonus, sizeof(float), cudaMemcpyDeviceToHost);
        memcpy(work, gram, d * d * sizeof(double));
        host_invert(work, d);
        double expect = 0.0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                expect += (double)f[i] * work[i * d + j] * f[j];
            }
        }
        if (episode_step == 0) {
            expect = 0.0;
        }
        double err = fabs(expect - b) / (1.0 + fabs(expect));
        max_err = err > max_err ? err : max_err;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                gram[i * d + j] += (double)f[i] * f[j];
            }
        }
        episode_step++;
        if (dn != 0.0f) {
            for (int i = 0; i < d * d; i++) {
                gram[i] = (i / d == i % d) ? ridge : 0.0;
            }
            episode_step = 0;
        }
    }
    int step_dev;
    cudaMemcpy(&step_dev, ep_step, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(step_dev == episode_step, "episode step counter %d vs %d", step_dev, episode_step);
    CHECK(max_err < 1e-3, "E3B bonus vs explicit inverse: %.3e", max_err);
    printf("PASS e3b kernel: %d steps with a reset, max relative error %.2e\n", steps, max_err);
    free(gram); free(work);
    cudaFree(cinv); cudaFree(ep_step); cudaFree(phi); cudaFree(bonus); cudaFree(done);
}

// The sequential normalizer matches the released chunked Welford formula
// (count advances by T per eight-lane chunk) across two consecutive calls.
static void test_normalizer(void) {
    int T = 5, B = 24, lanes = 8;
    float* raw = (float*)malloc(2 * T * B * sizeof(float));
    for (int i = 0; i < 2 * T * B; i++) {
        raw[i] = urand(0.0f, 3.0f);
    }
    float *d_raw, *d_out;
    double* d_stats;
    cudaMalloc((void**)&d_raw, T * B * sizeof(float));
    cudaMalloc((void**)&d_out, T * B * sizeof(float));
    cudaMalloc((void**)&d_stats, 3 * sizeof(double));
    cudaMemset(d_stats, 0, 3 * sizeof(double));
    double sum = 0.0, m2 = 0.0, count = 0.0, max_err = 0.0;
    for (int call = 0; call < 2; call++) {
        float* x = raw + call * T * B;
        cudaMemcpy(d_raw, x, T * B * sizeof(float), cudaMemcpyHostToDevice);
        ir_normalize_kernel<<<1, 256>>>(d_out, d_stats, d_raw, T, B, lanes, IR_NORM_TORCHBEAST);
        cudaDeviceSynchronize();
        float* out = (float*)malloc(T * B * sizeof(float));
        cudaMemcpy(out, d_out, T * B * sizeof(float), cudaMemcpyDeviceToHost);
        for (int c = 0; c < B / lanes; c++) {
            double bs = 0.0;
            for (int t = 0; t < T; t++) {
                for (int l = 0; l < lanes; l++) {
                    bs += x[t * B + c * lanes + l];
                }
            }
            double bc = T, bm = bs / bc;
            double bm2 = 0.0;
            for (int t = 0; t < T; t++) {
                for (int l = 0; l < lanes; l++) {
                    double dv = x[t * B + c * lanes + l] - bm;
                    bm2 += dv * dv;
                }
            }
            double old_mean = count > 0 ? sum / count : 0.0;
            double total = count + bc;
            double new_m2 = m2 + bm2 + count * bc / total * (bm - old_mean) * (bm - old_mean);
            double inv = 1.0 / sqrt(new_m2 / total + 1e-8);
            for (int t = 0; t < T; t++) {
                for (int l = 0; l < lanes; l++) {
                    int at = t * B + c * lanes + l;
                    double err = fabs(out[at] - x[at] * inv) / (1e-6 + fabs(x[at] * inv));
                    max_err = err > max_err ? err : max_err;
                }
            }
            sum += bs;
            m2 = new_m2;
            count = total;
        }
        free(out);
    }
    double stats[3];
    cudaMemcpy(stats, d_stats, sizeof(stats), cudaMemcpyDeviceToHost);
    CHECK(fabs(stats[2] - count) < 1e-9 && fabs(stats[0] - sum) < 1e-3 * (1 + fabs(sum)),
        "normalizer state");
    CHECK(max_err < 1e-4, "torchbeast normalizer mismatch %.3e", max_err);
    printf("PASS normalizer: %d chunks x 2 calls, max relative error %.2e\n", B / lanes, max_err);
    free(raw);
    cudaFree(d_raw); cudaFree(d_out); cudaFree(d_stats);
}

// Finite differences on the feature learner and heads for the IDM loss.
static void test_baseline_gradients(PuffeRL* p, const char* name) {
    Baseline* bl = p->ir->bl;
    if (USE_BF16 || !ir_has_idm(bl->method)) {
        printf("SKIP %s gradients: %s\n", name, USE_BF16 ? "bf16 build" : "no feature learner");
        return;
    }
    cudaStream_t stream = p->train_stream;
    RolloutBuf src = rollout_time_view(&p->rollouts, 0, bl->horizon);
    baseline_sample_rows(bl, stream);
    float base = baseline_forward_backward(bl, &src, stream);
    float* grad = prec_host(bl->idm.grad);
    // The trunk probe covers the default linear encoder; custom env encoders
    // expose their own parameter layout and are probed through the head only.
    long trunk_n = bl->feat.enc.forward == encoder_forward ? (long)bl->hidden * bl->obs_size : 0;
    Prec trunk = {.data = bl->idm.param.data, .shape = {trunk_n}};
    struct { const char* name; Prec* w; } params[] = {
        {"trunk", &trunk}, {"w_out", &bl->feat.w_out}, {"ln_gamma", &bl->feat.ln_gamma},
        {"inverse_w1", &bl->inverse.w1}, {"inverse_w2", &bl->inverse.w2},
        {"forward_w1", &bl->forward.w1}, {"forward_w2", &bl->forward.w2},
    };
    double worst = 0.0;
    int compared = 0, failures = 0;
    int nparams = bl->method == IR_ICM ? 7 : 5;
    for (int i = 0; i < nparams; i++) {
        if ((i == 2 && !bl->feat.layernorm) || (i == 0 && trunk_n == 0)) {
            continue;
        }
        long n = numel(params[i].w->shape);
        long offset = (long)(params[i].w->data - bl->idm.param.data);
        for (int k = 0; k < 3; k++) {
            long idx = lcg() % n;
            long flat = offset + idx;
            float saved;
            cudaMemcpy(&saved, bl->idm.master.data + flat, sizeof(float), cudaMemcpyDeviceToHost);
            double an = grad[flat];
            double err = 1.0, fd = 0.0;
            // ReLU kinks within the probe step spoil a central difference; a
            // narrower step is retried before counting an entry as wrong.
            float steps_eps[2] = {1e-2f, 2.5e-3f};
            for (int e = 0; e < 2 && err >= 0.1; e++) {
                float eps = steps_eps[e];
                float plus = saved + eps, minus = saved - eps;
                cudaMemcpy(bl->idm.master.data + flat, &plus, sizeof(float), cudaMemcpyHostToDevice);
                float lp = baseline_forward_backward(bl, &src, stream);
                cudaMemcpy(bl->idm.master.data + flat, &minus, sizeof(float), cudaMemcpyHostToDevice);
                float lm = baseline_forward_backward(bl, &src, stream);
                cudaMemcpy(bl->idm.master.data + flat, &saved, sizeof(float), cudaMemcpyHostToDevice);
                fd = (double)(lp - lm) / (2.0 * eps);
                err = fabs(fd - an) / (1e-2 + fabs(fd) + fabs(an));
            }
            worst = err > worst ? err : worst;
            compared++;
            if (err >= 0.1) {
                failures++;
                printf("  %s %s[%ld]: finite difference %.4e vs analytic %.4e (rel %.3f)\n",
                    name, params[i].name, idx, fd, an, err);
            }
        }
    }
    baseline_forward_backward(bl, &src, stream);
    CHECK(failures <= 1, "%s gradients: %d of %d entries disagree", name, failures, compared);
    printf("PASS %s gradients: %d entries, base loss %.4f, worst rel err %.3f, %d kink retries failed\n",
        name, compared, base, worst, failures);
    free(grad);
}

// RND gradient check on the predictor group (float build).
static void test_rnd_gradients(PuffeRL* p, const char* name) {
    Baseline* bl = p->ir->bl;
    if (USE_BF16 || !ir_has_rnd(bl->method)) {
        return;
    }
    cudaStream_t stream = p->train_stream;
    RolloutBuf src = rollout_time_view(&p->rollouts, 0, bl->horizon);
    baseline_sample_rows(bl, stream);
    float base = baseline_forward_backward(bl, &src, stream);
    float* grad = prec_host(bl->rnd.grad);
    long n = numel(bl->rnd.param.shape);
    double worst = 0.0;
    for (int k = 0; k < 6; k++) {
        long flat = lcg() % n;
        float saved;
        cudaMemcpy(&saved, bl->rnd.master.data + flat, sizeof(float), cudaMemcpyDeviceToHost);
        float eps = 1e-2f;
        float plus = saved + eps, minus = saved - eps;
        cudaMemcpy(bl->rnd.master.data + flat, &plus, sizeof(float), cudaMemcpyHostToDevice);
        float lp = baseline_forward_backward(bl, &src, stream);
        cudaMemcpy(bl->rnd.master.data + flat, &minus, sizeof(float), cudaMemcpyHostToDevice);
        float lm = baseline_forward_backward(bl, &src, stream);
        cudaMemcpy(bl->rnd.master.data + flat, &saved, sizeof(float), cudaMemcpyHostToDevice);
        double fd = (double)(lp - lm) / (2.0 * eps);
        double an = grad[flat];
        double err = fabs(fd - an) / (1e-2 + fabs(fd) + fabs(an));
        worst = err > worst ? err : worst;
        CHECK(err < 0.1, "%s rnd[%ld]: finite difference %.4e vs analytic %.4e", name, flat, fd, an);
    }
    baseline_forward_backward(bl, &src, stream);
    printf("PASS %s rnd gradients: base loss %.4f, worst rel err %.3f\n", name, base, worst);
    free(grad);
}

// Per-method smoke: rollouts, PPO epoch with the intrinsic reward applied,
// intrinsic training; rewards finite, bonuses present.
static void run_baseline_tests(Ini* ini, const char* name) {
    PuffeRL* p = make_trainer(ini, name);
    Baseline* bl = p->ir->bl;
    CHECK(bl != NULL && p->t2 == NULL, "%s created", name);
    for (int epoch = 0; epoch < 6; epoch++) {
        rollouts(p);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError() == cudaSuccess, "%s rollout kernels", name);
        train_impl(p, NULL);
        ir_train(p);
    }
    int T = bl->horizon, A = bl->A;
    float* rew = prec_host(p->train_rollouts.rewards);
    float* raw = to_host(bl->combined.data, (long)T * A, sizeof(float));
    int nonzero = 0, raw_nonzero = 0;
    for (int i = 0; i < T * A; i++) {
        CHECK(rew[i] == rew[i] && rew[i] >= -1.0f && rew[i] <= 1.0f, "%s reward range", name);
        CHECK(raw[i] == raw[i] && raw[i] >= 0.0f, "%s raw bonus", name);
        nonzero += rew[i] != 0.0f;
        raw_nonzero += raw[i] > 0.0f;
    }
    CHECK(raw_nonzero > 0 && nonzero > 0, "%s produces rewards", name);
    if (bl->cinv.data) {
        float* bonus = to_host(bl->bonus_e3b.data, (long)T * A, sizeof(float));
        for (int a = 0; a < A; a++) {
            CHECK(bonus[a] == 0.0f, "%s boundary row", name);
        }
        free(bonus);
    }
    test_baseline_gradients(p, name);
    test_rnd_gradients(p, name);
    ir_save(p, "build/test_intrinsic.bin");
    printf("PASS %s smoke: %d/%d nonzero rewards, %d raw bonuses\n", name, nonzero, T * A, raw_nonzero);
    free(rew); free(raw);
    close_pufferl(p);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    rng_state = 12345;
#ifdef PUFFER_CRAFTAX_CLASSIC
    test_codec();
    test_pairing();
#endif
    test_e3b_kernel();
    test_normalizer();
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, argc - 1, argv + 1);
#ifdef PUFFER_NETHACK
    test_nethack_pairing_and_codec(&ini);
#endif
    run_t2_tests(&ini);
    test_t2_history_limits(&ini);
    const char* methods[] = {"e3b", "icm", "rnd", "e3b_rnd"};
    for (int i = 0; i < 4; i++) {
        run_baseline_tests(&ini, methods[i]);
    }
    printf("ALL %d checks passed\n", checks);
    return 0;
}
