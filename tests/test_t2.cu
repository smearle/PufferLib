// T2 tests: env codec/pairing contract, kernel consistency, gradient checks,
// online scoring invariants, reservoir bookkeeping. Runs the real trainer
// substrate on a small craftax_classic configuration.
//   ./build.sh craftax_classic build/test_t2 --t2-test --float && ./build/test_t2
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

static void env_alloc(Env* env, unsigned lane, Dict* kwargs) {
    memset(env, 0, sizeof(Env));
    env->rng = lane;
    env->agents[0].observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    env->agents[0].actions = (float*)calloc(NUM_ATNS, sizeof(float));
    env->agents[0].rewards = (float*)calloc(1, sizeof(float));
    env->agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_init(env, kwargs);
}

static unsigned rng_state = 12345;
static unsigned lcg(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
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
    // Same actions, independent RNG streams: trajectories must diverge.
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
    // Determinism: a fresh lane 0 replays lane 0 exactly.
    Env d;
    env_alloc(&d, 0, &kwargs);
    puf_reset(&d);
    CHECK(memcmp(d.map_packed, first_world, sizeof(first_world)) == 0, "replay world");
    printf("PASS pairing: diverged on %d steps before the first terminal\n", differ);
}
#endif

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

static double host_gelu(double x) {
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
}

// Recompute the scoring head on the host from the device z rows.
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

// The three-group gate kernel must agree with PufferLib's single-group
// mingru_gate for each group's input state.
static void test_gate_consistency(PuffeRL* p) {
    T2* t2 = p->t2;
    int B = t2->A / t2->num_buffers, H = t2->H;
    T2Rollout* r = &t2->roll[0];
    // Inputs: random combined / x / states.
    long n3 = (long)3 * B * H;
    float* h_comb = (float*)malloc(3 * n3 * sizeof(float));
    float* h_x = (float*)malloc(n3 * sizeof(float));
    float* h_state = (float*)malloc(B * H * sizeof(float));
    float* h_prev = (float*)malloc(B * H * sizeof(float));
    for (long i = 0; i < 3 * n3; i++) h_comb[i] = ((int)(lcg() % 2001) - 1000) / 500.0f;
    for (long i = 0; i < n3; i++) h_x[i] = ((int)(lcg() % 2001) - 1000) / 500.0f;
    for (long i = 0; i < B * H; i++) h_state[i] = ((int)(lcg() % 2001) - 1000) / 1000.0f;
    for (long i = 0; i < B * H; i++) h_prev[i] = ((int)(lcg() % 2001) - 1000) / 1000.0f;
    precision_t *comb, *x, *state, *prev, *out, *ref_out, *ref_next;
    cudaMalloc((void**)&comb, 3 * n3 * sizeof(precision_t));
    cudaMalloc((void**)&x, n3 * sizeof(precision_t));
    cudaMalloc((void**)&state, B * H * sizeof(precision_t));
    cudaMalloc((void**)&prev, B * H * sizeof(precision_t));
    cudaMalloc((void**)&out, n3 * sizeof(precision_t));
    cudaMalloc((void**)&ref_out, B * H * sizeof(precision_t));
    cudaMalloc((void**)&ref_next, B * H * sizeof(precision_t));
    (void)r;
    precision_t* tmp = (precision_t*)malloc(3 * n3 * sizeof(precision_t));
    for (long i = 0; i < 3 * n3; i++) tmp[i] = from_float(h_comb[i]);
    cudaMemcpy(comb, tmp, 3 * n3 * sizeof(precision_t), cudaMemcpyHostToDevice);
    for (long i = 0; i < n3; i++) tmp[i] = from_float(h_x[i]);
    cudaMemcpy(x, tmp, n3 * sizeof(precision_t), cudaMemcpyHostToDevice);
    for (long i = 0; i < B * H; i++) tmp[i] = from_float(h_state[i]);
    cudaMemcpy(state, tmp, B * H * sizeof(precision_t), cudaMemcpyHostToDevice);
    for (long i = 0; i < B * H; i++) tmp[i] = from_float(h_prev[i]);
    cudaMemcpy(prev, tmp, B * H * sizeof(precision_t), cudaMemcpyHostToDevice);
    t2_gate3_kernel<<<grid_size(B * H), BLOCK_SIZE>>>(out, state, prev, comb, x, B, H);
    cudaDeviceSynchronize();
    float* g_out = to_host(out, n3, sizeof(precision_t));
    float* g_state = to_host(state, B * H, sizeof(precision_t));
    float* g_prev = to_host(prev, B * H, sizeof(precision_t));
    // Group 0 from the old state, group 1 from the old prev, group 2 from the new state.
    float* inputs[3] = {h_state, h_prev, g_state};
    double max_err = 0.0;
    for (int g = 0; g < 3; g++) {
        for (long i = 0; i < B * H; i++) tmp[i] = from_float(inputs[g][i]);
        cudaMemcpy(state, tmp, B * H * sizeof(precision_t), cudaMemcpyHostToDevice);
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
        CHECK(g_prev[i] == to_float(from_float(h_state[i])),
            "state_prev keeps the pre-step state");
    }
    CHECK(max_err < 1e-5, "gate3 vs mingru_gate mismatch %.3e", max_err);
    printf("PASS gate consistency: max abs error %.2e\n", max_err);
    free(h_comb); free(h_x); free(h_state); free(h_prev); free(tmp);
    free(g_out); free(g_state); free(g_prev);
    cudaFree(comb); cudaFree(x); cudaFree(state); cudaFree(prev);
    cudaFree(out); cudaFree(ref_out); cudaFree(ref_next);
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
    int resident = 0, paired = 0;
    int* act = (int*)malloc((size_t)t2->C * t2->M * sizeof(int));
    cudaMemcpy(act, t2->res_act.data, (size_t)t2->C * t2->M * sizeof(int), cudaMemcpyDeviceToHost);
    for (int j = 0; j < t2->C; j++) {
        int len = t2->res_len[j];
        CHECK(len >= 0 && len <= t2->M, "reservoir length");
        if (len == 0) {
            continue;
        }
        resident++;
        for (int s = 0; s < len; s++) {
            CHECK(act[(long)j * t2->M + s] >= 0 && act[(long)j * t2->M + s] < t2->num_actions,
                "recorded action out of range");
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
    int R = t2->R, W = t2->W, L = t2->L, M = t2->M;
    int* slot = (int*)malloc(R * sizeof(int));
    int* t_at = (int*)malloc(R * sizeof(int));
    cudaMemcpy(slot, tr->sample_slot.data, R * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(t_at, tr->sample_t.data, R * sizeof(int), cudaMemcpyDeviceToHost);
    unsigned char* seq = (unsigned char*)malloc((size_t)2 * R * W * L);
    unsigned char* res = (unsigned char*)malloc((size_t)t2->C * M * L);
    unsigned char* target = (unsigned char*)malloc((size_t)R * L);
    int* act_seq = (int*)malloc((size_t)2 * R * W * sizeof(int));
    int* res_act = (int*)malloc((size_t)t2->C * M * sizeof(int));
    float* term = prec_host(tr->term);
    cudaMemcpy(seq, tr->tok_seq.data, (size_t)2 * R * W * L, cudaMemcpyDeviceToHost);
    cudaMemcpy(res, t2->res_tok.data, (size_t)t2->C * M * L, cudaMemcpyDeviceToHost);
    cudaMemcpy(target, tr->target.data, (size_t)R * L, cudaMemcpyDeviceToHost);
    cudaMemcpy(act_seq, tr->act_seq.data, (size_t)2 * R * W * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_act, t2->res_act.data, (size_t)t2->C * M * sizeof(int), cudaMemcpyDeviceToHost);
    for (int r = 0; r < R; r++) {
        int j = slot[r], t = t_at[r];
        CHECK(t >= 0 && t + 1 < t2->res_len[j], "sampled step has a successor");
        CHECK(memcmp(target + (long)r * L, res + ((long)j * M + t + 1) * L, L) == 0, "target");
        for (int w = 0; w < W; w++) {
            int src = t - W + 1 + w;
            for (int half = 0; half < 2; half++) {
                long row = (long)(half * R + r) * W + w;
                if (src < 0) {
                    CHECK(term[row] == 0.0f && act_seq[row] == t2->num_actions,
                        "padding");
                } else {
                    CHECK(memcmp(seq + row * L, res + ((long)j * M + src) * L, L) == 0,
                        "window tokens");
                    CHECK(term[row] == (src == 0 ? 1.0f : 0.0f), "reset flag");
                    int expect = (half == 1 && w == W - 1)
                ? t2->num_actions : res_act[(long)j * M + src];
                    CHECK(act_seq[row] == expect, "window action");
                }
            }
        }
    }
    printf("PASS gather: %d rows, window %d\n", R, W);
    free(slot); free(t_at); free(seq); free(res); free(target); free(act_seq); free(res_act); free(term);
}

// Central finite differences on the fp32 master weights against the analytic
// gradient of the mean per-token NLL. Float build only.
static void test_gradients(PuffeRL* p) {
    T2* t2 = p->t2;
    if (USE_BF16) {
        printf("SKIP gradients: bf16 build\n");
        return;
    }
    cudaStream_t stream = p->train_stream;
    int L = t2->L;
    float base = t2_forward_backward(t2, stream) / L;
    float* grad = prec_host(t2->grad);
    long offset = 0;
    struct { const char* name; Prec* w; } params[] = {
        {"tok_embed", &t2->w.tok_embed}, {"act_embed", &t2->w.act_embed},
        {"w_in", &t2->w.w_in}, {"gru0", &t2->w.gru[0]}, {"role", &t2->w.role},
        {"w_h", &t2->w.w_h}, {"pos_embed", &t2->w.pos_embed}, {"w_out", &t2->w.w_out},
    };
    // Registration order matches params_alloc; recover each tensor's flat offset.
    int nparams = (int)(sizeof(params) / sizeof(params[0]));
    long offsets[8];
    for (int i = 0; i < nparams; i++) {
        offsets[i] = (long)(params[i].w->data - t2->param.data);
    }
    (void)offset;
    // Tokens present in this batch so tok_embed rows are actually used.
    int R = t2->R, W = t2->W;
    unsigned char* seq = (unsigned char*)malloc((size_t)2 * R * W * L);
    cudaMemcpy(seq, t2->tr.tok_seq.data, (size_t)2 * R * W * L, cudaMemcpyDeviceToHost);
    double worst = 0.0;
    int compared = 0;
    for (int i = 0; i < nparams; i++) {
        long n = numel(params[i].w->shape);
        for (int k = 0; k < 4; k++) {
            long idx;
            if (i == 0) {
                long row = lcg() % ((long)2 * R * W);
                int pp = lcg() % L;
                idx = ((long)pp * T2_VOCAB + seq[row * L + pp]) * t2->E + (lcg() % t2->E);
            } else {
                idx = lcg() % n;
            }
            long flat = offsets[i] + idx;
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
    // Restore the gradient buffer state for later calls.
    t2_forward_backward(t2, stream);
    printf("PASS gradients: %d entries, base loss %.4f nats/token, worst rel err %.3f\n",
        compared, base, worst);
    free(grad); free(seq);
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    rng_state = 12345;
#ifdef PUFFER_CRAFTAX_CLASSIC
    test_codec();
    test_pairing();
#endif
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, argc - 1, argv + 1);
    puf_ini_put(&ini, "t2.enabled", "1");
    puf_ini_put(&ini, "vec.total_agents", "32");
    puf_ini_put(&ini, "vec.num_buffers", "1");
    puf_ini_put(&ini, "vec.num_threads", "4");
    puf_ini_put(&ini, "train.horizon", "16");
    puf_ini_put(&ini, "train.minibatch_size", "512");
    puf_ini_put(&ini, "base.async", "0");
    puf_ini_put(&ini, "base.cudagraphs", "-1");
    puf_ini_put(&ini, "t2.embed_dim", "32");
    puf_ini_put(&ini, "t2.hidden_size", "32");
    puf_ini_put(&ini, "t2.head_dim", "32");
    puf_ini_put(&ini, "t2.max_episode", "64");
    puf_ini_put(&ini, "t2.reservoir", "16");
    puf_ini_put(&ini, "t2.wm_batch", "8");
    puf_ini_put(&ini, "t2.bptt_window", "8");
    puf_ini_put(&ini, "t2.wm_steps", "1");
    TrainContext ctx = {.world_size = 1, .artifact_owner = 0};
    PuffeRL* p = create_pufferl(&ini, &ctx);
    CHECK(p->t2 != NULL, "T2 created");
    // Collect until the reservoir holds complete episodes.
    for (int epoch = 0; epoch < 60; epoch++) {
        rollouts(p);
        cudaDeviceSynchronize();
    }
    CHECK(cudaGetLastError() == cudaSuccess, "rollout kernels");
    test_rollout_rewards(p);
    test_reservoir(p);
    test_head_reference(p);
    test_gate_consistency(p);
    CHECK(t2_sample(p->t2, p->train_stream) >= 0, "sampled a batch");
    t2_forward_backward(p->t2, p->train_stream);  // gathers the sampled windows
    test_gather(p);
    test_gradients(p);
    t2_train(p);
    CHECK(cudaGetLastError() == cudaSuccess, "train kernels");
    // Checkpoint roundtrip.
    t2_save(p, "build/test_t2_weights.bin");
    float* before = to_host(p->t2->master.data, numel(p->t2->master.shape), sizeof(float));
    cudaMemset(p->t2->master.data, 0, numel(p->t2->master.shape) * sizeof(float));
    t2_load(p, "build/test_t2_weights.bin");
    float* after = to_host(p->t2->master.data, numel(p->t2->master.shape), sizeof(float));
    CHECK(memcmp(before, after, numel(p->t2->master.shape) * sizeof(float)) == 0, "checkpoint");
    printf("PASS checkpoint roundtrip\n");
    printf("ALL %d checks passed\n", checks);
    return 0;
}
