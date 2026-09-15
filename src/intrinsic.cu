// Intrinsic reward dispatcher: [intrinsic] method selects T2 or one of the
// paper-faithful baselines (E3B, ICM, RND, E3B x RND). Included by pufferl.cu
// after PuffeRL when built with --t2 (-DPUFFER_T2). Every hook is a no-op
// when the method is none.

enum { IR_NONE, IR_T2, IR_E3B, IR_ICM, IR_RND, IR_E3B_RND };

static int* ir_act_sizes_dev = NULL;
static int* ir_act_offsets_dev = NULL;

struct T2;
struct Baseline;
struct Intrinsic {
    int method;
    T2* t2;
    Baseline* bl;
};

#include "t2.cu"
#include "baselines.cu"

int ir_method(Ini* ini) {
    const char* m = puf_ini_get_str(ini, "intrinsic", "method");
    if (strcmp(m, "none") == 0) {
        // Back-compat: [t2] enabled = 1 selects T2.
        return puf_ini_get(ini, "t2", "enabled") != 0 ? IR_T2 : IR_NONE;
    }
    const char* names[] = {"none", "t2", "e3b", "icm", "rnd", "e3b_rnd"};
    for (int i = 0; i < 6; i++) {
        if (strcmp(m, names[i]) == 0) {
            return i;
        }
    }
    fprintf(stderr, "[intrinsic] method must be one of none, t2, e3b, icm, rnd, e3b_rnd\n");
    exit(1);
}

// Env kwargs that depend on the method: T2 pairs lanes per world seed.
void ir_env_kwargs(Ini* ini, Dict* env_kwargs, int seed) {
    if (ir_method(ini) == IR_T2) {
        dict_set(env_kwargs, "t2_seed", (double)seed);
    }
}

Intrinsic* ir_create(PuffeRL* p, Ini* ini) {
    int method = ir_method(ini);
    if (method == IR_NONE) {
        return NULL;
    }
    Intrinsic* ir = (Intrinsic*)calloc(1, sizeof(Intrinsic));
    ir->method = method;
    if (method == IR_T2) {
        ir->t2 = t2_create(p, ini);
        p->t2 = ir->t2;
    } else {
        ir->bl = baseline_create(p, ini, method);
    }
    return ir;
}

void ir_worker_step(PuffeRL* p, int buf, int t, cudaStream_t stream) {
    Intrinsic* ir = p->ir;
    if (ir->t2) {
        t2_worker_step(p, buf, t, stream);
    } else {
        baseline_worker_step(p, buf, t, stream);
    }
}

void ir_apply_rewards(PuffeRL* p, RolloutBuf* train_view, int slot, cudaStream_t stream) {
    Intrinsic* ir = p->ir;
    if (ir->t2) {
        t2_apply_rewards(p, train_view, slot, stream);
    } else {
        baseline_apply_rewards(p, train_view, slot, stream);
    }
}

void ir_train(PuffeRL* p) {
    Intrinsic* ir = p->ir;
    if (ir->t2) {
        t2_train(p);
    } else {
        baseline_train(p);
    }
}

void ir_log(PuffeRL* p, Dict* out) {
    Intrinsic* ir = p->ir;
    if (ir->t2) {
        t2_log(p, out);
    } else {
        baseline_log(p, out);
    }
}

void ir_save(PuffeRL* p, const char* policy_checkpoint) {
    Intrinsic* ir = p->ir;
    char path[4200];
    if (ir->t2) {
        snprintf(path, sizeof(path), "%s.t2", policy_checkpoint);
        t2_save(p, path);
    } else {
        snprintf(path, sizeof(path), "%s.ir", policy_checkpoint);
        baseline_save(p, path);
    }
}
