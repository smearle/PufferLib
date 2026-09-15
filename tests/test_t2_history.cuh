// Independent sequential host encoder. No native gate/scan is called to
// construct the reference; stored records and current weights are inputs.
struct AuditWeights {
    float *tok,*act,*input,*gru;
};
struct AuditErrors {
    double lane,prev,episode,reservoir,summary;
    long active_steps,reservoir_steps;
};
static double audit_sigmoid(double x) {return 1.0/(1.0+exp(-x));}
static void audit_encode_step(T2* t,AuditWeights w,const unsigned char* tok,
        const int* actions,float* state,float* output) {
    int H=t->H,E=t->E,EA=t->EA;
    float* input=(float*)calloc(E+EA,sizeof(float));
    float* x=(float*)calloc(H,sizeof(float));
    for(int e=0;e<E;e++) {
        double sum=0;
        for(int p=0;p<t->L;p++)sum+=w.tok[((long)p*T2_VOCAB+tok[p])*E+e];
        input[e]=(float)sum;
    }
    for(int e=0;e<EA;e++) for(int h=0;h<t->heads;h++)
        input[E+e]+=w.act[(long)(t->head_off[h]+actions[h])*EA+e];
    for(int h=0;h<H;h++) {
        double sum=0;
        for(int e=0;e<E+EA;e++)sum+=(double)w.input[h*(E+EA)+e]*input[e];
        x[h]=(float)sum;
    }
    for(int h=0;h<H;h++) {
        double sums[3]={0,0,0};
        for(int g=0;g<3;g++)for(int k=0;k<H;k++)
            sums[g]+=(double)w.gru[(g*H+h)*H+k]*x[k];
        double z=audit_sigmoid(sums[1]);
        double candidate=sums[0]>=0?sums[0]+.5:audit_sigmoid(sums[0]);
        state[h]=(float)((1-z)*state[h]+z*candidate);
        double s=audit_sigmoid(sums[2]);
        output[h]=(float)(s*state[h]+(1-s)*x[h]);
    }
    free(input);free(x);
}
static AuditErrors audit_current_histories(PuffeRL* p) {
    T2* t=p->t2;CHECK(!USE_BF16&&t->Lg==1,"independent audit geometry");
    cudaDeviceSynchronize();
    AuditWeights w={prec_host(t->w.tok_embed),prec_host(t->w.act_embed),prec_host(t->w.w_in),prec_host(t->w.gru[0])};
    AuditErrors errors={0};
    float *lane=prec_host(t->state),*prev=prec_host(t->state_prev);
    float *ep=prec_host(t->ep_state),*res=prec_host(t->res_state),*final=prec_host(t->res_final);
    unsigned char* tokens=(unsigned char*)malloc((long)max(t->A,t->C)*t->M*t->L);
    int* actions=(int*)malloc((long)max(t->A,t->C)*t->M*t->heads*sizeof(int));
    for(int group=0;group<2;group++) {
        int records=group?t->C:t->A;
        cudaMemcpy(tokens,group?t->res_tok.data:t->ep_tok.data,(long)records*t->M*t->L,cudaMemcpyDeviceToHost);
        cudaMemcpy(actions,group?t->res_act.data:t->ep_act.data,(long)records*t->M*t->heads*sizeof(int),cudaMemcpyDeviceToHost);
        for(int j=0;j<records;j++) {
            int len=group?t->res_len[j]:t->host_len[j];
            if(len==0)continue;
            size_t total = len + (group ? 0 : t->overflow[j].len);
            float* state=(float*)calloc(t->H,sizeof(float));
            float* output=(float*)calloc(t->H,sizeof(float));
            for(int s=0;s<len;s++) {
                if(!group&&(size_t)s==total-1)for(int h=0;h<t->H;h++)
                    errors.prev=fmax(errors.prev,fabs(state[h]-prev[j*t->H+h]));
                long row=(long)j*t->M+s;
                audit_encode_step(t,w,tokens+row*t->L,actions+row*t->heads,state,output);
                for(int h=0;h<t->H;h++) {
                    double e=fabs(state[h]-(group?res:ep)[row*t->H+h]);
                    if(group)errors.reservoir=fmax(errors.reservoir,e);
                    else errors.episode=fmax(errors.episode,e);
                }
                if(group)errors.reservoir_steps++;else errors.active_steps++;
            }
            if(!group)for(size_t s=0;s<t->overflow[j].len;s++) {
                if(s+1==t->overflow[j].len)for(int h=0;h<t->H;h++)
                    errors.prev=fmax(errors.prev,fabs(state[h]-prev[j*t->H+h]));
                audit_encode_step(t,w,t->overflow[j].tok+s*t->L,
                    t->overflow[j].act+s*t->heads,state,output);
                errors.active_steps++;
            }
            for(int h=0;h<t->H;h++) {
                if(group)errors.summary=fmax(errors.summary,fabs(output[h]-final[j*t->H+h]));
                else errors.lane=fmax(errors.lane,fabs(state[h]-lane[j*t->H+h]));
            }
            free(state);free(output);
        }
    }
    free(w.tok);free(w.act);free(w.input);free(w.gru);free(lane);free(prev);free(ep);free(res);free(final);free(tokens);free(actions);
    return errors;
}
static void audit_print_errors(FILE* f,AuditErrors e) {
    fprintf(f,"{\"lane_max_error\":%.12g,\"previous_max_error\":%.12g,\"active_record_max_error\":%.12g,\"reservoir_max_error\":%.12g,\"peer_summary_max_error\":%.12g,\"active_steps\":%ld,\"reservoir_steps\":%ld}",e.lane,e.prev,e.episode,e.reservoir,e.summary,e.active_steps,e.reservoir_steps);
}

// This regression changes actual model weights. Equality under unchanged
// weights cannot detect an active record that was never refreshed.
static void test_updated_history_records(PuffeRL* p) {
    if (USE_BF16) { printf("SKIP updated history host reference: bf16 build\n"); return; }
    T2* t = p->t2;
    AuditErrors before = audit_current_histories(p);
    CHECK(before.active_steps > 0 && before.reservoir_steps > 0, "updated history coverage");
    CHECK(fmax(before.episode, fmax(before.reservoir, before.lane)) < 2e-5,
        "independent sequential histories before update");
    long n = numel(t->master.shape);
    float* weights = to_host(t->master.data, n, sizeof(float));
    t2_train(p);
    cudaDeviceSynchronize();
    float* updated = to_host(t->master.data, n, sizeof(float));
    double change = 0;
    for (long i = 0; i < n; i++) change = fmax(change, fabs(weights[i] - updated[i]));
    CHECK(change > 0, "updated history test changed actual weights");
    AuditErrors after = audit_current_histories(p);
    CHECK(fmax(after.episode, fmax(after.reservoir, fmax(after.lane, fmax(after.prev, after.summary)))) < 2e-5,
        "current-weight histories: active %.3e replay %.3e lane %.3e prev %.3e summary %.3e",
        after.episode, after.reservoir, after.lane, after.prev, after.summary);
    printf("PASS updated history records: %ld active / %ld replay steps, errors %.2e / %.2e\n",
        after.active_steps, after.reservoir_steps, after.episode, after.reservoir);
    free(weights); free(updated);
}
