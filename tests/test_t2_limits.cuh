// Current-weight histories and q_ctrl/q1 against an independent sequential
// CPU encoder/head, including the original limit and multiple overflow chunks.
struct LimitOracle {
    T2* t;
    float *tok, *act, *input, *gru[T2_MAX_LAYERS];
    float *role, *head, *pos, *output;
};
static LimitOracle limit_oracle(T2* t) {
    LimitOracle o={};o.t=t;
    o.tok=prec_host(t->w.tok_embed);o.act=prec_host(t->w.act_embed);
    o.input=prec_host(t->w.w_in);o.role=prec_host(t->w.role);
    o.head=prec_host(t->w.w_h);o.pos=prec_host(t->w.pos_embed);o.output=prec_host(t->w.w_out);
    for(int l=0;l<t->Lg;l++)o.gru[l]=prec_host(t->w.gru[l]);
    return o;
}
static void limit_free(LimitOracle o) {
    free(o.tok);free(o.act);free(o.input);free(o.role);free(o.head);free(o.pos);free(o.output);
    for(int l=0;l<o.t->Lg;l++)free(o.gru[l]);
}
static void limit_step(LimitOracle o,const unsigned char* tok,const int* act,float* state,float* out) {
    T2* t=o.t;int H=t->H,width=t->E+t->EA;
    float* in=(float*)calloc(width,sizeof(float));
    float* x=(float*)calloc(H,sizeof(float));
    for(int e=0;e<t->E;e++) {
        double v=0;for(int k=0;k<t->L;k++)v+=o.tok[((long)k*256+tok[k])*t->E+e];in[e]=v;
    }
    for(int e=0;e<t->EA;e++)for(int h=0;h<t->heads;h++)
        in[t->E+e]+=o.act[(long)(t->head_off[h]+act[h])*t->EA+e];
    for(int h=0;h<H;h++) {
        double v=0;for(int k=0;k<width;k++)v+=(double)o.input[h*width+k]*in[k];x[h]=v;
    }
    for(int l=0;l<t->Lg;l++) {
        for(int h=0;h<H;h++) {
            double g[3]={0,0,0};
            for(int k=0;k<H;k++)for(int j=0;j<3;j++)g[j]+=(double)o.gru[l][(j*H+h)*H+k]*x[k];
            double z=audit_sigmoid(g[1]),candidate=g[0]>=0?g[0]+.5:audit_sigmoid(g[0]);
            state[l*H+h]=(float)((1-z)*state[l*H+h]+z*candidate);
            double mix=audit_sigmoid(g[2]);out[h]=(float)(mix*state[l*H+h]+(1-mix)*x[h]);
        }
        if(l+1<t->Lg)memcpy(x,out,H*sizeof(float));
    }
    free(in);free(x);
}
static double limit_nll(LimitOracle o,const float* query,const float* support,const unsigned char* target) {
    T2* t=o.t;int H=t->H,D=t->D;
    double* u=(double*)calloc(D,sizeof(double));
    for(int d=0;d<D;d++)for(int h=0;h<H;h++)
        u[d]+=(double)o.head[d*2*H+h]*query[h]+(double)o.head[d*2*H+H+h]*(support[h]+o.role[h]);
    double total=0;
    for(int k=0;k<t->L;k++) {
        double logits[256],maximum=-INFINITY,sum=0;
        for(int v=0;v<256;v++) {
            logits[v]=0;
            for(int d=0;d<D;d++) {
                double pre=u[d]+o.pos[k*D+d];
                double gelu=.5*pre*(1+erf(pre/sqrt(2.0)));
                logits[v]+=o.output[v*D+d]*gelu;
            }
            maximum=fmax(maximum,logits[v]);
        }
        for(int v=0;v<256;v++)sum+=exp(logits[v]-maximum);
        total+=maximum+log(sum)-logits[target[k]];
    }
    free(u);return total;
}
static double limit_live_error(PuffeRL* p) {
    T2* t=p->t2;LimitOracle o=limit_oracle(t);int H=t->H;
    unsigned char* tok=(unsigned char*)malloc((long)t->A*t->M*t->L);
    int* act=(int*)malloc((long)t->A*t->M*t->heads*sizeof(int));
    cudaMemcpy(tok,t->ep_tok.data,(long)t->A*t->M*t->L,cudaMemcpyDeviceToHost);
    cudaMemcpy(act,t->ep_act.data,(long)t->A*t->M*t->heads*sizeof(int),cudaMemcpyDeviceToHost);
    float* actual=prec_host(t->state);double error=0;
    for(int a=0;a<t->A;a++) {
        float* state=(float*)calloc(t->Lg*H,sizeof(float));float* out=(float*)calloc(H,sizeof(float));
        for(int s=0;s<t->host_len[a];s++)limit_step(o,tok+((long)a*t->M+s)*t->L,act+((long)a*t->M+s)*t->heads,state,out);
        for(size_t s=0;s<t->overflow[a].len;s++)limit_step(o,t->overflow[a].tok+s*t->L,t->overflow[a].act+s*t->heads,state,out);
        for(int l=0;l<t->Lg;l++)for(int h=0;h<H;h++)error=fmax(error,fabs(state[l*H+h]-actual[((long)l*t->A+a)*H+h]));
        free(state);free(out);
    }
    limit_free(o);free(tok);free(act);free(actual);return error;
}
static void test_t2_history_limits(Ini* ini) {
    if(USE_BF16) {printf("SKIP history-limit FP32 CPU reference: bf16 build\n");return;}
    for(int layers=1;layers<=2;layers++) {
        PuffeRL* p=make_trainer(ini,"t2",1,layers);T2* t=p->t2;int H=t->H,M=t->M;
        int lengths[]={M-1,M,M+1,2*M,2*M+7};
        for(int ci=0;ci<5;ci++) {
            int len=lengths[ci];
            free(t->overflow[0].tok);free(t->overflow[0].act);t->overflow[0]=(T2Overflow){};
            memset(t->host_len,0,t->A*sizeof(int));
            cudaMemset(t->state.data,0,numel(t->state.shape)*sizeof(precision_t));
            unsigned char* tok=(unsigned char*)malloc((len+2)*t->L);
            int* act=(int*)malloc((len+2)*t->heads*sizeof(int));
            for(int s=0;s<len+2;s++) {
                for(int k=0;k<t->L;k++)tok[s*t->L+k]=(s*13+k*7)%256;
                for(int h=0;h<t->heads;h++)act[s*t->heads+h]=s%t->act_sizes[h];
#ifdef PUFFER_NETHACK
                int nv=0,na=0;const signed char* consumed=env_head_consume_map(&nv,&na);
                for(int h=1;h<t->heads;h++)if(consumed&&!consumed[act[s*t->heads]*na+h])act[s*t->heads+h]=t->act_sizes[h];
#endif
            }
            int prefix=min(len,M);t->host_len[0]=prefix;
            cudaMemcpy(t->ep_tok.data,tok,prefix*t->L,cudaMemcpyHostToDevice);
            cudaMemcpy(t->ep_act.data,act,prefix*t->heads*sizeof(int),cudaMemcpyHostToDevice);
            for(int s=M;s<len;s++) {
                memcpy(t->host_tok,tok+s*t->L,t->L);
                for(int h=0;h<t->heads;h++)p->vec->actions[h]=(float)act[s*t->heads+h];
                t2_capture_overflow(t,p->vec->actions,0,1);
            }
            CHECK(t->overflow[0].len==(size_t)(len-prefix),"all overflow records retained");
            for(int round=0;round<2;round++) {
                float* w=prec_host(t->w.w_in);
                for(long k=0;k<numel(t->w.w_in.shape);k++)w[k]*=1.05f;
                prec_upload(t->w.w_in.data,w,numel(t->w.w_in.shape));free(w);
                cudaMemset(t->state.data,0,numel(t->state.shape)*sizeof(precision_t));
                t2_reencode(t,p->train_stream);cudaDeviceSynchronize();
                double error=limit_live_error(p);CHECK(error<2e-5,"history limit L%d n%d round%d error %.9g",layers,len,round,error);
                LimitOracle o=limit_oracle(t);
                float *literal=(float*)calloc(layers*H,sizeof(float)),*out=(float*)calloc(H,sizeof(float));
                float* expected_prefix=(float*)calloc(H,sizeof(float));
                for(int s=0;s<len;s++) {
                    limit_step(o,tok+s*t->L,act+s*t->heads,literal,out);
                    if(s==prefix-1)memcpy(expected_prefix,out,H*sizeof(float));
                }
                float* stored=prec_host(t->out_last);double summary_error=0;
                for(int h=0;h<H;h++)summary_error=fmax(summary_error,fabs(stored[h]-expected_prefix[h]));
                CHECK(summary_error<2e-5,"bounded replay summary L%d n%d",layers,len);
                float *query=(float*)calloc(H,sizeof(float)),*ctrl=(float*)calloc(H,sizeof(float)),*q1=(float*)calloc(H,sizeof(float));
                float *query_state=(float*)malloc(layers*H*sizeof(float)),*ctrl_state=(float*)malloc(layers*H*sizeof(float));
                memcpy(query_state,literal,layers*H*sizeof(float));memcpy(ctrl_state,literal,layers*H*sizeof(float));
                int* pad=(int*)malloc(t->heads*sizeof(int));for(int h=0;h<t->heads;h++)pad[h]=t->act_sizes[h];
                limit_step(o,tok+len*t->L,act+len*t->heads,query_state,query);
                limit_step(o,tok+len*t->L,pad,ctrl_state,ctrl);
                limit_step(o,tok+(len+1)*t->L,pad,query_state,q1);
                for(int a=0;a<t->A;a++)memcpy(t->host_tok+(long)a*t->L,tok+(len+1)*t->L,t->L);
                cudaMemcpy(t->roll[0].tok_cur.data,tok+len*t->L,t->L,cudaMemcpyHostToDevice);
                float* actions=(float*)calloc(t->A*t->heads,sizeof(float));for(int h=0;h<t->heads;h++)actions[h]=act[len*t->heads+h];
                cudaMemcpy(p->rollouts.actions.data,actions,t->A*t->heads*sizeof(float),cudaMemcpyHostToDevice);
                cudaMemset(p->env.terminals.data,0,t->A*sizeof(float));
                cudaMemcpy(t->ep_len.data,t->host_len,t->A*sizeof(int),cudaMemcpyHostToDevice);
                t2_rollout_step(p,0,0,p->train_stream);cudaStreamSynchronize(p->train_stream);
                float* nll=to_host(t->roll[0].nll.data,2*t->A,sizeof(float));
                double ctrl_error=fabs(nll[0]-limit_nll(o,query,ctrl,tok+(len+1)*t->L));
                double q1_error=fabs(nll[t->A]-limit_nll(o,query,q1,tok+(len+1)*t->L));
                CHECK(ctrl_error<.002 && q1_error<.002,"q_ctrl/q1 L%d n%d errors %.8g / %.8g",layers,len,ctrl_error,q1_error);
                printf("PASS history limit L%d n%d update%d: state %.3g q_ctrl %.3g q1 %.3g\n",layers,len,round,error,ctrl_error,q1_error);
                // Restore prefix bytes overwritten by the synthetic extra score.
                cudaMemcpy(t->ep_tok.data,tok,prefix*t->L,cudaMemcpyHostToDevice);
                cudaMemcpy(t->ep_act.data,act,prefix*t->heads*sizeof(int),cudaMemcpyHostToDevice);
                free(nll);free(actions);free(pad);free(query);free(ctrl);free(q1);free(query_state);free(ctrl_state);
                free(stored);free(expected_prefix);free(literal);free(out);limit_free(o);
            }
            memset(p->vec->terminals,0,t->A*sizeof(float));p->vec->terminals[0]=1;
            t2_episode_bookkeeping(p,0,p->train_stream);
            CHECK(t->host_len[0]==0 && !t->overflow[0].len && !t->overflow[0].tok && !t->overflow[0].act,"terminal releases CPU suffix");
            free(tok);free(act);
        }
        close_pufferl(p);
    }
    // Real async/four-worker collection verifies the CPU suffix against actual
    // online GPU carries before an update, then against the new-weight oracle.
    PuffeRL* p=make_trainer(ini,"t2",4,2,1);T2* t=p->t2;size_t covered=0;
    for(int epoch=0;epoch<12;epoch++) {
        rollouts(p);cudaDeviceSynchronize();
        double before=limit_live_error(p);CHECK(before<2e-5,"native overflow capture %.9g",before);
        for(int a=0;a<t->A;a++)covered+=t->overflow[a].len;
        float* w=prec_host(t->w.w_in);for(long k=0;k<numel(t->w.w_in.shape);k++)w[k]*=1.01f;
        prec_upload(t->w.w_in.data,w,numel(t->w.w_in.shape));free(w);
        t2_reencode(t,p->train_stream);cudaDeviceSynchronize();
        double after=limit_live_error(p);CHECK(after<2e-5,"native overflow re-encode %.9g",after);
    }
    CHECK(covered>0,"real async four-worker overflow coverage");
    printf("PASS native async/four-worker overflow: %zu resident suffix steps checked\n",covered);
    close_pufferl(p);
}
