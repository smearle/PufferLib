// Adversarial legal launch geometry: one writer per block ensures a record
// crosses scheduling waves. Every writer must observe the same old length.
static void test_record_publication(PuffeRL* p) {
    T2* t=p->t2;T2Rollout* r=&t->roll[0];int A=t->A,M=t->M,L=t->L,H=t->H,Lg=t->Lg;
    CHECK(t->num_buffers==1,"record test geometry");
    cudaStream_t stream=p->default_stream;
    cudaMemset(t->ep_len.data,0,A*sizeof(int));
    cudaMemset(t->ep_tok.data,255,(long)A*M*L);
    cudaMemset(p->env.terminals.data,0,A*sizeof(float));
    unsigned char* expected=(unsigned char*)malloc((long)A*L);
    unsigned char* actual=(unsigned char*)malloc((long)A*M*L);
    cudaMemcpy(expected,r->tok_cur.data,(long)A*L,cudaMemcpyDeviceToHost);
    int steps=min(M,5);
    for(int s=0;s<steps;s++) {
        t2_record_kernel<<<A*(L+Lg*H),1,0,stream>>>(t->ep_tok.data,t->ep_act.data,t->ep_state.data,
            t->ep_len.data,t->out_last.data,r->tok_cur.data,r->act.data,t->state.data,r->out.data,
            p->env.terminals.data,0,A,A,L,M,Lg,H,t->heads);
#ifndef T2_TEST_OLD_RECORD
        t2_advance_record_lengths<<<grid_size(A),BLOCK_SIZE,0,stream>>>(t->ep_len.data,p->env.terminals.data,0,A,M);
#endif
    }
    cudaStreamSynchronize(stream);cudaMemcpy(actual,t->ep_tok.data,(long)A*M*L,cudaMemcpyDeviceToHost);
    long bad=0;for(int a=0;a<A;a++)for(int s=0;s<steps;s++)for(int k=0;k<L;k++)
        bad+=actual[((long)a*M+s)*L+k]!=expected[(long)a*L+k];
    printf("Record publication: %ld bad bytes / %d checked under one-thread blocks\n",bad,A*steps*L);
    CHECK(bad==0,"all record writers use the old published length");
    free(expected);free(actual);
}
