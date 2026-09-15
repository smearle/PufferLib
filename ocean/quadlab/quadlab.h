// Native shared-sequence QuadLab adapter. Audit fields never enter observations or rewards.
#pragma once
#include <assert.h>
typedef float obs_t;
#define PUF_T2_TOKENS 65
#define PUF_T2_PAIRED 1
#include "pufferenv.h"
#include "quadlab_layouts.h"
#define OBS_SIZE 317
#define ACT_SIZES {4}
#define NUM_ATNS 1
#define TL_H 24
#define TL_W 33
struct Log {
    float perf, score, episode_return, episode_length, full_mazes, later_equivalents, ladder_uses, multi_maze_fraction, monster_fraction, n;
};
struct Env {
    Log log;
    Agent agents[1];
    unsigned rng;
    int num_agents, tag, boundary_reached, tick, max_steps, eval, episode, row, col;
    unsigned layout, known;
    int generation, prior_choices, prior_full, first_choices;
    uint64_t world_seed;
    uint64_t salt, dyn;
    unsigned char grid[TL_H][TL_W], tok[65];
    int monsters[5][2], monster_steps;
};
static void tl_maze(unsigned char maze[11][11], unsigned id) {
    memset(maze, 2, 121);
    for (int r=0;r<5;r++) for(int c=0;c<5;c++) {
        maze[2*r+1][2*c+1]=1;
        if (!r && !c) continue;
        int north = r && (!c || ((id >> ((r-1)*4+c-1)) & 1));
        maze[2*r+(north?0:1)][2*c+(north?1:0)]=1;
    }
    maze[5][10]=1;
}
static void tl_grid(Env* e, unsigned id) {
    unsigned char west[11][11], east[11][11];
    tl_maze(west, TL_FIXED_ID); tl_maze(east,id);
    memset(e->grid,0,sizeof(e->grid));
    for(int r=0;r<11;r++) for(int c=0;c<11;c++) {
        e->grid[r+1][c]=west[r][c]; e->grid[r+1][c+22]=east[r][10-c];
    }
    for(int c=10;c<=22;c++) e->grid[6][c]=1;
    for(int r=12;r<=22;r++) for(int c=11;c<=21;c++)
        e->grid[r][c]=(r==12||r==22||c==11||c==21)?2:1;
    for(int r=6;r<=12;r++) e->grid[r][16]=1;
    for(int r=1;r<=6;r++) e->grid[r][16]=1;
    e->grid[0][16]=4;
}
static void tl_observe(Env* e) {
    float* o=e->agents[0].observations;
    memset(o,0,OBS_SIZE*sizeof(float));
    for(int y=0;y<7;y++) for(int x=0;x<9;x++) {
        int r=e->row+y-3,c=e->col+x-4,k=y*9+x,v=0;
        if(r>=0&&r<TL_H&&c>=0&&c<TL_W) v=e->grid[r][c];
        for(int m=0;m<5;m++) if(e->monsters[m][0]==r&&e->monsters[m][1]==c) v=3;
        e->tok[k]=(unsigned char)v; o[5*k+v]=1;
    }
    e->tok[63]=e->row; e->tok[64]=e->col;
    o[315]=(float)e->row*(1.0f/(TL_H-1)); o[316]=(float)e->col*(1.0f/(TL_W-1));
    for(int r=1;r<5;r++) for(int c=1;c<5;c++) {
        int a=abs(e->row-(1+2*r))<=3 && abs(e->col-(32-(2*c+1)))<=4;
        int b=abs(e->row-(2+2*r))<=3 && abs(e->col-(32-2*c))<=4;
        if(a||b) e->known |= 1u<<((r-1)*4+c-1);
    }
}
static void tl_reset_layout(Env* e,unsigned id) {
    e->layout=id; tl_grid(e,id); e->row=6;e->col=16;e->tick=0;e->known=0;e->monster_steps=0;e->generation=0;e->prior_choices=0;e->prior_full=0;e->first_choices=0;
    const int spawn[5][2]={{15,14},{15,18},{17,16},{19,14},{19,18}};
    memcpy(e->monsters,spawn,sizeof(spawn)); tl_observe(e);
}
void puf_init(Env* e,Dict* kw) {
    e->num_agents=1;e->episode=0;e->max_steps=1984;
    if(dict_find(kw,"max_steps")) e->max_steps=(int)dict_get(kw,"max_steps");
    if(dict_find(kw,"eval")) e->eval=(int)dict_get(kw,"eval");
    if(dict_find(kw,"t2_seed")) e->salt=(uint64_t)dict_get(kw,"t2_seed");
}
void puf_reset(Env* e) {
    uint64_t world=puf_t2_world_seed(e->rng,e->episode,e->salt);
    e->dyn=puf_t2_dyn_seed(e->rng,e->episode,e->salt);
    unsigned id=(unsigned)world & 65535;
    if(e->eval) id=TL_HELDOUT[(e->rng>>1)%128];
    else for(;;) {
        int excluded=0;for(int i=0;i<128;i++) excluded |= id==TL_HELDOUT[i];
        if(!excluded) break;
        world=puf_t2_mix(world);id=(unsigned)world & 65535;
    }
    e->world_seed=world;e->episode++;tl_reset_layout(e,id);
}
static void ql_regenerate(Env* e,unsigned id) {
    int k=__builtin_popcount(e->known);
    if(e->generation==0) e->first_choices=k;
    e->prior_choices+=k;e->prior_full+=e->known==65535;e->generation++;
    e->layout=id;tl_grid(e,id);e->row=6;e->col=16;e->known=0;
    const int spawn[5][2]={{15,14},{15,18},{17,16},{19,14},{19,18}};
    memcpy(e->monsters,spawn,sizeof(spawn));
}
static void tl_step_choices(Env* e,int action,const int choices[5],unsigned next_layout) {
    const int d[5][2]={{0,0},{0,-1},{0,1},{-1,0},{1,0}};
    assert(action>=0&&action<4);
    int r=e->row+d[action+1][0],c=e->col+d[action+1][1],occupied=0;
    for(int m=0;m<5;m++) occupied |= e->monsters[m][0]==r&&e->monsters[m][1]==c;
    if(r>=0&&r<TL_H&&c>=0&&c<TL_W&&(e->grid[r][c]==1||e->grid[r][c]==4)&&!occupied) {e->row=r;e->col=c;}
    if(e->row==0&&e->col==16) ql_regenerate(e,next_layout);
    else for(int m=0;m<5;m++) {
        int ch=choices[m];assert(ch>=0&&ch<5);
        r=e->monsters[m][0]+d[ch][0];c=e->monsters[m][1]+d[ch][1];occupied=0;
        for(int n=0;n<5;n++) if(n!=m) occupied |= e->monsters[n][0]==r&&e->monsters[n][1]==c;
        if(r>12&&r<22&&c>11&&c<21&&e->grid[r][c]==1&&!occupied&&!(r==e->row&&c==e->col))
            {e->monsters[m][0]=r;e->monsters[m][1]=c;}
    }
    e->tick++;e->monster_steps+=e->row>=12;tl_observe(e);
}
void puf_step(Env* e) {
    int choices[5];
    // Rejection removes modulo bias; reset and transition streams are separate.
    for(int m=0;m<5;m++) {uint64_t v;do {e->dyn=puf_t2_mix(e->dyn);v=e->dyn;} while(v==UINT64_MAX);choices[m]=v%5;}
    unsigned next_layout=e->eval?TL_SEQUENCES[(e->rng>>1)%128][e->generation+1]:
        (unsigned)puf_t2_mix(e->world_seed ^ ((uint64_t)(e->generation+1)*0x9E3779B97F4A7C15ULL))&65535;
    tl_step_choices(e,(int)e->agents[0].actions[0],choices,next_layout);
    e->agents[0].rewards[0]=0;e->agents[0].terminals[0]=0;
    if(e->tick>=e->max_steps) {
        e->agents[0].terminals[0]=1;e->log.n++;e->log.episode_length+=e->tick;
        int total=e->prior_choices+__builtin_popcount(e->known);
        int first=e->generation?e->first_choices:__builtin_popcount(e->known);
        e->log.perf+=total/16.0f;e->log.score+=total/16.0f;
        e->log.full_mazes+=e->prior_full+(e->known==65535);
        e->log.later_equivalents+=(total-first)/16.0f;e->log.ladder_uses+=e->generation;
        e->log.multi_maze_fraction+=total>16;e->log.monster_fraction+=(float)e->monster_steps/e->tick;
        if(e->eval) printf("EPISODE {\"lane\":%u,\"initial_layout\":%u,\"choices\":%d,\"later_choices\":%d,\"full_mazes\":%d,\"ladder_uses\":%d,\"monster_steps\":%d,\"steps\":%d}\n",e->rng,TL_HELDOUT[(e->rng>>1)%128],total,total-first,e->prior_full+(e->known==65535),e->generation,e->monster_steps,e->tick);
        puf_reset(e);
    }
}
void puf_t2_tokens(Env* e,unsigned char* out) {memcpy(out,e->tok,65);}
void puf_render(Env* e) {(void)e;}
void puf_close(Env* e) {(void)e;}
void puf_log(Log* l,Dict* out) {
    dict_set(out,"perf",l->perf);dict_set(out,"score",l->score);
    dict_set(out,"episode_return",l->episode_return);dict_set(out,"episode_length",l->episode_length);
    dict_set(out,"full_mazes",l->full_mazes);dict_set(out,"later_equivalents",l->later_equivalents);
    dict_set(out,"ladder_uses",l->ladder_uses);dict_set(out,"multi_maze_fraction",l->multi_maze_fraction);
    dict_set(out,"monster_fraction",l->monster_fraction);
    dict_set(out,"n",l->n);
}
