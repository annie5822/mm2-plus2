#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include "mmpriv.h"
#include "kalloc.h"
#include "krmq.h"
#include <stdbool.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <future>
#include <map>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>
    #include "parallel_chaining_v2_22.h"
#endif

extern float mean_itr_1_, mean_itr_2_;
extern int32_t max_itr_1, max_itr_2;
extern int32_t num_threads_b2b;
extern float rmq_1_time_, rmq_2_time_, btk_1_time_, btk_2_time_;
extern int32_t count_stages, is_g2g_aln, is_splice_sr;
extern std::fstream anchor_dist_file, count_chains_file;

/* sub_part_t 不依賴 lc_elem_t，放最前面 */
typedef struct {
    int64_t start;
    int64_t end;
} sub_part_t;

static sub_part_t *build_sub_partitions(const mm128_t *a,
                                         const int64_t *st_vec,
                                         const int64_t *end_vec,
                                         int32_t  num_chr,
                                         int64_t  max_dist,
                                         int32_t *n_sub_out)
{
    int32_t n_sub = 0;
    for (int32_t c = 0; c < num_chr; c++) {
        n_sub++;
        for (int64_t i = st_vec[c] + 1; i <= end_vec[c]; i++) {
            int64_t gap = (int64_t)(uint32_t)a[i].x - (int64_t)(uint32_t)a[i-1].x;
            if (gap > max_dist) n_sub++;
        }
    }
    sub_part_t *sp = (sub_part_t *)malloc(n_sub * sizeof(sub_part_t));
    int32_t idx = 0;
    for (int32_t c = 0; c < num_chr; c++) {
        int64_t seg_start = st_vec[c];
        for (int64_t i = st_vec[c] + 1; i <= end_vec[c]; i++) {
            int64_t gap = (int64_t)(uint32_t)a[i].x - (int64_t)(uint32_t)a[i-1].x;
            if (gap > max_dist) {
                sp[idx].start = seg_start; sp[idx].end = i - 1; idx++;
                seg_start = i;
            }
        }
        sp[idx].start = seg_start; sp[idx].end = end_vec[c]; idx++;
    }
    assert(idx == n_sub);
    *n_sub_out = n_sub;
    return sp;
}

/* =========================================================
   [OPT-1] force_split_partitions:
   將超過 max_part_len*2 的 partition 強制切割成多個 tile。
   在目標切割點 ±200 的範圍內搜尋 reference gap 最大處作為斷點，
   盡量減少跨 tile 的 chain 斷裂。
   對 primate g2g，chr1 可有數百萬 anchor，切開後才能充分利用多核。
   ========================================================= */
static void force_split_partitions(const mm128_t *a,
                                   sub_part_t   *&parts,
                                   int32_t      &n_parts,
                                   int64_t       max_part_len)
{
    bool need_split = false;
    for (int32_t i = 0; i < n_parts; i++) {
        if (parts[i].end - parts[i].start + 1 > max_part_len * 2) {
            need_split = true; break;
        }
    }
    if (!need_split) return;

    /* count final partition number (upper bound) */
    int32_t n_final = 0;
    for (int32_t i = 0; i < n_parts; i++) {
        int64_t sz = parts[i].end - parts[i].start + 1;
        if (sz > max_part_len * 2)
            n_final += (int32_t)((sz + max_part_len - 1) / max_part_len);
        else
            n_final++;
    }

    sub_part_t *np = (sub_part_t *)malloc(n_final * sizeof(sub_part_t));
    int32_t idx = 0;
    for (int32_t i = 0; i < n_parts; i++) {
        int64_t sz = parts[i].end - parts[i].start + 1;
        if (sz <= max_part_len * 2) {
            np[idx++] = parts[i];
        } else {
            int32_t n_tiles = (int32_t)((sz + max_part_len - 1) / max_part_len);
            int64_t tile_sz = sz / n_tiles;
            int64_t cur = parts[i].start;
            for (int32_t t = 0; t < n_tiles; t++) {
                np[idx].start = cur;
                if (t == n_tiles - 1) {
                    np[idx].end = parts[i].end;
                } else {
                    int64_t target = cur + tile_sz - 1;
                    /* search ±200 for the largest ref gap as split point */
                    int64_t best = target, best_gap = 0;
                    int64_t slo = target - 200; if (slo < cur + 1) slo = cur + 1;
                    int64_t shi = target + 200; if (shi >= parts[i].end) shi = parts[i].end - 1;
                    for (int64_t k = slo; k <= shi; k++) {
                        int64_t g = (int64_t)(uint32_t)a[k+1].x - (int64_t)(uint32_t)a[k].x;
                        if (g > best_gap) { best_gap = g; best = k; }
                    }
                    np[idx].end = best;
                }
                cur = np[idx].end + 1;
                idx++;
            }
        }
    }
    free(parts);
    parts = np;
    n_parts = idx;
}

/* backtrack helpers (不依賴 lc_elem_t) */
static int64_t mg_chain_bk_end_par(const int32_t max_drop, const std::vector<mm128_t> &z, const int32_t *f, std::pair<int64_t, int32_t> *p_t_vec, const int64_t k_, const int64_t k)
{
    int64_t i = k_, end_i = -1, max_i = i; int32_t max_s = 0;
    if (i < 0 || p_t_vec[i].second != 0) return i;
    do {
        int32_t s = 0; p_t_vec[i].second = 2; end_i = i = p_t_vec[i].first;
        s = i < 0? z[k].x : (int32_t)z[k].x - f[i];
        if (__builtin_expect(s > max_s, 1)) { max_s = s; max_i = i; }
        else if (__builtin_expect(max_s - s > max_drop, 0)) break;
    } while (i >= 0 && p_t_vec[i].second == 0);
    for (i = k_; i >= 0 && i != end_i; i = p_t_vec[i].first) p_t_vec[i].second = 0;
    return max_i;
}

static int64_t mg_chain_bk_end(int32_t max_drop, const mm128_t *z, const int32_t *f, const int64_t *p, int32_t *t, int64_t k)
{
    int64_t i = z[k].y, end_i = -1, max_i = i; int32_t max_s = 0;
    if (i < 0 || t[i] != 0) return i;
    do {
        int32_t s; t[i] = 2; end_i = i = p[i];
        s = i < 0? z[k].x : (int32_t)z[k].x - f[i];
        if (s > max_s) max_s = s, max_i = i;
        else if (max_s - s > max_drop) break;
    } while (i >= 0 && t[i] == 0);
    for (i = z[k].y; i >= 0 && i != end_i; i = p[i]) t[i] = 0;
    return max_i;
}

/* =========================================================
   lc_elem_t 必須在 slab_t 之前定義
   ========================================================= */
typedef struct lc_elem_s {
    int32_t y; int64_t i; double pri;
    KRMQ_HEAD(struct lc_elem_s) head;
} lc_elem_t;

#define lc_elem_cmp(a, b) ((a)->y < (b)->y? -1 : (a)->y > (b)->y? 1 : ((a)->i > (b)->i) - ((a)->i < (b)->i))
#define lc_elem_lt2(a, b) ((a)->pri < (b)->pri)
KRMQ_INIT(lc_elem, lc_elem_t, head, lc_elem_cmp, lc_elem_lt2)
KALLOC_POOL_INIT(rmq, lc_elem_t)

/* krmq_clear 不支援 NULL，用 no-op 函數取代 */
static void lc_elem_noop_free(lc_elem_t *p) { (void)p; }

/* slab allocator：定義在 lc_elem_t 之後 */
typedef struct { lc_elem_t *buf; int64_t top, cap; } slab_t;

static inline slab_t slab_create(int64_t cap) {
    slab_t s; s.buf = (lc_elem_t*)malloc(cap * sizeof(lc_elem_t)); s.top = 0; s.cap = cap; return s;
}
static inline lc_elem_t *slab_alloc(slab_t *s) { assert(s->top < s->cap); return &s->buf[s->top++]; }
static inline void slab_reset(slab_t *s) { s->top = 0; }
static inline void slab_reserve(slab_t *s, int64_t cap) {
    if (cap > s->cap) {
        free(s->buf);
        s->buf = (lc_elem_t*)malloc(cap * sizeof(lc_elem_t));
        s->cap = cap;
    }
    s->top = 0;
}
static inline void slab_destroy(slab_t *s) { free(s->buf); s->buf = NULL; s->top = s->cap = 0; }

#ifndef PAR_SORT_MIN_N
#define PAR_SORT_MIN_N 32768
#endif

#ifndef PAR_CHAIN1_SMALL_PARTITION
#define PAR_CHAIN1_SMALL_PARTITION 65536
#endif

/* [OPT-2] fast-path look-back depth：往回看幾個 anchor 搜尋同 diagonal predecessor
   對 primate asm20 高 synteny 資料，depth=8 就夠了；
   可透過 -DDIAG_FAST_PATH_DEPTH=N 在 compile time 調整 */
#ifndef DIAG_FAST_PATH_DEPTH
#define DIAG_FAST_PATH_DEPTH 8
#endif

typedef struct {
    slab_t slab;
    int32_t *t_local;
    int64_t t_cap;
} pc1_workspace_t;

static inline void pc1_workspace_init(pc1_workspace_t *ws)
{
    ws->slab.buf = NULL;
    ws->slab.top = 0;
    ws->slab.cap = 0;
    ws->t_local = NULL;
    ws->t_cap = 0;
}

static inline void pc1_workspace_reserve(pc1_workspace_t *ws, int64_t sz)
{
    int64_t slab_need = sz * 2 + 4;
    slab_reserve(&ws->slab, slab_need);
    if (sz > ws->t_cap) {
        free(ws->t_local);
        ws->t_local = (int32_t*)malloc(sz * sizeof(int32_t));
        ws->t_cap = sz;
    }
    memset(ws->t_local, 0, sz * sizeof(int32_t));
}

static inline void pc1_workspace_destroy(pc1_workspace_t *ws)
{
    slab_destroy(&ws->slab);
    free(ws->t_local);
    ws->t_local = NULL;
    ws->t_cap = 0;
}

static inline void sort_mm128_array(mm128_t *x, int64_t n)
{
    if (n <= 1) return;
#if defined(PAR_SORT)
    if (is_g2g_aln && n >= PAR_SORT_MIN_N) parallel_sort(x, n, num_threads_b2b);
    else radix_sort_128x(x, x + n);
#else
    radix_sort_128x(x, x + n);
#endif
}

static inline int32_t comput_sc_simple(const mm128_t *ai, const mm128_t *aj, float chn_pen_gap, float chn_pen_skip, int32_t *exact, int32_t *width)
{
    int32_t dq = (int32_t)ai->y - (int32_t)aj->y, dr, dd, dg, q_span, sc;
    dr = (int32_t)(ai->x - aj->x);
    *width = dd = dr > dq? dr - dq : dq - dr;
    dg = dr < dq? dr : dq; q_span = aj->y>>32&0xff; sc = q_span < dg? q_span : dg;
    if (exact) *exact = (dd == 0 && dg <= q_span);
    if (dd || dq > q_span) {
        float lin_pen = chn_pen_gap*(float)dd + chn_pen_skip*(float)dg;
        float log_pen = dd >= 1? mg_log2(dd+1) : 0.0f;
        sc -= (int)(lin_pen + .5f*log_pen);
    }
    return sc;
}

/* mg_chain_backtrack */
uint64_t *mg_chain_backtrack(void *km, int64_t n, int32_t *f, int64_t *p, int32_t *v, int32_t *t, int32_t min_cnt, int32_t min_sc, int32_t max_drop, int32_t *n_u_, int32_t *n_v_)
{
    mm128_t *z; uint64_t *u; int64_t i, k, n_z, n_v; int32_t n_u;
    *n_u_ = *n_v_ = 0;
    for (i = 0, n_z = 0; i < n; ++i) if (f[i] >= min_sc) ++n_z;
    if (n_z == 0) return 0;
    z = Kmalloc(km, mm128_t, n_z);
    for (i = 0, k = 0; i < n; ++i) if (f[i] >= min_sc) z[k].x = f[i], z[k++].y = i;
    radix_sort_128x(z, z + n_z);
    memset(t, 0, n * 4);
    for (k = n_z-1, n_v = n_u = 0; k >= 0; --k) {
        if (t[z[k].y] == 0) {
            int64_t n_v0 = n_v, end_i; int32_t sc;
            end_i = mg_chain_bk_end(max_drop, z, f, p, t, k);
            for (i = z[k].y; i != end_i; i = p[i]) ++n_v, t[i] = 1;
            sc = i < 0? z[k].x : (int32_t)z[k].x - f[i];
            if (sc >= min_sc && n_v > n_v0 && n_v-n_v0 >= min_cnt) ++n_u; else n_v = n_v0;
        }
    }
    u = Kmalloc(km, uint64_t, n_u);
    memset(t, 0, n * 4);
    for (k = n_z-1, n_v = n_u = 0; k >= 0; --k) {
        if (t[z[k].y] == 0) {
            int64_t n_v0 = n_v, end_i; int32_t sc;
            end_i = mg_chain_bk_end(max_drop, z, f, p, t, k);
            for (i = z[k].y; i != end_i; i = p[i]) v[n_v++] = i, t[i] = 1;
            sc = i < 0? z[k].x : (int32_t)z[k].x - f[i];
            if (sc >= min_sc && n_v > n_v0 && n_v-n_v0 >= min_cnt) u[n_u++] = (uint64_t)sc<<32|(n_v-n_v0);
            else n_v = n_v0;
        }
    }
    kfree(km, z);
    if (is_g2g_aln) {
#if defined(PAR_CHAIN_1) || defined(PAR_CHAIN_2)
        free(f);
        free(p);
        free(t);
#endif
    }
    assert(n_v < INT32_MAX);
    *n_u_ = n_u, *n_v_ = n_v;
    return u;
}

uint64_t *mg_chain_backtrack_par(void *km, int64_t n, int32_t *f, int64_t *p, int32_t *v, int32_t *t, int32_t min_cnt, int32_t min_sc, int32_t max_drop, int32_t *n_u_, int32_t *n_v_, const mm128_t *a)
{
    uint64_t *u; int64_t i, n_v; int32_t n_u;
    if (is_g2g_aln) {
#if defined(PAR_CHAIN_1) || defined(PAR_CHAIN_2)
        free(t);
#endif
    }
    auto future = std::async(std::launch::async, [&](){
        std::pair<int64_t,int32_t> *pt = (std::pair<int64_t,int32_t>*)malloc(n*sizeof(std::pair<int64_t,int32_t>));
        #pragma omp parallel for num_threads(num_threads_b2b)
        for (int64_t i = 0; i < n; i++) pt[i] = std::make_pair(p[i], 0);
        return pt;
    });
    *n_u_ = *n_v_ = 0;
    int32_t num_chr = 1;
    for (i = 1; i < n; i++) if (a[i-1].x>>32 != a[i].x>>32) num_chr++;
    std::vector<int64_t> st_vec(num_chr,0), end_vec(num_chr,0);
    int32_t chrom_id = 0;
    if (n > 0) {
        st_vec[0] = 0;
        for (i = 1; i < n; i++) {
            if (a[i-1].x>>32 != a[i].x>>32) { end_vec[chrom_id]=i-1; chrom_id++; st_vec[chrom_id]=i; }
        }
        end_vec[chrom_id] = n-1;
    }
    if (num_chr == 1) { st_vec[0]=0; end_vec[0]=n-1; }
    int32_t n_u_global=0; int64_t n_v_global=0;
    std::pair<int64_t,int32_t> *p_t_vec = future.get();
    if (is_g2g_aln) {
#if defined(PAR_CHAIN_1) || defined(PAR_CHAIN_2)
        free(p);
#endif
    }

    int32_t n_threads = num_threads_b2b > 0 ? num_threads_b2b : 1;
    std::vector<std::vector<int32_t>> v_parts(n_threads);
    std::vector<std::vector<uint64_t>> u_parts(n_threads);

    #pragma omp parallel num_threads(n_threads) reduction(+:n_u_global,n_v_global)
    {
        int32_t tid = omp_get_thread_num();
        std::vector<int32_t> &v_vec_global = v_parts[tid];
        std::vector<uint64_t> &u_vec_global = u_parts[tid];

        #pragma omp for schedule(dynamic,1)
        for (int32_t thd = 0; thd < num_chr; thd++) {
            int64_t n_z=0, i=0, k; std::vector<mm128_t> z;
            for (i = st_vec[thd]; i <= end_vec[thd]; ++i)
                if (f[i] >= min_sc) { mm128_t tmp; tmp.x=f[i]; tmp.y=i; z.push_back(tmp); }
            n_z = z.size();
            sort_mm128_array(z.data(), n_z);

            std::vector<int32_t> v_vec; std::vector<uint64_t> u_vec;
            int64_t n_v_local=0; int32_t n_u_local=0;
#if defined(__x86_64__)||defined(_M_X64)||defined(__i386)||defined(_M_IX86)
            int64_t prefetch_dist=16;
#endif
            for (k = n_z-1; k >= 0; --k) {
#if defined(__x86_64__)||defined(_M_X64)||defined(__i386)||defined(_M_IX86)
                if (k-prefetch_dist>=0) _mm_prefetch(reinterpret_cast<const char*>(&p_t_vec[z[k-prefetch_dist].y].second),_MM_HINT_T0);
#endif
                if (p_t_vec[z[k].y].second == 0) {
                    int64_t n_v0=n_v_local, end_i; int32_t sc;
                    end_i = mg_chain_bk_end_par(max_drop,z,f,p_t_vec,z[k].y,k);
                    for (i=z[k].y; i!=end_i; i=p_t_vec[i].first) { ++n_v_local; v_vec.push_back(i); p_t_vec[i].second=1; }
                    sc = i<0? z[k].x : (int32_t)z[k].x-f[i];
                    if (sc>=min_sc && n_v_local>n_v0 && n_v_local-n_v0>=min_cnt) { ++n_u_local; u_vec.push_back((uint64_t)sc<<32|(n_v_local-n_v0)); }
                    else { for (int64_t j=0; j<(n_v_local-n_v0); j++) v_vec.pop_back(); n_v_local=n_v0; }
                }
            }
            n_u_global += n_u_local;
            n_v_global += n_v_local;
            if (!v_vec.empty()) v_vec_global.insert(v_vec_global.end(), v_vec.begin(), v_vec.end());
            if (!u_vec.empty()) u_vec_global.insert(u_vec_global.end(), u_vec.begin(), u_vec.end());
        }
    }

    n_u=n_u_global; n_v=n_v_global;
    free(p_t_vec);
    if (is_g2g_aln) {
#if defined(PAR_CHAIN_1) || defined(PAR_CHAIN_2)
        free(f);
#endif
    }

    int64_t v_off = 0;
    for (int32_t tid = 0; tid < n_threads; ++tid) {
        if (!v_parts[tid].empty()) {
            memcpy(v + v_off, v_parts[tid].data(), v_parts[tid].size() * sizeof(int32_t));
            v_off += (int64_t)v_parts[tid].size();
        }
    }
    assert(v_off == n_v);

    u = Kmalloc(km,uint64_t,n_u);
    int64_t u_off = 0;
    for (int32_t tid = 0; tid < n_threads; ++tid) {
        if (!u_parts[tid].empty()) {
            memcpy(u + u_off, u_parts[tid].data(), u_parts[tid].size() * sizeof(uint64_t));
            u_off += (int64_t)u_parts[tid].size();
        }
    }
    assert(u_off == n_u);

    st_vec.clear(); end_vec.clear();
    assert(n_v < INT32_MAX);
    *n_u_=n_u, *n_v_=n_v;
    return u;
}

static mm128_t *compact_a(void *km, int32_t n_u, uint64_t *u, int32_t n_v, int32_t *v, mm128_t *a)
{
    mm128_t *b, *w; uint64_t *u2; int64_t i, j, k;
    if (is_g2g_aln) {
        b = Kmalloc(km, mm128_t, n_v);
        #if defined(__x86_64__)||defined(_M_X64)||defined(__i386)||defined(_M_IX86)
            const int64_t PREF_DIST=64;
        #endif
        for (i=0, k=0; i<n_u; ++i) {
            int32_t k0=k, ni=(int32_t)u[i];
            for (j=0; j<ni; ++j) {
                int64_t idx=k0+(ni-j-1);
                #if defined(__x86_64__)||defined(_M_X64)||defined(__i386)||defined(_M_IX86)
                    int64_t pidx=idx-PREF_DIST;
                    if (pidx >= k0) _mm_prefetch(reinterpret_cast<const char*>(&a[v[pidx]]),_MM_HINT_T0);
                #endif
                b[k++]=a[v[idx]];
            }
        }
#if defined(PAR_CHAIN_1) || defined(PAR_CHAIN_2)
        free(v);
#else
        kfree(km, v);
#endif
    } else {
        b = Kmalloc(km, mm128_t, n_v);
        for (i=0, k=0; i<n_u; ++i) { int32_t k0=k, ni=(int32_t)u[i]; for (j=0; j<ni; ++j) b[k++]=a[v[k0+(ni-j-1)]]; }
        kfree(km, v);
    }
    w = Kmalloc(km, mm128_t, n_u);
    for (i=k=0; i<n_u; ++i) { w[i].x=b[k].x, w[i].y=(uint64_t)k<<32|i; k+=(int32_t)u[i]; }
    sort_mm128_array(w, n_u);
    u2 = Kmalloc(km, uint64_t, n_u);
    for (i=k=0; i<n_u; ++i) {
        int32_t jj=(int32_t)w[i].y, nn=(int32_t)u[jj];
        u2[i]=u[jj]; memcpy(&a[k],&b[w[i].y>>32],nn*sizeof(mm128_t)); k+=nn;
    }
    memcpy(u,u2,n_u*8); memcpy(b,a,k*sizeof(mm128_t));
    kfree(km,a); kfree(km,w); kfree(km,u2);
    return b;
}

static inline int32_t comput_sc(const mm128_t *ai, const mm128_t *aj, int32_t max_dist_x, int32_t max_dist_y, int32_t bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg)
{
    int32_t dq=(int32_t)ai->y-(int32_t)aj->y, dr, dd, dg, q_span, sc;
    int32_t sidi=(ai->y&MM_SEED_SEG_MASK)>>MM_SEED_SEG_SHIFT;
    int32_t sidj=(aj->y&MM_SEED_SEG_MASK)>>MM_SEED_SEG_SHIFT;
    if (dq<=0||dq>max_dist_x) return INT32_MIN;
    dr=(int32_t)(ai->x-aj->x);
    if (sidi==sidj&&(dr==0||dq>max_dist_y)) return INT32_MIN;
    dd=dr>dq?dr-dq:dq-dr;
    if (sidi==sidj&&dd>bw) return INT32_MIN;
    if (n_seg>1&&!is_cdna&&sidi==sidj&&dr>max_dist_y) return INT32_MIN;
    dg=dr<dq?dr:dq; q_span=aj->y>>32&0xff; sc=q_span<dg?q_span:dg;
    if (dd||dg>q_span) {
        float lin_pen=chn_pen_gap*(float)dd+chn_pen_skip*(float)dg;
        float log_pen=dd>=1?mg_log2(dd+1):0.0f;
        if (is_cdna||sidi!=sidj) {
            if (sidi!=sidj&&dr==0) ++sc;
            else if (dr>dq||sidi!=sidj) sc-=(int)(lin_pen<log_pen?lin_pen:log_pen);
            else sc-=(int)(lin_pen+.5f*log_pen);
        } else sc-=(int)(lin_pen+.5f*log_pen);
    }
    return sc;
}

mm128_t *mg_lchain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
                      int is_cdna, int n_seg, int64_t n, mm128_t *a, int *n_u_, uint64_t **_u, void *km)
{
    int32_t *f,*t,*v,n_u,n_v,mmax_f=0,max_drop=bw; int64_t *p,i,j,max_ii,st=0; uint64_t *u;
    if (_u) *_u=0,*n_u_=0;
    if (n==0||a==0){kfree(km,a);return 0;}
    if (max_dist_x<bw) max_dist_x=bw;
    if (max_dist_y<bw&&!is_cdna) max_dist_y=bw;
    if (is_cdna) max_drop=INT32_MAX;
    p=Kmalloc(km,int64_t,n); f=Kmalloc(km,int32_t,n); v=Kmalloc(km,int32_t,n); t=Kcalloc(km,int32_t,n);
    if (is_splice_sr) {
        for (i=0,max_ii=-1; i<n; ++i) {
            int64_t max_j=-1,end_j; int32_t max_f=a[i].y>>32&0xff,n_skip=0;
            while (st<i&&(a[i].x>>32!=a[st].x>>32||a[i].x>a[st].x+max_dist_x)) ++st;
            if (i-st>max_iter) st=i-max_iter;
            for (j=i-1; j>=st; --j) {
                int32_t sc=comput_sc(&a[i],&a[j],max_dist_x,max_dist_y,bw,chn_pen_gap,chn_pen_skip,is_cdna,n_seg);
                if (sc==INT32_MIN) continue; sc+=f[j];
                if (sc>max_f){max_f=sc,max_j=j;if(n_skip>0)--n_skip;}
                else if (t[j]==(int32_t)i){if(++n_skip>max_skip)break;}
                if (p[j]>=0) t[p[j]]=i;
            }
            end_j=j;
            if (max_ii<0||a[i].x-a[max_ii].x>(int64_t)max_dist_x){int32_t max=INT32_MIN;max_ii=-1;for(j=i-1;j>=st;--j)if(max<f[j])max=f[j],max_ii=j;}
            if (max_ii>=0&&max_ii<end_j){int32_t tmp=comput_sc(&a[i],&a[max_ii],max_dist_x,max_dist_y,bw,chn_pen_gap,chn_pen_skip,is_cdna,n_seg);if(tmp!=INT32_MIN&&max_f<tmp+f[max_ii])max_f=tmp+f[max_ii],max_j=max_ii;}
            f[i]=max_f,p[i]=max_j; v[i]=max_j>=0&&v[max_j]>max_f?v[max_j]:max_f;
            if(max_ii<0||(a[i].x-a[max_ii].x<=(int64_t)max_dist_x&&f[max_ii]<f[i]))max_ii=i;
            if(mmax_f<max_f)mmax_f=max_f;
        }
    } else {
        #if defined(PAR_DP_CHAIN)&&(defined(__AVX512BW__)||defined(__AVX2__))
            int32_t *v_1,*p_1; uint32_t *f_1;
            v_1=Kmalloc(km,int32_t,n); p_1=Kmalloc(km,int32_t,n); f_1=Kmalloc(km,uint32_t,n);
            anchor_t *anchors=(anchor_t*)malloc(n*sizeof(anchor_t));
            for(i=0;i<n;++i){anchors[i].r=a[i].x;anchors[i].q=(int32_t)a[i].y;anchors[i].l=a[i].y>>32&0xff;}
            num_bits_t *anchor_r,*anchor_q,*anchor_l;
            create_SoA_Anchors_32_bit(anchors,n,anchor_r,anchor_q,anchor_l);
            dp_chain obj(max_dist_x,max_dist_y,bw,max_skip,max_iter,min_cnt,min_sc,chn_pen_gap,chn_pen_skip,is_cdna,n_seg);
            obj.mm_dp_vectorized(n,&anchors[0],anchor_r,anchor_q,anchor_l,f_1,p_1,v_1,max_dist_x,max_dist_y,NULL,NULL);
            anchor_r-=16;anchor_q-=16;anchor_l-=16;free(anchor_r);free(anchor_q);free(anchor_l);free(anchors);
            for(i=0;i<n;i++){f[i]=f_1[i];p[i]=p_1[i];v[i]=v_1[i];}
            kfree(km,p_1);kfree(km,f_1);kfree(km,v_1);
        #else
            for (i=0,max_ii=-1;i<n;++i){
                int64_t max_j=-1,end_j;int32_t max_f=a[i].y>>32&0xff,n_skip=0;
                while(st<i&&(a[i].x>>32!=a[st].x>>32||a[i].x>a[st].x+max_dist_x))++st;
                if(i-st>max_iter)st=i-max_iter;
                for(j=i-1;j>=st;--j){
                    int32_t sc=comput_sc(&a[i],&a[j],max_dist_x,max_dist_y,bw,chn_pen_gap,chn_pen_skip,is_cdna,n_seg);
                    if(sc==INT32_MIN)continue;sc+=f[j];
                    if(sc>max_f){max_f=sc,max_j=j;if(n_skip>0)--n_skip;}
                    else if(t[j]==(int32_t)i){if(++n_skip>max_skip)break;}
                    if(p[j]>=0)t[p[j]]=i;
                }
                end_j=j;
                if(max_ii<0||a[i].x-a[max_ii].x>(int64_t)max_dist_x){int32_t max=INT32_MIN;max_ii=-1;for(j=i-1;j>=st;--j)if(max<f[j])max=f[j],max_ii=j;}
                if(max_ii>=0&&max_ii<end_j){int32_t tmp=comput_sc(&a[i],&a[max_ii],max_dist_x,max_dist_y,bw,chn_pen_gap,chn_pen_skip,is_cdna,n_seg);if(tmp!=INT32_MIN&&max_f<tmp+f[max_ii])max_f=tmp+f[max_ii],max_j=max_ii;}
                f[i]=max_f,p[i]=max_j;v[i]=max_j>=0&&v[max_j]>max_f?v[max_j]:max_f;
                if(max_ii<0||(a[i].x-a[max_ii].x<=(int64_t)max_dist_x&&f[max_ii]<f[i]))max_ii=i;
                if(mmax_f<max_f)mmax_f=max_f;
            }
        #endif
    }
    u=mg_chain_backtrack(km,n,f,p,v,t,min_cnt,min_sc,max_drop,&n_u,&n_v);
    *n_u_=n_u,*_u=u; kfree(km,p);kfree(km,f);kfree(km,t);
    if(n_u==0){kfree(km,a);kfree(km,v);return 0;}
    return compact_a(km,n_u,u,n_v,v,a);
}

/* ============================================================
   mg_lchain_rmq_opt  —  OPTIMIZED VERSION
   ============================================================ */
mm128_t *mg_lchain_rmq_opt(int max_dist, int max_dist_inner, int bw, int max_chn_skip, int cap_rmq_size, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
                       int64_t n, mm128_t *a, int *n_u_, uint64_t **_u, void *km, int32_t num_threads)
{
    struct timeval start_chain, end_chain;
    if (is_g2g_aln) {
#ifdef PROFILE
        gettimeofday(&start_chain, NULL);
#endif
    }

#if defined(PAR_CHAIN_2)
    /* PAR_CHAIN_2：原版邏輯 + slab allocator (unchanged) */
    int32_t *f,*t,*v,n_u,n_v,mmax_f=0,max_rmq_size=0,max_drop=bw;
    int64_t *p; uint64_t *u; int64_t i;
    if(_u)*_u=0,*n_u_=0;
    if(n==0||a==0){kfree(km,a);return 0;}
    if(max_dist<bw)max_dist=bw;
    if(max_dist_inner<=0||max_dist_inner>=max_dist)max_dist_inner=0;
    p=(int64_t*)malloc(n*sizeof(int64_t));f=(int32_t*)malloc(n*sizeof(int32_t));
    v=(int32_t*)malloc(n*sizeof(int32_t));t=(int32_t*)malloc(n*sizeof(int32_t));
    #pragma omp parallel for num_threads(num_threads) private(i)
    for(i=0;i<n;i++){p[i]=-1;f[i]=0;v[i]=0;t[i]=0;}
    int64_t chunk_size=n/num_threads;
    int64_t *st_idx=(int64_t*)malloc(num_threads*sizeof(int64_t));
    for(int32_t thd=0;thd<num_threads;thd++){
        int64_t start_id=thd*chunk_size;
        st_idx[thd]=start_id;
        if(thd==0){st_idx[thd]=0;continue;}
        int64_t i0=start_id-1;
        while(i0>=0&&((int32_t)a[start_id].x<=((int32_t)a[i0].x+max_dist))&&(a[start_id].x>>32==a[i0].x>>32)){st_idx[thd]=i0;i0--;}
    }
    int32_t itr=0;
    for(itr=0;itr<num_threads;itr++){
        int32_t sum=0;
        #pragma omp parallel num_threads(num_threads)
        {
            int32_t thd=omp_get_thread_num();
            lc_elem_t *root=0,*root_inner=0;
            int64_t start=thd*chunk_size;
            int64_t end=(thd+1==num_threads)?n:(thd+1)*chunk_size;
            int64_t st=st_idx[thd],st_inner=st_idx[thd],i0=st_idx[thd];
            int64_t count=0; int32_t mmax_f_loc=0,max_rmq_size_loc=0;
            slab_t slab=slab_create((end-start+1)*2+4);
            for(int64_t i=start;i<end;i++){
                if(itr>thd)break;
                int64_t max_j=-1; int32_t max_f=a[i].y>>32&0xff;
                lc_elem_t s,*q,*r,lo,hi;
                if(i0<i&&a[i0].x!=a[i].x){
                    int64_t jj;
                    for(jj=i0;jj<i;++jj){
                        if((int32_t)a[jj].x==(int32_t)a[i].x)break;
                        q=slab_alloc(&slab); q->y=(int32_t)a[jj].y;q->i=jj;
                        q->pri=-(f[jj]+0.5*chn_pen_gap*((int32_t)a[jj].x+(int32_t)a[jj].y));
                        krmq_insert(lc_elem,&root,q,0);
                        if(max_dist_inner>0){r=slab_alloc(&slab);*r=*q;krmq_insert(lc_elem,&root_inner,r,0);}
                    }
                    i0=jj;
                }
                while(st<i&&(a[i].x>>32!=a[st].x>>32||a[i].x>a[st].x+max_dist||krmq_size(head,root)>cap_rmq_size)){
                    s.y=(int32_t)a[st].y;s.i=st;
                    if((q=krmq_find(lc_elem,root,&s,0))!=0)krmq_erase(lc_elem,&root,q,0);
                    ++st;
                }
                if(max_dist_inner>0){
                    while(st_inner<i&&(a[i].x>>32!=a[st_inner].x>>32||a[i].x>a[st_inner].x+max_dist_inner||krmq_size(head,root_inner)>cap_rmq_size)){
                        s.y=(int32_t)a[st_inner].y;s.i=st_inner;
                        if((q=krmq_find(lc_elem,root_inner,&s,0))!=0)krmq_erase(lc_elem,&root_inner,q,0);
                        ++st_inner;
                    }
                }
                lo.i=INT32_MAX;lo.y=(int32_t)a[i].y-max_dist;hi.i=0;hi.y=(int32_t)a[i].y;
                if((q=krmq_rmq(lc_elem,root,&lo,&hi))!=0){
                    int32_t sc,exact,width,n_skip=0; int64_t jj=q->i;
                    sc=f[jj]+comput_sc_simple(&a[i],&a[jj],chn_pen_gap,chn_pen_skip,&exact,&width);
                    if(width<=bw&&sc>max_f)max_f=sc,max_j=jj;
                    if(!exact&&root_inner&&(int32_t)a[i].y>0){
                        lc_elem_t *lo2,*hi2;s.y=(int32_t)a[i].y-1;s.i=n;
                        krmq_interval(lc_elem,root_inner,&s,&lo2,&hi2);
                        if(lo2){
                            const lc_elem_t *q2;int32_t width2;
                            krmq_itr_t(lc_elem) itr2;krmq_itr_find(lc_elem,root_inner,lo2,&itr2);
                            while((q2=krmq_at(&itr2))!=0){
                                if(q2->y<(int32_t)a[i].y-max_dist_inner)break;
                                jj=q2->i;
                                sc=f[jj]+comput_sc_simple(&a[i],&a[jj],chn_pen_gap,chn_pen_skip,0,&width2);
                                if(width2<=bw){
                                    if(sc>max_f){max_f=sc;max_j=jj;if(n_skip>0)--n_skip;}
                                    else if(t[jj]==(int32_t)i){if(++n_skip>max_chn_skip)break;}
                                    if(p[jj]>=0)t[p[jj]]=i;
                                }
                                if(!krmq_itr_prev(lc_elem,&itr2))break;
                            }
                        }
                    }
                }
                assert(max_j<0||(a[max_j].x<a[i].x&&(int32_t)a[max_j].y<(int32_t)a[i].y));
                if(max_f!=f[i]||max_j!=p[i]){
                    count++;f[i]=max_f;p[i]=max_j;
                    v[i]=max_j>=0&&v[max_j]>max_f?v[max_j]:max_f;
                    mmax_f_loc=mmax_f_loc>max_f?mmax_f_loc:max_f;
                    int32_t rs=krmq_size(head,root);
                    max_rmq_size_loc=max_rmq_size_loc>rs?max_rmq_size_loc:rs;
                }
            }
            krmq_clear(lc_elem,&root,lc_elem_noop_free);
            krmq_clear(lc_elem,&root_inner,lc_elem_noop_free);
            slab_destroy(&slab);
            #pragma omp atomic
            sum+=count;
            #pragma omp critical
            {if(mmax_f<mmax_f_loc)mmax_f=mmax_f_loc;if(max_rmq_size<max_rmq_size_loc)max_rmq_size=max_rmq_size_loc;}
        }
        if(sum==0)break;
    }
    itr+=1;
    if(bw!=100000){mean_itr_1_+=(float)itr;max_itr_1=max_itr_1>itr?max_itr_1:itr;}
    else{mean_itr_2_+=(float)itr;max_itr_2=max_itr_2>itr?max_itr_2:itr;}
    count_stages+=1; free(st_idx);

#elif defined(PAR_CHAIN_1)
    /* =========================================================
       PAR_CHAIN_1 — OPTIMIZED with:
         [OPT-1] force_split_partitions  → better load balancing
         [OPT-2] fast-path diagonal pred → skip expensive RMQ
       ========================================================= */
    int32_t *f,*t,*v,n_u,n_v,mmax_f=0,max_rmq_size=0,max_drop=bw;
    int64_t *p; uint64_t *u; int64_t i;
    if(_u)*_u=0,*n_u_=0;
    if(n==0||a==0){kfree(km,a);return 0;}

    int32_t num_chr=1;
    for(i=1;i<n;i++) if(a[i-1].x>>32!=a[i].x>>32)num_chr++;
    int64_t *st_vec=(int64_t*)calloc(num_chr,sizeof(int64_t));
    int64_t *end_vec=(int64_t*)calloc(num_chr,sizeof(int64_t));
    int32_t chrom_id=0;
    if(n>0){
        st_vec[0]=0;
        for(i=1;i<n;i++){if(a[i-1].x>>32!=a[i].x>>32){end_vec[chrom_id]=i-1;chrom_id++;st_vec[chrom_id]=i;}}
        end_vec[chrom_id]=n-1;
    }
    if(num_chr==1){st_vec[0]=0;end_vec[0]=n-1;}

#ifdef GET_DIST
    int64_t max_anchor_chrom=-1;
    for(i=0;i<num_chr;i++) if(end_vec[i]-st_vec[i]+1>max_anchor_chrom)max_anchor_chrom=end_vec[i]-st_vec[i]+1;
    anchor_dist_file<<(double)max_anchor_chrom/(double)n<<"\n";
#endif

    if(max_dist<bw)max_dist=bw;
    if(max_dist_inner<=0||max_dist_inner>=max_dist)max_dist_inner=0;
    p=(int64_t*)malloc(n*sizeof(int64_t));f=(int32_t*)malloc(n*sizeof(int32_t));
    v=(int32_t*)malloc(n*sizeof(int32_t));t=(int32_t*)calloc(n,sizeof(int32_t));
    #pragma omp parallel for num_threads(num_threads) private(i)
    for(i=0;i<n;i++){p[i]=-1;f[i]=0;v[i]=0;}

    int32_t n_sub=0;
    sub_part_t *sub_parts=build_sub_partitions(a,st_vec,end_vec,num_chr,(int64_t)max_dist,&n_sub);

    /* ---- [OPT-1] force-split 超大 partition ----
       目標: 每個 partition ≈ n/(4*T) anchors，確保 dynamic scheduling 有足夠 granularity。
       在切割目標 ±200 anchor 範圍內搜尋 ref gap 最大處作為斷點。 */
    {
        int64_t max_part_len = n / (4 * num_threads);
        if (max_part_len < 100000) max_part_len = 100000;
        force_split_partitions(a, sub_parts, n_sub, max_part_len);
    }

    std::sort(sub_parts,sub_parts+n_sub,[](const sub_part_t &x,const sub_part_t &y){return(x.end-x.start)>(y.end-y.start);});

    int32_t mmax_f_g=0,max_rmq_size_g=0;
    #pragma omp parallel num_threads(num_threads) reduction(max:mmax_f_g,max_rmq_size_g)
    {
        pc1_workspace_t ws;
        pc1_workspace_init(&ws);

        #pragma omp for schedule(dynamic,1) nowait
        for(int32_t sp=0;sp<n_sub;sp++){
            int64_t start=sub_parts[sp].start,end=sub_parts[sp].end,sz=end-start+1;
            pc1_workspace_reserve(&ws, sz);
            slab_t *slab=&ws.slab;
            int32_t *t_local=ws.t_local;
            lc_elem_t *root=0,*root_inner=0;
            int64_t st=start,st_inner=start,i0=start;
            int32_t mmax_f_loc=0,max_rmq_size_loc=0;
            const int32_t local_skip = (sz < PAR_CHAIN1_SMALL_PARTITION)? 0 : max_chn_skip;

            for(int64_t i=start;i<=end;i++){
                int64_t max_j=-1; int32_t max_f=a[i].y>>32&0xff;
                lc_elem_t s,*q,*r,lo,hi;

                /* --- deferred insert into RMQ trees (unchanged) --- */
                if(i0<i&&a[i0].x!=a[i].x){
                    int64_t jj;
                    for(jj=i0;jj<i;++jj){
                        if((int32_t)a[jj].x==(int32_t)a[i].x)break;
                        q=slab_alloc(slab); q->y=(int32_t)a[jj].y;q->i=jj;
                        q->pri=-(f[jj]+0.5*chn_pen_gap*((int32_t)a[jj].x+(int32_t)a[jj].y));
                        krmq_insert(lc_elem,&root,q,0);
                        if(max_dist_inner>0){r=slab_alloc(slab);*r=*q;krmq_insert(lc_elem,&root_inner,r,0);}
                    }
                    i0=jj;
                }

                /* --- prune expired anchors from RMQ trees (unchanged) --- */
                while(st<i&&(a[i].x>>32!=a[st].x>>32||a[i].x>a[st].x+max_dist||krmq_size(head,root)>cap_rmq_size)){
                    s.y=(int32_t)a[st].y;s.i=st;
                    if((q=krmq_find(lc_elem,root,&s,0))!=0)krmq_erase(lc_elem,&root,q,0);
                    ++st;
                }
                if(max_dist_inner>0){
                    while(st_inner<i&&(a[i].x>>32!=a[st_inner].x>>32||a[i].x>a[st_inner].x+max_dist_inner||krmq_size(head,root_inner)>cap_rmq_size)){
                        s.y=(int32_t)a[st_inner].y;s.i=st_inner;
                        if((q=krmq_find(lc_elem,root_inner,&s,0))!=0)krmq_erase(lc_elem,&root_inner,q,0);
                        ++st_inner;
                    }
                }

                /* =============================================
                   [OPT-2] Fast-path diagonal predecessor check.

                   在高 synteny 的 g2g 資料中（如 primate asm20），大多數
                   anchor 的最佳 predecessor 是同一 diagonal 上的前一個 anchor。
                   掃最近 DIAG_FAST_PATH_DEPTH 個 anchor，找到 exact match
                   （same diagonal + within q_span）就跳過 RMQ query。

                   為什麼安全：
                   - insert/prune 仍然照常執行，RMQ tree 狀態一致
                   - 只跳過 query 步驟
                   - exact match 意味著 score = f[j] + min(q_span, dg)，
                     是該 diagonal 上的最大可能分數
                   ============================================= */
                bool skip_rmq = false;
                {
                    int32_t diag_i = (int32_t)a[i].x - (int32_t)a[i].y;
                    int64_t fp_lo = i - DIAG_FAST_PATH_DEPTH;
                    if (fp_lo < start) fp_lo = start;
                    for (int64_t fp_j = i - 1; fp_j >= fp_lo; --fp_j) {
                        int32_t diag_j = (int32_t)a[fp_j].x - (int32_t)a[fp_j].y;
                        if (diag_j == diag_i) {
                            int32_t exact_fp = 0, width_fp = 0;
                            int32_t sc_fp = f[fp_j] + comput_sc_simple(&a[i], &a[fp_j],
                                                        chn_pen_gap, chn_pen_skip,
                                                        &exact_fp, &width_fp);
                            if (exact_fp && width_fp <= bw && sc_fp > max_f) {
                                max_f = sc_fp;
                                max_j = fp_j;
                                skip_rmq = true;
                            }
                            break;  /* 同 diagonal 只看最近一個 */
                        }
                    }
                }

                /* --- RMQ query (skipped when fast-path hit exact match) --- */
                if (!skip_rmq) {
                    lo.i=INT32_MAX;lo.y=(int32_t)a[i].y-max_dist;hi.i=0;hi.y=(int32_t)a[i].y;
                    if((q=krmq_rmq(lc_elem,root,&lo,&hi))!=0){
                        int32_t sc,exact,width,n_skip=0; int64_t jj=q->i;
                        assert(q->y>=lo.y&&q->y<=hi.y);
                        sc=f[jj]+comput_sc_simple(&a[i],&a[jj],chn_pen_gap,chn_pen_skip,&exact,&width);
                        if(width<=bw&&sc>max_f)max_f=sc,max_j=jj;
                        if(!exact&&root_inner&&(int32_t)a[i].y>0){
                            lc_elem_t *lo2,*hi2;s.y=(int32_t)a[i].y-1;s.i=n;
                            krmq_interval(lc_elem,root_inner,&s,&lo2,&hi2);
                            if(lo2){
                                const lc_elem_t *q2;int32_t width2;
                                krmq_itr_t(lc_elem) itr2;krmq_itr_find(lc_elem,root_inner,lo2,&itr2);
                                while((q2=krmq_at(&itr2))!=0){
                                    if(q2->y<(int32_t)a[i].y-max_dist_inner)break;
                                    jj=q2->i;
                                    sc=f[jj]+comput_sc_simple(&a[i],&a[jj],chn_pen_gap,chn_pen_skip,0,&width2);
                                    if(width2<=bw){
                                        if(sc>max_f){max_f=sc;max_j=jj;if(n_skip>0)--n_skip;}
                                        else if(local_skip>0){
                                            int64_t jj_loc=jj-start;
                                            if(jj_loc>=0&&jj_loc<sz&&t_local[jj_loc]==(int32_t)(i-start)){if(++n_skip>local_skip)break;}
                                        }
                                        if(local_skip>0 && p[jj]>=start&&p[jj]<=end)t_local[p[jj]-start]=(int32_t)(i-start);
                                    }
                                    if(!krmq_itr_prev(lc_elem,&itr2))break;
                                }
                            }
                        }
                    }
                } /* end if (!skip_rmq) */

                assert(max_j<0||(a[max_j].x<a[i].x&&(int32_t)a[max_j].y<(int32_t)a[i].y));
                if(max_f!=f[i]||max_j!=p[i]){
                    f[i]=max_f;p[i]=max_j;v[i]=max_j>=0&&v[max_j]>max_f?v[max_j]:max_f;
                    mmax_f_loc=mmax_f_loc>max_f?mmax_f_loc:max_f;
                    int32_t rs=krmq_size(head,root);
                    max_rmq_size_loc=max_rmq_size_loc>rs?max_rmq_size_loc:rs;
                }
            }
            krmq_clear(lc_elem,&root,lc_elem_noop_free);
            krmq_clear(lc_elem,&root_inner,lc_elem_noop_free);
            mmax_f_g=mmax_f_g>mmax_f_loc?mmax_f_g:mmax_f_loc;
            max_rmq_size_g=max_rmq_size_g>max_rmq_size_loc?max_rmq_size_g:max_rmq_size_loc;
        }

        pc1_workspace_destroy(&ws);
    }
    mmax_f=mmax_f_g; max_rmq_size=max_rmq_size_g;
    free(sub_parts); free(st_vec); free(end_vec);

#else
    /* Serial baseline（原版）*/
    int32_t *f,*t,*v,n_u,n_v,mmax_f=0,max_rmq_size=0,max_drop=bw;
    int64_t *p,i,i0,st=0,st_inner=0; uint64_t *u;
    lc_elem_t *root=0,*root_inner=0; void *mem_mp=0; kmp_rmq_t *mp;
    if(_u)*_u=0,*n_u_=0;
    if(n==0||a==0){kfree(km,a);return 0;}
    if(max_dist<bw)max_dist=bw;
    if(max_dist_inner<0)max_dist_inner=0;
    if(max_dist_inner>max_dist)max_dist_inner=max_dist;
    p=Kmalloc(km,int64_t,n);f=Kmalloc(km,int32_t,n);t=Kcalloc(km,int32_t,n);v=Kmalloc(km,int32_t,n);
    mem_mp=km_init2(km,0x10000);mp=kmp_init_rmq(mem_mp);
    for(i=i0=0;i<n;++i){
        int64_t max_j=-1;int32_t q_span=a[i].y>>32&0xff,max_f=q_span;
        lc_elem_t s,*q,*r,lo,hi;
        if(i0<i&&a[i0].x!=a[i].x){
            int64_t j;
            for(j=i0;j<i;++j){
                q=kmp_alloc_rmq(mp);q->y=(int32_t)a[j].y;q->i=j;
                q->pri=-(f[j]+0.5*chn_pen_gap*((int32_t)a[j].x+(int32_t)a[j].y));
                krmq_insert(lc_elem,&root,q,0);
                if(max_dist_inner>0){r=kmp_alloc_rmq(mp);*r=*q;krmq_insert(lc_elem,&root_inner,r,0);}
            }
            i0=i;
        }
        while(st<i&&(a[i].x>>32!=a[st].x>>32||a[i].x>a[st].x+max_dist||krmq_size(head,root)>cap_rmq_size)){
            s.y=(int32_t)a[st].y;s.i=st;
            if((q=krmq_find(lc_elem,root,&s,0))!=0){q=krmq_erase(lc_elem,&root,q,0);kmp_free_rmq(mp,q);}
            ++st;
        }
        if(max_dist_inner>0){
            while(st_inner<i&&(a[i].x>>32!=a[st_inner].x>>32||a[i].x>a[st_inner].x+max_dist_inner||krmq_size(head,root_inner)>cap_rmq_size)){
                s.y=(int32_t)a[st_inner].y;s.i=st_inner;
                if((q=krmq_find(lc_elem,root_inner,&s,0))!=0){q=krmq_erase(lc_elem,&root_inner,q,0);kmp_free_rmq(mp,q);}
                ++st_inner;
            }
        }
        lo.i=INT32_MAX;lo.y=(int32_t)a[i].y-max_dist;hi.i=0;hi.y=(int32_t)a[i].y;
        if((q=krmq_rmq(lc_elem,root,&lo,&hi))!=0){
            int32_t sc,exact,width,n_skip=0;int64_t j=q->i;
            assert(q->y>=lo.y&&q->y<=hi.y);
            sc=f[j]+comput_sc_simple(&a[i],&a[j],chn_pen_gap,chn_pen_skip,&exact,&width);
            if(width<=bw&&sc>max_f)max_f=sc,max_j=j;
            if(!exact&&root_inner&&(int32_t)a[i].y>0){
                lc_elem_t *lo2,*hi2;s.y=(int32_t)a[i].y-1;s.i=n;
                krmq_interval(lc_elem,root_inner,&s,&lo2,&hi2);
                if(lo2){
                    const lc_elem_t *q2;int32_t width2;
                    krmq_itr_t(lc_elem) itr2;krmq_itr_find(lc_elem,root_inner,lo2,&itr2);
                    while((q2=krmq_at(&itr2))!=0){
                        if(q2->y<(int32_t)a[i].y-max_dist_inner)break;
                        j=q2->i;
                        sc=f[j]+comput_sc_simple(&a[i],&a[j],chn_pen_gap,chn_pen_skip,0,&width2);
                        if(width2<=bw){
                            if(sc>max_f){max_f=sc;max_j=j;if(n_skip>0)--n_skip;}
                            else if(t[j]==(int32_t)i){if(++n_skip>max_chn_skip)break;}
                            if(p[j]>=0)t[p[j]]=i;
                        }
                        if(!krmq_itr_prev(lc_elem,&itr2))break;
                    }
                }
            }
        }
        assert(max_j<0||(a[max_j].x<a[i].x&&(int32_t)a[max_j].y<(int32_t)a[i].y));
        f[i]=max_f;p[i]=max_j;v[i]=max_j>=0&&v[max_j]>max_f?v[max_j]:max_f;
        if(mmax_f<max_f)mmax_f=max_f;
        if(max_rmq_size<krmq_size(head,root))max_rmq_size=krmq_size(head,root);
    }
    km_destroy(mem_mp);
#endif

    /* Backtracking */
    if(max_dist==100000){
        struct timeval start_btk,end_btk;double time=0.0;
        if (is_g2g_aln) {
#ifdef PROFILE
            gettimeofday(&end_chain, NULL);
            time = (end_chain.tv_sec - start_chain.tv_sec) + (end_chain.tv_usec - start_chain.tv_usec) / 1e6;
            rmq_2_time_ = rmq_2_time_ > time ? rmq_2_time_ : time;
            gettimeofday(&start_btk, NULL);
#endif
        }
        #if defined(PAR_BTK)
            u=mg_chain_backtrack_par(km,n,f,p,v,t,min_cnt,min_sc,max_drop,&n_u,&n_v,a);
        #else
            u=mg_chain_backtrack(km,n,f,p,v,t,min_cnt,min_sc,max_drop,&n_u,&n_v);
        #endif
        if (is_g2g_aln) {
#ifdef PROFILE
            gettimeofday(&end_btk, NULL);
            time = (end_btk.tv_sec - start_btk.tv_sec) + (end_btk.tv_usec - start_btk.tv_usec) / 1e6;
            btk_2_time_ = btk_2_time_ > time ? btk_2_time_ : time;
#endif
        }
    }else{
        struct timeval start_btk,end_btk;double time=0.0;
        if (is_g2g_aln) {
#ifdef PROFILE
            gettimeofday(&end_chain, NULL);
            time = (end_chain.tv_sec - start_chain.tv_sec) + (end_chain.tv_usec - start_chain.tv_usec) / 1e6;
            rmq_1_time_ = rmq_1_time_ > time ? rmq_1_time_ : time;
            gettimeofday(&start_btk, NULL);
#endif
        }
        #if defined(PAR_BTK)
            u=mg_chain_backtrack_par(km,n,f,p,v,t,min_cnt,min_sc,max_drop,&n_u,&n_v,a);
        #else
            u=mg_chain_backtrack(km,n,f,p,v,t,min_cnt,min_sc,max_drop,&n_u,&n_v);
        #endif
        if (is_g2g_aln) {
#ifdef PROFILE
            gettimeofday(&end_btk, NULL);
            time = (end_btk.tv_sec - start_btk.tv_sec) + (end_btk.tv_usec - start_btk.tv_usec) / 1e6;
            btk_1_time_ = btk_1_time_ > time ? btk_1_time_ : time;
#endif
        }
    }
    #if defined(PAR_CHAIN_1)||defined(PAR_CHAIN_2)
    #else
        kfree(km,p);kfree(km,f);kfree(km,t);
    #endif
    if (bw == 100000) {
#ifdef GET_DIST
        count_chains_file << n_u << "\n";
#endif
    }
    *n_u_=n_u;*_u=u;
    if (n_u == 0) {
        kfree(km, a);
#if defined(PAR_CHAIN_1) || defined(PAR_CHAIN_2)
        free(v);
#else
        kfree(km, v);
#endif
        return 0;
    }
    return compact_a(km,n_u,u,n_v,v,a);
}

mm128_t *mg_lchain_rmq_mm2(int max_dist, int max_dist_inner, int bw, int max_chn_skip, int cap_rmq_size, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
                       int64_t n, mm128_t *a, int *n_u_, uint64_t **_u, void *km, int32_t temp)
{
    int32_t *f,*t,*v,n_u,n_v,mmax_f=0,max_rmq_size=0,max_drop=bw;
    int64_t *p,i,i0,st=0,st_inner=0; uint64_t *u;
    lc_elem_t *root=0,*root_inner=0; void *mem_mp=0; kmp_rmq_t *mp;
    if(_u)*_u=0,*n_u_=0;
    if(n==0||a==0){kfree(km,a);return 0;}
    if(max_dist<bw)max_dist=bw;
    if(max_dist_inner<0)max_dist_inner=0;
    if(max_dist_inner>max_dist)max_dist_inner=max_dist;
    p=Kmalloc(km,int64_t,n);f=Kmalloc(km,int32_t,n);t=Kcalloc(km,int32_t,n);v=Kmalloc(km,int32_t,n);
    mem_mp=km_init2(km,0x10000);mp=kmp_init_rmq(mem_mp);
    for(i=i0=0;i<n;++i){
        int64_t max_j=-1;int32_t q_span=a[i].y>>32&0xff,max_f=q_span;
        lc_elem_t s,*q,*r,lo,hi;
        if(i0<i&&a[i0].x!=a[i].x){
            int64_t j;
            for(j=i0;j<i;++j){
                q=kmp_alloc_rmq(mp);q->y=(int32_t)a[j].y;q->i=j;
                q->pri=-(f[j]+0.5*chn_pen_gap*((int32_t)a[j].x+(int32_t)a[j].y));
                krmq_insert(lc_elem,&root,q,0);
                if(max_dist_inner>0){r=kmp_alloc_rmq(mp);*r=*q;krmq_insert(lc_elem,&root_inner,r,0);}
            }
            i0=i;
        }
        while(st<i&&(a[i].x>>32!=a[st].x>>32||a[i].x>a[st].x+max_dist||krmq_size(head,root)>cap_rmq_size)){
            s.y=(int32_t)a[st].y;s.i=st;if((q=krmq_find(lc_elem,root,&s,0))!=0){q=krmq_erase(lc_elem,&root,q,0);kmp_free_rmq(mp,q);}++st;
        }
        if(max_dist_inner>0){
            while(st_inner<i&&(a[i].x>>32!=a[st_inner].x>>32||a[i].x>a[st_inner].x+max_dist_inner||krmq_size(head,root_inner)>cap_rmq_size)){
                s.y=(int32_t)a[st_inner].y;s.i=st_inner;if((q=krmq_find(lc_elem,root_inner,&s,0))!=0){q=krmq_erase(lc_elem,&root_inner,q,0);kmp_free_rmq(mp,q);}++st_inner;
            }
        }
        lo.i=INT32_MAX;lo.y=(int32_t)a[i].y-max_dist;hi.i=0;hi.y=(int32_t)a[i].y;
        if((q=krmq_rmq(lc_elem,root,&lo,&hi))!=0){
            int32_t sc,exact,width,n_skip=0;int64_t j=q->i;
            assert(q->y>=lo.y&&q->y<=hi.y);
            sc=f[j]+comput_sc_simple(&a[i],&a[j],chn_pen_gap,chn_pen_skip,&exact,&width);
            if(width<=bw&&sc>max_f)max_f=sc,max_j=j;
            if(!exact&&root_inner&&(int32_t)a[i].y>0){
                lc_elem_t *lo2,*hi2;s.y=(int32_t)a[i].y-1;s.i=n;
                krmq_interval(lc_elem,root_inner,&s,&lo2,&hi2);
                if(lo2){
                    const lc_elem_t *q2;int32_t width2;
                    krmq_itr_t(lc_elem) itr2;krmq_itr_find(lc_elem,root_inner,lo2,&itr2);
                    while((q2=krmq_at(&itr2))!=0){
                        if(q2->y<(int32_t)a[i].y-max_dist_inner)break;
                        j=q2->i;sc=f[j]+comput_sc_simple(&a[i],&a[j],chn_pen_gap,chn_pen_skip,0,&width2);
                        if(width2<=bw){
                            if(sc>max_f){max_f=sc;max_j=j;if(n_skip>0)--n_skip;}
                            else if(t[j]==(int32_t)i){if(++n_skip>max_chn_skip)break;}
                            if(p[j]>=0)t[p[j]]=i;
                        }
                        if(!krmq_itr_prev(lc_elem,&itr2))break;
                    }
                }
            }
        }
        assert(max_j<0||(a[max_j].x<a[i].x&&(int32_t)a[max_j].y<(int32_t)a[i].y));
        f[i]=max_f;p[i]=max_j;v[i]=max_j>=0&&v[max_j]>max_f?v[max_j]:max_f;
        if(mmax_f<max_f)mmax_f=max_f;
        if(max_rmq_size<krmq_size(head,root))max_rmq_size=krmq_size(head,root);
    }
    km_destroy(mem_mp);
    u=mg_chain_backtrack(km,n,f,p,v,t,min_cnt,min_sc,max_drop,&n_u,&n_v);
    *n_u_=n_u;*_u=u;kfree(km,p);kfree(km,f);kfree(km,t);
    if(n_u==0){kfree(km,a);kfree(km,v);return 0;}
    return compact_a(km,n_u,u,n_v,v,a);
}

mm128_t *mg_lchain_rmq(int max_dist, int max_dist_inner, int bw, int max_chn_skip, int cap_rmq_size, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
                       int64_t n, mm128_t *a, int *n_u_, uint64_t **_u, void *km, int32_t num_threads)
{
    if(is_g2g_aln)
        return mg_lchain_rmq_opt(max_dist,max_dist_inner,bw,max_chn_skip,cap_rmq_size,min_cnt,min_sc,chn_pen_gap,chn_pen_skip,n,a,n_u_,_u,km,num_threads);
    else
        return mg_lchain_rmq_mm2(max_dist,max_dist_inner,bw,max_chn_skip,cap_rmq_size,min_cnt,min_sc,chn_pen_gap,chn_pen_skip,n,a,n_u_,_u,km,num_threads);
}
