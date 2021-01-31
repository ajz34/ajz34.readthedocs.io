#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <memory.h>
#include <math.h>


// C bool
typedef enum {
    true=1, false=0
} bool;

inline void update_min(float* p1, float v2) {
    if (v2 < *p1) *p1 = v2;
}

// https://stackoverflow.com/questions/28258590/using-openmp-to-get-the-index-of-minimum-element-parallelly
struct Compare { float val; size_t index; };
#pragma omp declare reduction(maximum : struct Compare : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out)

void kennard_stone(float* cdist, size_t* seed, size_t* result, float* v_dist, size_t n_sample, size_t n_seed, size_t n_result) {
    // 00. Assertions and Result Vector Initialization
    struct Compare sup;
    if (n_seed == 2) v_dist[0] = cdist[seed[0] * n_sample + seed[1]];
    if (n_seed == 0) {
        size_t n_sample_2 = n_sample * n_sample;
        sup.val = -1.;
        sup.index = 0;
        #pragma omp parallel for reduction(maximum:sup)
        for (size_t i = 0; i < n_sample_2; ++i) {
            if (cdist[i] > sup.val) {
                sup.val = cdist[i];
                sup.index = i;
            }
        }
        seed[0] = sup.index / n_sample;
        seed[1] = sup.index % n_sample;
        n_seed = 2;
        v_dist[0] = sup.val;
    }
    n_result = n_result == 0 ? n_sample : n_result;
    assert(n_result <= n_sample);
    assert(n_seed <= n_sample);
    memcpy(result, seed, n_seed * sizeof(size_t));
    memset(result + n_seed, 0, (n_result - n_seed) * sizeof(size_t));
    // 01. Scratch Area Initialization
    bool* selected = (bool*)malloc(n_sample * sizeof(bool));
    memset(selected, false, n_sample * sizeof(bool));
    for (size_t i = 0; i < n_seed; ++i)
        selected[result[i]] = true;
    // 02. Minimum Out-of-Group Initialization
    float* min_vals = (float*)malloc(n_sample * sizeof(float));
    memcpy(min_vals, cdist + n_sample * result[0], n_sample * sizeof(float));
    for (size_t n = 1; n < n_seed; ++n) {
        size_t idx_starting = result[n] * n_sample;
    	#pragma omp parallel for
        for (size_t i = 0; i < n_sample; ++i) {
            if (selected[i]) continue;
            update_min(&min_vals[i], cdist[idx_starting + i]);
        }
    }
    // 03. Main Algorithm
    for (size_t n = n_seed; n < n_result; ++n) {
        // Find sup of the minimum
        sup.val = -1.;
        sup.index = 0;
        #pragma omp parallel for reduction(maximum:sup)
        for (size_t i = 0; i < n_sample; ++i) {
            if (selected[i]) continue;
            if (min_vals[i] > sup.val) {
                sup.index = i;
                sup.val = min_vals[i];
            }
        }
        v_dist[n - 1] = sup.val;
        selected[sup.index] = true;
        result[n] = sup.index;
        size_t idx_starting = sup.index * n_sample;
        #pragma omp parallel for
        for (size_t i = 0; i < n_sample; ++i) {
            if (selected[i]) continue;
            update_min(&min_vals[i], cdist[idx_starting + i]);
        }
    }
    free(selected);
    free(min_vals);
}

float euclid_distance_vector(float* x1, float* x2, size_t n_feature) {
    float res = 0.;
    do {
    	res += (*x1 - *x2) * (*x1 - *x2);
        ++x1, ++x2;
    } while (--n_feature);
    return sqrtf(res);
}

void kennard_stone_mem(float* X, size_t* seed, size_t* result, float* v_dist, size_t n_sample, size_t n_feature, size_t n_seed, size_t n_result) {
    // 00. Assertions and Result Vector Initialization
    struct Compare sup;
    if (n_seed == 2) v_dist[0] = euclid_distance_vector(X + n_feature * seed[0], X + n_feature * seed[1], n_feature);
    assert(n_seed != 0);           // Seed should be supplied from outer program.
    assert(n_result <= n_sample);
    assert(n_seed <= n_sample);
    memcpy(result, seed, n_seed * sizeof(size_t));
    memset(result + n_seed, 0, (n_result - n_seed) * sizeof(size_t));
    // 01. Scratch Area Initialization
    bool* selected = (bool*)malloc(n_sample * sizeof(bool));
    memset(selected, false, n_sample * sizeof(bool));
    for (size_t i = 0; i < n_seed; ++i)
        selected[result[i]] = true;
    // 02. Minimum Out-of-Group Initialization
    float* min_vals = (float*)malloc(n_sample * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < n_sample; ++i) {
        if (selected[i]) continue;
        min_vals[i] = euclid_distance_vector(X + n_feature * result[0], X + n_feature * i, n_feature);
    }
    for (size_t n = 1; n < n_seed; ++n) {
        float* p_starting = X + result[n] * n_feature;
    	#pragma omp parallel for
        for (size_t i = 0; i < n_sample; ++i) {
            if (selected[i]) continue;
            update_min(&min_vals[i], euclid_distance_vector(p_starting, X + n_feature * i, n_feature));
        }
    }
    // 03. Main Algorithm
    for (size_t n = n_seed; n < n_result; ++n) {
        // Find sup of the minimum
        sup.val = -1.;
        sup.index = 0;
        #pragma omp parallel for reduction(maximum:sup)
        for (size_t i = 0; i < n_sample; ++i) {
            if (selected[i]) continue;
            if (min_vals[i] > sup.val) {
                sup.index = i;
                sup.val = min_vals[i];
            }
        }
        v_dist[n - 1] = sup.val;
        selected[sup.index] = true;
        result[n] = sup.index;
        float* p_starting = X + sup.index * n_feature;
        #pragma omp parallel for
        for (size_t i = 0; i < n_sample; ++i) {
            if (selected[i]) continue;
            update_min(&min_vals[i], euclid_distance_vector(p_starting, X + n_feature * i, n_feature));
        }
    }
    free(selected);
    free(min_vals);
}
