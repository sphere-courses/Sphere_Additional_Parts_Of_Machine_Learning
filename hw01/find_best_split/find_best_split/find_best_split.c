#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define SPEEDUP

#if !defined(SPEEDUP)

double evaluate_antiloss(
	double * S_p, int32_t * n_p,
	double * S_n_1, int32_t * n_n_1,
	int32_t n_node
)
{
	double antiloss = 0.;
	for (int32_t j = 0; j < n_node; ++j) {
		antiloss += (n_p[j] ? S_p[j] * S_p[j] / n_p[j] : 0.) + ((n_n_1[j] - n_p[j]) ? (S_n_1[j] - S_p[j]) * (S_n_1[j] - S_p[j]) / (n_n_1[j] - n_p[j]) : 0.);
	}
	return antiloss;
}
#else

double evaluate_antiloss(
	double * restrict _S_p, int32_t * restrict _n_p,
	double * restrict _S_n_1, int32_t * restrict _n_n_1,
	int32_t n_node
)
{
	double *S_p = __builtin_assume_aligned(_S_p, 64);
	int *n_p = __builtin_assume_aligned(_n_p, 32);
	double *S_n_1 = __builtin_assume_aligned(_S_n_1, 64);
	int *n_n_1 = __builtin_assume_aligned(_n_n_1, 32);

	double antiloss = 0.;
	for (int32_t j = 0; j < n_node; ++j) {
		antiloss += (n_p[j] ? S_p[j] * S_p[j] / n_p[j] : 0.) + ((n_n_1[j] - n_p[j]) ? (S_n_1[j] - S_p[j]) * (S_n_1[j] - S_p[j]) / (n_n_1[j] - n_p[j]) : 0.);
	}
	return antiloss;
}

#endif // SPEEDUP

// x_*, y_*, idx_*, node_* must be feature-along-horizon c-style arrays
double* find_best_split(
	double *x_sliced_sorted, double *y_sliced_sorted,
	int32_t *idx_sliced_sorted, int32_t *node_idx,
	int32_t *allowed_fea, int32_t allowed_fea_len,
	int32_t n_obj, int32_t n_fea, int32_t depth
)
{
	// node_idxs in range(2**depth - 1, 2**(depth + 1) - 2) therefore node_idx - (2**depth - 1) in range(0, 2 ** depth - 1)
	int32_t node_bias = (1 << depth) - 1;
	// n_node = 2 ** depth
	int32_t n_node = (1 << depth);
	// alloc memory for result (split_fea, split_thr), this memory should be freed outside the function' scope
	double *result = (double *)malloc(sizeof(double) * 2);

	// alloc memory for additional arrays (S_l,p, S_l,N-1, n_i, n_i,p)
	double *S_n_1 = (double *)malloc(sizeof(double) * n_node);
	double *S_p = (double *)malloc(sizeof(double) * n_node);
	int32_t *n_n_1 = (int32_t *)malloc(sizeof(int32_t) * n_node);
	int32_t *n_p = (int32_t *)malloc(sizeof(int32_t) * n_node);

	// initialize S_n_1, n_n_1 with zeros
	for (int32_t p = 0; p < n_node; ++p) {
		S_n_1[p] = n_n_1[p] = 0;
	}

	// compute S_n_1, n_n_1
	for (int32_t i = 0; i < n_obj; ++i) {
		S_n_1[node_idx[idx_sliced_sorted[i]] - node_bias] += y_sliced_sorted[i];
		n_n_1[node_idx[idx_sliced_sorted[i]] - node_bias] += 1;
	}

	int32_t best_fea = -1;
	double best_antiloss = 0., antiloss = 0., best_thr = 0.;
	for(int32_t k, r = 0; r < allowed_fea_len; ++r){
		k = allowed_fea[r];

		// initialize S_p , n_p with zeros
		for (int32_t p = 0; p < n_node; ++p) {
			S_p[p] = n_p[p] = 0;
		}

		// compute split_thr
		for (int32_t i = 0; i < n_obj - 1; ++i) {
			S_p[node_idx[idx_sliced_sorted[n_obj * k + i]] - node_bias] += y_sliced_sorted[n_obj * k + i];
			n_p[node_idx[idx_sliced_sorted[n_obj * k + i]] - node_bias] += 1;
			antiloss = evaluate_antiloss(S_p, n_p, S_n_1, n_n_1, n_node);

			if (fabs(x_sliced_sorted[n_obj * k + i] - x_sliced_sorted[n_obj * k + i + 1]) > 2.22e-16 &&
				antiloss > best_antiloss) {
				best_antiloss = antiloss;
				best_fea = k;
				best_thr = x_sliced_sorted[n_obj * k + i];
			}
		}
	}

	free(S_n_1);
	free(S_p);
	free(n_n_1);
	free(n_p);

	result[0] = best_fea;
	result[1] = best_thr;
	return result;
}