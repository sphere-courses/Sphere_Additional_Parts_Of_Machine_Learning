#include "pch.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>


double evaluate_node_antiloss(double S_p, int n_p, double S_n_1, int n_n_1)
{
	double result = 0.;
	if (!n_n_1) {
		return result;
	}
	if (n_p) {
		result += S_p * S_p / n_p;
	}
	if (n_n_1 - n_p) {
		result += (S_n_1 - S_p) * (S_n_1 - S_p) / (n_n_1 - n_p);
	}
	return result;
}

// x_*, y_*, idx_*, node_* must be feature-along-horizon c-style arrays
double* find_best_split(double *x_sliced_sorted, double *y_sliced_sorted, int *idx_sliced_sorted, int *node_idx,
	int n_obj, int n_fea, int depth
)
{
	// node_idxs in range(2**depth - 1, 2**(depth + 1) - 2) threfore node_idx - (2**depth - 1) in range(0, 2 ** depth - 1)
	int node_bias = (1 << depth) - 1;
	// n_node = 2 ** depth
	int n_node = (1 << depth);
	// alloc memory for result (split_fea, split_thr), this memory should be freed outside the function' scope
	double *result = (double *)malloc(sizeof(double) * 2);

	// alloc memory for additional arrays (S_l,p, S_l,N-1, n_i, n_i,p)
	double *S_n_1 = (double *)malloc(sizeof(double) * n_node);
	double *S_p = (double *)malloc(sizeof(double) * n_node);
	int *n_n_1 = (int *)malloc(sizeof(int) * n_node);
	int *n_p = (int *)malloc(sizeof(int) * n_node);

	// initialize S_n_1, n_n_1 with zeros
	for (int p = 0; p < n_node; ++p) {
		S_n_1[p] = n_n_1[p] = 0;
	}
	// compute S_n_1, n_n_1
	for (int i = 0; i < n_obj; ++i) {
		S_n_1[node_idx[idx_sliced_sorted[i]] - node_bias] += y_sliced_sorted[i];
		n_n_1[node_idx[idx_sliced_sorted[i]] - node_bias] += 1;
	}

	int best_fea = -1;
	double best_antiloss = 0., antiloss = 0., best_thr = 0.;
	for (int k = 0; k < n_fea; ++k) {
		// initialize S_p , n_p with zeros
		for (int p = 0; p < n_node; ++p) {
			S_p[p] = n_p[p] = 0;
		}

		// compute split_thr
		for (int i = 0; i < n_obj - 1; ++i) {
			S_p[node_idx[idx_sliced_sorted[n_obj * k + i]] - node_bias] += y_sliced_sorted[n_obj * k + i];
			n_p[node_idx[idx_sliced_sorted[n_obj * k + i]] - node_bias] += 1;
			antiloss = 0.;
			for (int j = 0; j < n_node; ++j) {
				antiloss += evaluate_node_antiloss(S_p[j], n_p[j], S_n_1[j], n_n_1[j]);
			}
			if (abs(x_sliced_sorted[n_obj * k + i] - x_sliced_sorted[n_obj * k + i + 1]) > 1e-5 &&
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

int main() 
{
	double x_s[] = {
		1, 1, 2, 4,
		1, 2, 4, 4,
		1, 2, 2, 4
	}, y_s[] = {
		4, 1, 1, 2,
		1, 2, 1, 4,
		1, 4, 1, 2
	};
	int idx_s[] = {
		1, 3, 0, 2,
		3, 2, 0, 1,
		0, 1, 3, 2
	}, n_idx[] = {
		0, 0, 0, 0
	};
	double *res = find_best_split(x_s, y_s, idx_s, n_idx, 4, 3, 0);
	std::cout << res[0] << ' ' << res[1] << std::endl;
	return 0;
}