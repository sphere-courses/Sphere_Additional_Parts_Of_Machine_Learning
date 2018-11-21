#pragma GCC optimize("Ofast") 
#pragma optimize("unroll-loops", "agressive-loop-optimizations")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#include <stdlib.h>

double* predict(double *x, double w_0, double *w, double *v, int n, int p, int k, double *y_predict) {
	for (int i = 0; i < n; ++i) {
		y_predict[i] = w_0;
		for (int j = 0; j < p; ++j) {
			y_predict[i] += w[j] * x[i * p + j];
		}
		for (int f = 0; f < k; ++f) {
			double coeff_1 = 0., coeff_2 = 0.;
			for (int j = 0; j < p; ++j) {
				coeff_1 += v[j * k + f] * x[i * p + j];
				coeff_2 += v[j * k + f] * x[i * p + j] * v[j * k + f] * x[i * p + j];
			}
			y_predict[i] += 0.5 * (coeff_1 * coeff_1 - coeff_2);
		}
	}

	double *result = (double *)malloc(sizeof(double) * 2);
	result[0] = 0.;
	result[1] = 0.;
	return result;
}

double* get_e(double *x, double *y, double w0, double *w, double *v, int n, int p, int k, double *e) {
	predict(x, w0, w, v, n, p, k, e);
	for (int i = 0; i < n; ++i) {
		e[i] = y[i] - e[i];
	}
}

double* get_w_star(double *x, double *y, double *e, double w0, double *w, double *v, double l_w, int n, int p, int k) {
	double coeff_1, coeff_2, w_l_star;

	for (int l = 0; l < p; ++l) {
		coeff_1 = 0.;
		coeff_2 = 0.;
		for (int i = 0; i < n; ++i) {
			coeff_1 += x[i * p + l] * x[i * p + l];
			coeff_2 += x[i * p + l] * e[i];
		}
		w_l_star = (w[l] * coeff_1 + coeff_2) / (coeff_1 + l_w);
		w[l] = w_l_star;
	}

	double *result = (double *)malloc(sizeof(double) * 2);
	result[0] = 0.;
	result[1] = 0.;
	return result;
}

double* get_v_star(double *x, double *y, double *e, double w0, double *w, double *v, double l_v, int n, int p, int k) {
	double coeff_1, coeff_2, h_l_f_i;
	double *q = (double *)malloc(sizeof(double) * n);
	for (int f = 0; f < k; ++f) {
		for (int i = 0; i < n; ++i) {
			q[i] = 0.;
			for (int l = 0; l < p; ++l) {
				q[i] += v[l * k + f] * x[i * p + l];
			}
		}
		for (int l = 0; l < p; ++l) {
			coeff_1 = 0.;
			coeff_2 = 0.;
			for (int i = 0; i < n; ++i) {
				h_l_f_i = x[i * p + l] * (q[i] - v[l * k + f] * x[i * p + l]);
				coeff_1 += h_l_f_i * h_l_f_i;
				coeff_2 += h_l_f_i * e[i];
			}
			v[l * k + f] = (v[l * k + f] * coeff_1 + coeff_2) / (coeff_1 + l_v);
		}
	}
	free(q);

	double *result = (double *)malloc(sizeof(double) * 2);
	result[0] = 0.;
	result[1] = 0.;
	return result;
}