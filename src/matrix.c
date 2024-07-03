#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1
    return mat->data[(mat->cols * row) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1
    mat->data[(mat->cols * row) + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows < 1 || cols < 1) {
        return -1;
    }
    matrix *mat_pointer = malloc(sizeof(matrix));
    if (mat_pointer == NULL) {
        return -2;
    }
    double *data = calloc(rows * cols, sizeof(double));
    if (data == NULL) {
        free(mat_pointer);
        return -2;
    }
    mat_pointer->rows = rows;
    mat_pointer->cols = cols;
    mat_pointer->data = data;
    mat_pointer->parent = NULL;
    mat_pointer->ref_cnt = 1;
    *mat = mat_pointer;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
	if (mat == NULL) {
		return;
	}
	if (mat->parent == NULL) {
		mat->ref_cnt--;
		if (mat->ref_cnt == 0) {
			free(mat->data);
			free(mat);
		}
	} else {
		deallocate_matrix(mat->parent);
		free(mat);
	}
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
	if (rows < 1 || cols < 1) {
		return -1;
	}
	matrix* matrix_struct = (matrix *) malloc(sizeof(matrix));
	if (matrix_struct == NULL) {
		return -2;
	}
	matrix_struct->data = from->data + offset;
	matrix_struct->rows = rows;
	matrix_struct->cols = cols;
	matrix_struct->parent = from;
	from->ref_cnt++;
	matrix_struct->ref_cnt = 1;
	*mat = matrix_struct;
	return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
	int size = mat->rows * mat->cols;
	__m256d data = _mm256_set1_pd(val);
	#pragma omp parallel for if (size > 10000)
	for (int i = 0; i < size / 16 * 16; i += 16) {
		_mm256_storeu_pd(mat->data + i, data);
		_mm256_storeu_pd(mat->data + i + 4, data);
		_mm256_storeu_pd(mat->data + i + 8, data);
		_mm256_storeu_pd(mat->data + i + 12, data);
	}
	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		mat->data[i] = val;
	}
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5
	int size = mat->rows * mat->cols;
	__m256d neg_one = _mm256_set1_pd(-1);
	#pragma omp parallel for
	for (int i = 0; i < size / 16 * 16; i += 16) {
		__m256d data = _mm256_loadu_pd(mat->data + i);
		__m256d neg_data = _mm256_mul_pd(neg_one, data);
		__m256d abs_data = _mm256_max_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i, abs_data);

		data = _mm256_loadu_pd(mat->data + i + 4);
		neg_data = _mm256_mul_pd(neg_one, data);
		abs_data = _mm256_max_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i + 4, abs_data);

		data = _mm256_loadu_pd(mat->data + i + 8);
		neg_data = _mm256_mul_pd(neg_one, data);
		abs_data = _mm256_max_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i + 8, abs_data);

		data = _mm256_loadu_pd(mat->data + i + 12);
		neg_data = _mm256_mul_pd(neg_one, data);
		abs_data = _mm256_max_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i + 12, abs_data);
	}
	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		if (mat->data[i] < 0) {
			result->data[i] = mat->data[i] * -1;
		} else {
			result->data[i] = mat->data[i];
		}
	}
	return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5
	int size = mat->rows * mat->cols;
	__m256d neg_one = _mm256_set1_pd(-1);
	#pragma omp parallel for
	for (int i = 0; i < size / 16 * 16; i += 16) {
		__m256d data = _mm256_loadu_pd(mat->data + i);
		__m256d neg_data = _mm256_mul_pd(neg_one, data);
		__m256d abs_data = _mm256_min_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i, abs_data);

		data = _mm256_loadu_pd(mat->data + i + 4);
		neg_data = _mm256_mul_pd(neg_one, data);
		abs_data = _mm256_min_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i + 4, abs_data);

		data = _mm256_loadu_pd(mat->data + i + 8);
		neg_data = _mm256_mul_pd(neg_one, data);
		abs_data = _mm256_min_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i + 8, abs_data);

		data = _mm256_loadu_pd(mat->data + i + 12);
		neg_data = _mm256_mul_pd(neg_one, data);
		abs_data = _mm256_min_pd(neg_data, data);
		_mm256_storeu_pd(result->data + i + 12, abs_data);
	}
	#pragma omp parallel for
	for (int i = size / 16 * 16; i < size; i++) {
		if (mat->data[i] > 0) {
			result->data[i] = mat->data[i] * -1;
		} else {
			result->data[i] = mat->data[i];
		}
	}
	return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5
	int size = mat1->rows * mat1->cols;
	#pragma omp parallel for if (size > 10000)
	for (int i = 0; i < size / 16 * 16; i+= 16) {
		__m256d data1 = _mm256_loadu_pd(mat1->data + i);
		__m256d data2 = _mm256_loadu_pd(mat2->data + i);
		__m256d res = _mm256_add_pd(data1, data2);
		_mm256_storeu_pd(result->data + i, res);
		
		data1 = _mm256_loadu_pd(mat1->data + i + 4);
		data2 = _mm256_loadu_pd(mat2->data + i + 4);
		res = _mm256_add_pd(data1, data2);
		_mm256_storeu_pd(result->data + i + 4, res);
		
		data1 = _mm256_loadu_pd(mat1->data + i + 8);
		data2 = _mm256_loadu_pd(mat2->data + i + 8);
		res = _mm256_add_pd(data1, data2);
		_mm256_storeu_pd(result->data + i + 8, res);
		
		data1 = _mm256_loadu_pd(mat1->data + i + 12);
		data2 = _mm256_loadu_pd(mat2->data + i + 12);
		res = _mm256_add_pd(data1, data2);
		_mm256_storeu_pd(result->data + i + 12, res);
	}
	for (int i = size / 16 * 16; i < size; i++) {
		result->data[i] = mat1->data[i] + mat2->data[i];
	}
	return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5
	int size = mat1->rows * mat1->cols;
	#pragma omp parallel for if (size > 10000)
	for (int i = 0; i < size / 16 * 16; i+= 16) {
		__m256d data1 = _mm256_loadu_pd(mat1->data + i);
		__m256d data2 = _mm256_loadu_pd(mat2->data + i);
		__m256d res = _mm256_sub_pd(data1, data2);
		_mm256_storeu_pd(result->data + i, res);
		
		data1 = _mm256_loadu_pd(mat1->data + i + 4);
		data2 = _mm256_loadu_pd(mat2->data + i + 4);
		res = _mm256_sub_pd(data1, data2);
		_mm256_storeu_pd(result->data + i + 4, res);
		
		data1 = _mm256_loadu_pd(mat1->data + i + 8);
		data2 = _mm256_loadu_pd(mat2->data + i + 8);
		res = _mm256_sub_pd(data1, data2);
		_mm256_storeu_pd(result->data + i + 8, res);
		
		data1 = _mm256_loadu_pd(mat1->data + i + 12);
		data2 = _mm256_loadu_pd(mat2->data + i + 12);
		res = _mm256_sub_pd(data1, data2);
		_mm256_storeu_pd(result->data + i + 12, res);
	}
	for (int i = size / 16 * 16; i < size; i++) {
		result->data[i] = mat1->data[i] - mat2->data[i];
	}
	return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
double* do_transpose(matrix* mat) {
	double *transpose = malloc(sizeof(double) * mat->rows * mat->cols);
	#pragma omp parallel for
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			transpose[mat->cols * i + j] = mat->data[mat->cols * j + i];
		}
	}
	return transpose;
}

int mul_matrix (matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6
	double* transpose = do_transpose(mat2);
	#pragma omp parallel for
	for (int i = 0; i < result->rows * result->cols; i++) {
		int add1 = mat1->cols * (i / result->cols);
		int add2 = mat1->cols * (i % result->cols);
		__m256d res = _mm256_set1_pd(0);
		for (int j = 0 ; j < mat1->cols / 16 * 16; j += 16) {
			__m256d data1 = _mm256_loadu_pd(mat1->data + add1 + j);
			__m256d data2 = _mm256_loadu_pd(transpose + add2 + j);
			res = _mm256_fmadd_pd(data1, data2, res);
			
			data1 = _mm256_loadu_pd(mat1->data + add1 + j + 4);
			data2 = _mm256_loadu_pd(transpose + add2 + j + 4);
			res = _mm256_fmadd_pd(data1, data2, res);
			
			data1 = _mm256_loadu_pd(mat1->data + add1 + j + 8);
			data2 = _mm256_loadu_pd(transpose + add2 + j + 8);
			res = _mm256_fmadd_pd(data1, data2, res);
			
			data1 = _mm256_loadu_pd(mat1->data + add1 + j + 12);
			data2 = _mm256_loadu_pd(transpose + add2 + j + 12);
			res = _mm256_fmadd_pd(data1, data2, res);
		}
		result->data[i] = res[0] + res[1] + res[2] + res[3];
		for (int j = mat1->cols / 16 * 16; j < mat1->cols; j++) {
			result->data[i] += transpose[add2 + j] * mat1->data[add1 + j];
		}
	}
	free(transpose);
	return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int unit_matrix(matrix *mat) {
	int diagonal = 0;
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			if (j == diagonal) {
				set(mat, i, j, 1);
			} else {
				set(mat, i,j, 0);
			}
		}
		diagonal++;
	}
	return 0;
}

void copy_matrix (matrix *result, matrix* mat) {
	result->cols = mat->cols;
	result->rows = mat->rows;
	for (int i = 0; i < mat->cols * mat->rows; i++) {
			result->data[i] = mat->data[i];
	}
}

int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
	if (pow == 0) {
		unit_matrix(result);
		return 0;
	}
	if (pow == 1) {
		copy_matrix(result, mat);
		return 0;
	} 
	if (pow == 2) {
		mul_matrix(result, mat, mat);
		return 0;
	}
	matrix *local_result;
	allocate_matrix(&local_result, mat->rows, mat->cols);
	if (pow % 2){
		pow_matrix(local_result, mat, pow-1);
		mul_matrix(result, local_result, mat);
	} else {
		pow_matrix(local_result, mat, pow/2);
		mul_matrix(result, local_result, local_result);
	}
	deallocate_matrix(local_result);
	return 0;



		/*for (int i = 0; i < mat->cols * mat->rows; i++) {
			local_result->data[i] = mat->data[i];
		}
		matrix *placeholder;
		allocate_matrix(&placeholder, mat->rows, mat->cols);
		for (int i = pow; i > 1; i--) {
			for (int i = 0; i < mat->cols * mat->rows; i++) {
				placeholder->data[i] = local_result->data[i];
			}
			mul_matrix(local_result, placeholder, mat);
		}
		for (int i = 0; i < mat->cols * mat->rows; i++) {
			result->data[i] = local_result->data[i];
		}
		deallocate_matrix(local_result);
		deallocate_matrix(placeholder);*/
	return 0;
}
