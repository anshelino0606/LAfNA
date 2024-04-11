extern crate core;

use core::f64;
use std::fs::File;
use std::{fmt, io};
use std::io::{BufRead, BufReader};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::path::Path;
use std::str::FromStr;

struct Vector {
    data: Vec<f64>,
    len: usize
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
pub struct PQLU {
   // pub p: Vec<usize>, // Permutation matrix for rows
   // pub q: Vec<usize>, // Permutation matrix for columns
    pub l: Matrix,
    pub u: Matrix,
}

impl Matrix
{
    // Constructor for a new Matrix with given rows and columns, initialized to default for type T
    fn new(rows: usize, cols: usize) -> Self {
        let default_value = f64::default();
        Matrix {
            data: vec![vec![default_value; cols]; rows],
            rows,
            cols,
        }
    }

    // Create an identity matrix of size dim
    // Note: This makes sense only for numeric types that can represent 1 and 0
    fn identity(dim: usize) -> Self
    {
        let mut matrix = Self::new(dim, dim);
        for i in 0..dim {
            matrix.data[i][i] = f64::from(1);
        }
        matrix
    }

    // Create a zero matrix with given dimensions
    // This uses T::default() and assumes that the default value for T represents "zero"
    fn zero(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    // Prints the matrix
    fn print(&self)
    {
        for row in &self.data {
            println!("{:?}", row);
        }
    }

    // Method to add two matrices
    fn add(&self, other: &Matrix) -> Option<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j].clone() + other.data[i][j].clone();
            }
        }
        Some(result)
    }

    // Method to subtract two matrices
    fn subtract(&self, other: &Matrix) -> Option<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j].clone() - other.data[i][j].clone();
            }
        }
        Some(result)
    }

    // Method to multiply each element of a matrix by a scalar
    // Assumes that T can be multiplied by T and returns T
    fn scalar_multiply(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = self.data[i][j].clone() * scalar.clone();
            }
        }
    }

    // Method to multiply two matrices
    // fn multiply(&self, other: &Matrix) -> Option<Matrix> {
    //     if self.cols != other.rows {
    //         return None; // Incompatible dimensions
    //     }
    //
    //     let mut result = Matrix::new(self.rows, other.cols);
    //     for i in 0..self.rows {
    //         for j in 0..other.cols {
    //             result.data[i][j] = (0..self.cols)
    //                 .map(|k| self.data[i][k].clone() * other.data[k][j].clone())
    //                 .sum();
    //         }
    //     }
    //     Some(result)
    // }

    // Swap two rows by indices
    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 < self.rows && row2 < self.rows {
            self.data.swap(row1, row2);
        }
    }

    // Swap two columns by indices
    fn swap_cols(&mut self, col1: usize, col2: usize) {
        if col1 < self.cols && col2 < self.cols {
            for row in self.data.iter_mut() {
                row.swap(col1, col2);
            }
        }
    }

    // Multiply a row by a scalar
    fn multiply_row_by_scalar(&mut self, row_index: usize, scalar: f64) {
        if row_index < self.rows {
            for j in 0..self.cols {
                self.data[row_index][j] = self.data[row_index][j].clone() * scalar.clone();
            }
        }
    }

    // Multiply a column by a scalar
    fn multiply_col_by_scalar(&mut self, col_index: usize, scalar: f64) {
        if col_index < self.cols {
            for i in 0..self.rows {
                self.data[i][col_index] = self.data[i][col_index].clone() * scalar.clone();
            }
        }
    }

    fn gaussian_elimination(&mut self) -> i32 {
        let n = self.rows;
        let mut sign = 1; // Keep track of the sign of the determinant
        let epsilon = 1e-10; // Arbitrarily chosen small value

        for i in 0..n {
            // Find the pivot row based on maximum absolute value
            let mut max_row = i;
            for k in i + 1..n {
                if self.data[k][i].abs() > self.data[max_row][i].abs() {
                    max_row = k;
                }
            }

            // If the pivot is effectively zero, skip the current column
            if self.data[max_row][i].abs() < epsilon {
                continue;
            }

            // Swap if the pivot row is different from the current row
            if i != max_row {
                self.swap_rows(i, max_row);
                sign = -sign; // A row swap changes the sign of the determinant
            }

            // Perform elimination below the pivot
            for j in i + 1..n {
                let factor = self.data[j][i] / self.data[i][i];
                for k in i..self.cols {
                    self.data[j][k] -= factor * self.data[i][k];
                }
            }
        }

        sign
    }

    /// Check if matrix is triadiagonal
    /// | 1 1 0 0 |
    /// | 1 1 1 0 |
    /// | 0 1 1 1 |
    /// | 0 0 1 1 |
    pub fn is_tridiagonal(&self) -> bool {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if (i != j) && (i != j + 1) && (i + 1 != j) {
                    if self.data[i][j].abs() > 1e-10 { // Consider as zero if abs value is smaller than epsilon
                        return false;
                    }
                }
            }
        }
        true
    }
}


impl Matrix
{
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut data: Vec<Vec<f64>> = Vec::new();

        for line in reader.lines() {
            let row_data = line?
                .split_whitespace()
                .map(|s| s.parse().expect("Could not parse number"))
                .collect::<Vec<f64>>();
            data.push(row_data);
        }

        let rows = data.len();
        // Assuming the matrix is not empty and is well-formed (all rows have the same length)
        let cols = if rows > 0 { data[0].len() } else { 0 };

        Ok(Matrix { data, rows, cols })
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    // Method to set an element of the matrix
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j][i] = self.data[i][j];
            }
        }
        transposed
    }

    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols != other.rows {
            return Err("Matrix dimensions do not match for multiplication.");
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Ok(result)
    }

    pub fn multiply_vector(&self, vector: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
        if self.cols != vector.len() {
            return Err("Dimensions do not match for multiplication.");
        }
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * vector[j];
            }
        }
        Ok(result)
    }
}

impl Vector {
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let data: Vec<f64> = reader.lines()
            .map(|line| line.expect("Could not read line")
                .parse::<f64>().expect("Could not parse number"))
            .collect();

        let len = data.len();
        Ok(Vector { data, len })
    }
}


mod numerical_methods {
    use super::Matrix; // Assuming Matrix is defined in the parent module
    use super::PQLU; // Assuming Matrix is defined in the parent module
    pub fn gaussian_elimination(matrix: &mut Matrix) -> Result<f64, &'static str> {
        let n = matrix.rows;
        let mut sign = 1;
        let epsilon = 1e-10;
        let mut det = 1.0;

        for i in 0..n {
            // Find the pivot row based on maximum absolute value
            let mut max_row = i;
            for k in i + 1..n {
                if matrix.data[k][i].abs() > matrix.data[max_row][i].abs() {
                    max_row = k;
                }
            }

            if matrix.data[max_row][i].abs() < epsilon {
                return Err("Matrix is singular or nearly singular");
            }

            // Swap if the pivot row is different from the current row
            if i != max_row {
                matrix.swap_rows(i, max_row);
                sign = -sign; // A row swap changes the sign of the determinant
            }

            // Perform elimination below the pivot
            for j in i + 1..n {
                let factor = matrix.data[j][i] / matrix.data[i][i];
                for k in i..matrix.cols {
                    matrix.data[j][k] -= factor * matrix.data[i][k];
                }
                matrix.data[j][i] = 0.0; // Explicitly setting to zero for clarity
            }

            det *= matrix.data[i][i];
        }

        Ok(det * sign as f64)
    }

    pub fn solve(matrix: &Matrix, b: Vec<f64>) -> Result<Vec<f64>, &'static str> {
        if matrix.rows != b.len() {
            return Err("Matrix and vector dimensions do not match");
        }

        // Clone matrix and augment it with b
        let mut aug_matrix = matrix.clone();
        for i in 0..aug_matrix.rows {
            aug_matrix.data[i].push(b[i]);
        }
        aug_matrix.cols += 1;

        // Apply Gaussian elimination
        if gaussian_elimination(&mut aug_matrix).is_err() {
            return Err("Matrix is singular or nearly singular");
        }

        // Back substitution
        let mut x = vec![0.0; matrix.rows];
        for i in (0..matrix.rows).rev() {
            x[i] = aug_matrix.data[i][matrix.cols] / aug_matrix.data[i][i];
            for k in (0..i).rev() {
                aug_matrix.data[k][matrix.cols] -= aug_matrix.data[k][i] * x[i];
            }
        }

        Ok(x)
    }

    pub fn invert(matrix: &Matrix) -> Result<Matrix, &'static str> {
        if matrix.rows != matrix.cols {
            return Err("Matrix must be square to compute its inverse");
        }

        // Clone matrix and augment it with the identity matrix
        let mut aug_matrix = matrix.clone();
        for i in 0..aug_matrix.rows {
            for j in 0..matrix.rows {
                aug_matrix.data[i].push(if i == j { 1.0 } else { 0.0 });
            }
        }
        aug_matrix.cols *= 2;

        // Apply Gaussian elimination
        if gaussian_elimination(&mut aug_matrix).is_err() {
            return Err("Matrix is singular or nearly singular, cannot be inverted");
        }

        // Transform to echelon form and normalize to get the identity matrix on the left
        for i in 0..matrix.rows {
            let divisor = aug_matrix.data[i][i];
            for j in 0..aug_matrix.cols {
                aug_matrix.data[i][j] /= divisor;
            }

            for k in 0..matrix.rows {
                if k != i {
                    let factor = aug_matrix.data[k][i];
                    for j in 0..aug_matrix.cols {
                        aug_matrix.data[k][j] -= factor * aug_matrix.data[i][j];
                    }
                }
            }
        }

        // Extract the inverse matrix from the augmented matrix
        let inverse_data = aug_matrix.data.iter().map(|row| row[matrix.cols..].to_vec()).collect();
        Ok(Matrix {
            data: inverse_data,
            rows: matrix.rows,
            cols: matrix.cols,
        })
    }

    pub fn lu(matrix: &Matrix) -> PQLU {
        let n = matrix.rows;
        let mut l = Matrix::new(n, n);
        let mut u = matrix.clone();

        for k in 0..n {
            // Set the diagonal of L to 1
            l.data[k][k] = 1.0;
            for i in k+1..n {
                l.data[i][k] = u.data[i][k] / u.data[k][k];
                u.data[i][k] = 0.0; // Explicitly setting to zero to maintain upper triangular form
                for j in k+1..n {
                    u.data[i][j] -= l.data[i][k] * u.data[k][j];
                }
            }
        }

        PQLU {
            l,
            u
        }
    }

    pub fn det_lu(matrix: &Matrix) -> f64 {
        let pqlu = lu(matrix);
        let u = pqlu.u;
        let mut det = 1f64;
        for i in 0.. u.rows {
            det *= u.data[i][i];
        }
        det
    }

    pub fn forward_substitution(matrix: &Matrix, b: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
        let n = matrix.rows;
        if n != b.len() {
            return Err("Matrix and vector dimensions do not match");
        }

        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += matrix.data[i][j] * y[j];
            }
            y[i] = (b[i] - sum) / matrix.data[i][i];
        }

        Ok(y)
    }

    pub fn back_substitution(matrix: &Matrix, y: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
        let n = matrix.rows;
        if n != y.len() {
            return Err("Matrix and vector dimensions do not match");
        }

        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i+1..n).rev() {
                sum += matrix.data[i][j] * x[j];
            }
            x[i] = (y[i] - sum) / matrix.data[i][i];
        }

        Ok(x)
    }

    pub fn solve_lu(matrix: &Matrix, b: Vec<f64>) -> Result<Vec<f64>, &'static str> {
        let PQLU { l, u } = lu(matrix);

        let y = forward_substitution(&l, &b)?;

        let x = back_substitution(&u, &y)?;

        Ok(x)
    }

    pub fn utu(u: &Matrix) -> Result<Matrix, &'static str> {
        let u_transposed = u.transpose();
        u_transposed.multiply(u)
    }

    pub fn cholesky_decompose(matrix: &Matrix) -> Result<Matrix, &'static str> {
        if matrix.rows != matrix.cols {
            return Err("Matrix must be square.");
        }

        let n = matrix.rows;
        let mut l = Matrix::new(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                // Summation for diagonals
                if j == i {
                    for k in 0..j {
                        sum += l.data[j][k] * l.data[j][k];
                    }
                    let val = matrix.data[j][j] - sum;
                    if val <= 0.0 {
                        return Err("Matrix is not positive definite.");
                    }
                    l.data[j][j] = val.sqrt();
                } else {
                    // Summation for non-diagonals
                    for k in 0..j {
                        sum += l.data[i][k] * l.data[j][k];
                    }
                    if l.data[j][j] == 0.0 {
                        return Err("Division by zero encountered.");
                    }
                    l.data[i][j] = (matrix.data[i][j] - sum) / l.data[j][j];
                }
            }
        }

        Ok(l)
    }

    pub fn solve_spd(matrix: &Matrix, b: Vec<f64>) -> Result<Vec<f64>, &'static str> {
        // Perform Cholesky decomposition to get L where A = LL^T
        let l = cholesky_decompose(&matrix)?;

        // Solve Ly = b using forward substitution
        let y = forward_substitution(&l, &b)?;

        // Transpose L to get L^T
        let lt = l.transpose();

        // Solve L^Tx = y using back substitution
        let x = back_substitution(&lt, &y)?;

        Ok(x)
    }

    pub fn solve_utu(u: &Matrix, b: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
        if u.rows != b.len() {
            return Err("Dimensions of U and b do not match.");
        }

        // Compute U^Tb
        let ut = u.transpose();
        let utb = ut.multiply_vector(b)?;

        // Compute U^TU
        let utu = ut.multiply(u)?;

        // Now solve U^TUx = U^Tb. Assuming you have a method to solve such systems,
        // possibly through Cholesky decomposition if U^TU is positive definite.
        // This placeholder assumes such a method is implemented.
        let x = solve_spd(&utu, utb)?;

        Ok(x)
    }

    // Solves Ax = d for a tridiagonal matrix A and vector d.
    // Assumes self is tridiagonal and square.
    pub fn thompson_algorithm_rs(matrix: &Matrix, d: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
        if !matrix.is_tridiagonal() {
            return Err("Matrix is not tridiagonal");
        }
        if matrix.rows != d.len() {
            return Err("Dimension mismatch between matrix and vector");
        }

        let n = matrix.rows;
        let mut c_star = vec![0.0; n]; // Modified coefficients
        let mut d_star = vec![0.0; n]; // Modified vector
        let mut x = vec![0.0; n];      // Solution vector

        // Forward sweep for c_star and d_star
        c_star[0] = matrix.get(0, 1) / matrix.get(0, 0);
        d_star[0] = d[0] / matrix.get(0, 0);
        for i in 1..n-1 {
            let denom = matrix.get(i, i) - matrix.get(i, i-1) * c_star[i-1];
            c_star[i] = matrix.get(i, i+1) / denom;
            d_star[i] = (d[i] - matrix.get(i, i-1) * d_star[i-1]) / denom;
        }
        d_star[n-1] = (d[n-1] - matrix.get(n-1, n-2) * d_star[n-2]) / (matrix.get(n-1, n-1) - matrix.get(n-1, n-2) * c_star[n-2]);

        // Backward substitution for x
        x[n-1] = d_star[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_star[i] - c_star[i] * x[i+1];
        }

        Ok(x)
    }

    // Solves Ax = d for a tridiagonal matrix A and vector d using a left-sided sweep.
    // Assumes self is tridiagonal and square.
    pub fn thompson_algorithm_ls(matrix: &Matrix, d: &Vec<f64>) -> Result<Vec<f64>, &'static str> {
        if !matrix.is_tridiagonal() {
            return Err("Matrix is not tridiagonal");
        }
        if matrix.rows != d.len() {
            return Err("Dimension mismatch between matrix and vector");
        }

        let n = matrix.rows;
        let mut b_star = vec![0.0; n]; // Modified coefficients for lower diagonal
        let mut d_star = vec![0.0; n]; // Modified right-hand side vector
        let mut x = vec![0.0; n];      // Solution vector

        // Initial values based on the bottom row of the matrix
        b_star[n-1] = matrix.get(n-1, n-2) / matrix.get(n-1, n-1);
        d_star[n-1] = d[n-1] / matrix.get(n-1, n-1);

        // Backward sweep for b_star and d_star
        for i in (1..n-1).rev() {
            let denom = matrix.get(i, i) - matrix.get(i, i+1) * b_star[i+1];
            b_star[i] = matrix.get(i, i-1) / denom;
            d_star[i] = (d[i] - matrix.get(i, i+1) * d_star[i+1]) / denom;
        }
        let denom = matrix.get(0, 0) - matrix.get(0, 1) * b_star[1];
        d_star[0] = (d[0] - matrix.get(0, 1) * d_star[1]) / denom;

        // Forward substitution for x
        x[0] = d_star[0];
        for i in 1..n {
            x[i] = d_star[i] - b_star[i] * x[i-1];
        }

        Ok(x)
    }

}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for (j, &val) in row.iter().enumerate() {
                if j == 0 {
                    write!(f, "|{:.2}", val)?;
                } else {
                    write!(f, " {:.2}", val)?;
                }
            }
            writeln!(f, "|")?;
        }
        Ok(())
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &val) in self.data.iter().enumerate() {
            if i < self.data.len() - 1 {
                write!(f, "{:.2}, ", val)?;
            } else {
                write!(f, "{:.2}", val)?;
            }
        }
        write!(f, "]")
    }
}

struct LinearSystem {
    // Solve Ax = b
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl LinearSystem {
    fn new(a: Vec<Vec<f64>>, b: Vec<f64>) -> Self {
        Self { a, b }
    }

    // Method stubs for solvers
    fn jacobi(&self, tolerance: f64, max_iterations: usize) -> (Vec<f64>, f64, usize) {
        let n = self.b.len();
        let mut x = vec![0.0; n]; // Initial guess (x_0)
        let mut x_prev = x.clone();
        let mut error = 0.0;
        let mut iterations = 0;

        for _ in 0..max_iterations {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += self.a[i][j] * x_prev[j];
                    }
                }
                x[i] = (self.b[i] - sum) / self.a[i][i];
            }

            // Calculate the norm of the difference between successive approximations
            error = x.iter().zip(x_prev.iter()).fold(0.0, |acc, (&xi, &prev_xi)| acc + (xi - prev_xi).powi(2)).sqrt();
            if error < tolerance {
                break;
            }

            x_prev = x.clone();
            iterations += 1;
        }

        (x, error, iterations)
    }
    fn gauss_seidel(&self, tolerance: f64, max_iterations: usize) -> (Vec<f64>, f64, usize) {
        let n = self.b.len();
        let mut x = vec![0.0; n]; // Initial guess (x_0)
        let mut x_prev = x.clone(); // To store the previous iteration's values for convergence check
        let mut error = 0.0;
        let mut iterations = 0;

        for _ in 0..max_iterations {
            for i in 0..n {
                let mut sum_before = 0.0;
                let mut sum_after = 0.0;
                for j in 0..i {
                    sum_before += self.a[i][j] * x[j]; // Use the latest available values
                }
                for j in i+1..n {
                    sum_after += self.a[i][j] * x_prev[j]; // Use values from the previous iteration
                }
                x[i] = (self.b[i] - sum_before - sum_after) / self.a[i][i];
            }

            // Calculate the norm of the difference between successive approximations
            error = x.iter().zip(x_prev.iter()).fold(0.0, |acc, (&xi, &prev_xi)| acc + (xi - prev_xi).powi(2)).sqrt();
            if error < tolerance {
                break;
            }

            x_prev = x.clone();
            iterations += 1;
        }

        (x, error, iterations)
    }
    fn sor(&self, omega: f64, tolerance: f64, max_iterations: usize) -> (Vec<f64>, f64, usize) {
        let n = self.b.len();
        let mut x = vec![0.0; n]; // Initial guess (x_0)
        let mut x_prev = vec![0.0; n]; // To store the previous iteration's values for convergence check
        let mut error = 0.0;
        let mut iterations = 0;

        for _ in 0..max_iterations {
            for i in 0..n {
                let mut sum_before = 0.0; // Sum for j < i using latest x
                let mut sum_after = 0.0; // Sum for j > i using x from the previous iteration
                for j in 0..i {
                    sum_before += self.a[i][j] * x[j];
                }
                for j in i+1..n {
                    sum_after += self.a[i][j] * x_prev[j];
                }
                let x_new = (self.b[i] - sum_before - sum_after) / self.a[i][i];
                x[i] = x_prev[i] + omega * (x_new - x_prev[i]); // Apply relaxation
            }

            // Calculate the norm of the difference between successive approximations
            error = x.iter().zip(x_prev.iter()).fold(0.0, |acc, (&xi, &prev_xi)| acc + (xi - prev_xi).powi(2)).sqrt();
            if error < tolerance {
                break;
            }

            x_prev = x.clone();
            iterations += 1;
        }

        (x, error, iterations)
    }

    // TODO: Helper methods for input, convergence check, etc.
    // TODO: ORGANIZE EVERYTHING (THIS IS A MESS)
}

fn thompson_algorithm_optimized(diag_main: &[f64], diag_sub: &[f64], diag_sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag_main.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut y = vec![0.0; n];

    // Forward sweep
    c_prime[0] = diag_sup[0] / diag_main[0];
    d_prime[0] = rhs[0] / diag_main[0];
    for i in 1..n {
        let m = diag_main[i] - diag_sub[i-1] * c_prime[i-1];
        c_prime[i] = if i < n-1 { diag_sup[i] / m } else { 0.0 };
        d_prime[i] = (rhs[i] - diag_sub[i-1] * d_prime[i-1]) / m;
    }

    // Backward substitution
    y[n-1] = d_prime[n-1];
    for i in (0..n-1).rev() {
        y[i] = d_prime[i] - c_prime[i] * y[i+1];
    }

    y
}

fn main() {
    // Example linear system (A * x = b)
    let a = vec![
        vec![10.0, -1.0, 2.0, 0.0],
        vec![-1.0, 11.0, -1.0, 3.0],
        vec![2.0, -1.0, 10.0, -1.0],
        vec![0.0, 3.0, -1.0, 8.0],
    ];
    let b = vec![6.0, 25.0, -11.0, 15.0];

    // Create a LinearSystem instance
    let system = LinearSystem::new(a, b);

    // Solve the system using the Jacobi method
    let tolerance = 1e-10; // Convergence tolerance
    let max_iterations = 100; // Maximum number of iterations
    let (solution, error, iterations) = system.jacobi(tolerance, max_iterations);

    println!("Solution: {:?}", solution);
    println!("Error: {:?}", error);
    println!("Iterations: {:?}", iterations);
}