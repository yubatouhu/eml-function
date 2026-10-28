use super::eml::EmlTree;
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct Vector {
    pub elements: Vec<EmlTree>,
}

impl Vector {
    pub fn new(elements: Vec<EmlTree>) -> Self {
        Self { elements }
    }

    pub fn size(&self) -> usize {
        self.elements.len()
    }

    pub fn evaluate(&self, x: Complex64) -> Vec<Complex64> {
        self.elements.iter().map(|e| e.eval(x, &[])).collect()
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<EmlTree>>,
}

impl Matrix {
    pub fn new(data: Vec<Vec<EmlTree>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    pub fn apply(&self, vec: &Vector) -> Vector {
        self.mul_vec(vec)
    }

    pub fn evaluate_all(&self, x: Complex64) -> Vec<Vec<Complex64>> {
        self.data.iter()
            .map(|row| row.iter().map(|e| e.eval(x, &[])).collect())
            .collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(EmlTree::add(self.data[i][j].clone(), other.data[i][j].clone()));
            }
            result_data.push(row);
        }
        Matrix::new(result_data)
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result_data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(other.cols);
            for j in 0..other.cols {
                if self.cols == 0 { row.push(EmlTree::zero()); continue; }
                let mut cell = EmlTree::mul(self.data[i][0].clone(), other.data[0][j].clone());
                for k in 1..self.cols { cell = EmlTree::add(cell, EmlTree::mul(self.data[i][k].clone(), other.data[k][j].clone())); }
                row.push(cell);
            }
            result_data.push(row);
        }
        Matrix::new(result_data)
    }

    pub fn mul_vec(&self, vec: &Vector) -> Vector {
        assert_eq!(self.cols, vec.size());
        let mut elements = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            if self.cols == 0 { elements.push(EmlTree::zero()); continue; }
            let mut cell = EmlTree::mul(self.data[i][0].clone(), vec.elements[0].clone());
            for k in 1..self.cols { cell = EmlTree::add(cell, EmlTree::mul(self.data[i][k].clone(), vec.elements[k].clone())); }
            elements.push(cell);
        }
        Vector::new(elements)
    }

    pub fn rank(&self, x: Complex64) -> usize {
        let mut mat = self.evaluate_all(x);
        let mut rank = 0;
        let mut pivot_row = 0;
        let eps = 1e-10;
        for j in 0..self.cols {
            if pivot_row >= self.rows { break; }
            let mut max_row = pivot_row;
            for i in pivot_row + 1..self.rows { if mat[i][j].norm() > mat[max_row][j].norm() { max_row = i; } }
            if mat[max_row][j].norm() < eps { continue; }
            mat.swap(pivot_row, max_row);

            let pivot_row_data = mat[pivot_row].clone(); // クローンして借用を避ける
            for i in 0..self.rows {
                if i != pivot_row {
                    let factor = mat[i][j] / pivot_row_data[j];
                    for k in j..self.cols { mat[i][k] -= factor * pivot_row_data[k]; }
                }
            }
            pivot_row += 1; rank += 1;
        }
        rank
    }

    pub fn det(&self, x: Complex64) -> Complex64 {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut mat = self.evaluate_all(x);
        let mut det = Complex64::new(1.0, 0.0);
        let eps = 1e-12;
        for i in 0..n {
            let mut pivot = i;
            for j in i + 1..n { if mat[j][i].norm() > mat[pivot][i].norm() { pivot = j; } }
            if mat[pivot][i].norm() < eps { return Complex64::new(0.0, 0.0); }
            if pivot != i { mat.swap(i, pivot); det = -det; }
            det *= mat[i][i];

            let pivot_row_data = mat[i].clone(); // クローン
            for j in i + 1..n {
                let factor = mat[j][i] / pivot_row_data[i];
                for k in i + 1..n { mat[j][k] -= factor * pivot_row_data[k]; }
            }
        }
        det
    }

    pub fn inv(&self, x: Complex64) -> Option<Matrix> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut mat = self.evaluate_all(x);
        let mut inv = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n { inv[i][i] = Complex64::new(1.0, 0.0); }
        let eps = 1e-12;
        for i in 0..n {
            let mut pivot = i;
            for j in i + 1..n { if mat[j][i].norm() > mat[pivot][i].norm() { pivot = j; } }
            if mat[pivot][i].norm() < eps { return None; }
            mat.swap(i, pivot); inv.swap(i, pivot);
            let div = mat[i][i];
            for j in 0..n { mat[i][j] /= div; inv[i][j] /= div; }

            let pivot_row_mat = mat[i].clone();
            let pivot_row_inv = inv[i].clone();
            for j in 0..n {
                if i != j {
                    let factor = mat[j][i];
                    for k in 0..n {
                        mat[j][k] -= factor * pivot_row_mat[k];
                        inv[j][k] -= factor * pivot_row_inv[k];
                    }
                }
            }
        }
        Some(Matrix::new(inv.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect()))
    }

    pub fn qr(&self, x: Complex64) -> (Matrix, Matrix) {
        let n = self.rows; let m = self.cols;
        let mat = self.evaluate_all(x);
        let mut q = vec![vec![Complex64::new(0.0, 0.0); m]; n];
        let mut r = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for j in 0..m {
            let mut v = vec![Complex64::new(0.0, 0.0); n];
            for i in 0..n { v[i] = mat[i][j]; }
            for k in 0..j {
                let mut dot = Complex64::new(0.0, 0.0);
                for i in 0..n { dot += q[i][k].conj() * mat[i][j]; }
                r[k][j] = dot;
                for i in 0..n { v[i] -= dot * q[i][k]; }
            }
            let mut norm_sq = 0.0;
            for i in 0..n { norm_sq += v[i].norm_sqr(); }
            let norm = norm_sq.sqrt();
            r[j][j] = Complex64::new(norm, 0.0);
            if norm > 1e-12 { for i in 0..n { q[i][j] = v[i] / norm; } }
        }
        let q_mat = Matrix::new(q.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect());
        let r_mat = Matrix::new(r.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect());
        (q_mat, r_mat)
    }

    pub fn eigenvalues(&self, x: Complex64) -> Vec<Complex64> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut mat = self.evaluate_all(x);
        let max_iter = 200;
        let eps = 1e-12;
        for _ in 0..max_iter {
            let current_matrix = Matrix::new(mat.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect());
            let (q, r) = current_matrix.qr(Complex64::new(0.0, 0.0));
            let r_eval = r.evaluate_all(Complex64::new(0.0, 0.0));
            let q_eval = q.evaluate_all(Complex64::new(0.0, 0.0));
            let mut next_mat = vec![vec![Complex64::new(0.0, 0.0); n]; n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n { next_mat[i][j] += r_eval[i][k] * q_eval[k][j]; }
                }
            }
            let mut sum_lower = 0.0;
            for i in 1..n { for j in 0..i { sum_lower += next_mat[i][j].norm(); } }
            mat = next_mat;
            if sum_lower < eps { break; }
        }
        (0..n).map(|i| mat[i][i]).collect()
    }

    pub fn diagonalize(&self, x: Complex64) -> Option<(Matrix, Matrix)> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let evals = self.eigenvalues(x);
        let mat_eval = self.evaluate_all(x);
        let mut p_data = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let mut d_data = vec![vec![Complex64::new(0.0, 0.0); n]; n];

        for i in 0..n {
            d_data[i][i] = evals[i];
            let mut v = vec![Complex64::new(1.0, 0.0); n];
            let shift = evals[i] + Complex64::new(1e-4, 1e-4);
            for _ in 0..5 {
                let mut a_shifted = vec![vec![Complex64::new(0.0, 0.0); n]; n];
                for r in 0..n {
                    for c in 0..n {
                        a_shifted[r][c] = mat_eval[r][c];
                        if r == c { a_shifted[r][c] -= shift; }
                    }
                }
                let shifted_matrix = Matrix::new(a_shifted.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect());
                if let Some(inv_a) = shifted_matrix.inv(Complex64::new(0.0, 0.0)) {
                    // 数値的に計算（EmlTreeの構築を経由すると exp(2500) などでオーバーフローするため）
                    let inv_a_eval = inv_a.evaluate_all(Complex64::new(0.0, 0.0));
                    let mut v_next = vec![Complex64::new(0.0, 0.0); n];
                    for r in 0..n {
                        for c in 0..n {
                            v_next[r] += inv_a_eval[r][c] * v[c];
                        }
                    }

                    let mut max_norm = 0.0;
                    for val in &v_next { if val.norm() > max_norm { max_norm = val.norm(); } }

                    if max_norm < 1e-15 || max_norm.is_nan() || max_norm.is_infinite() {
                        if max_norm.is_infinite() {
                            let mut found = false;
                            for val in v.iter_mut() { *val = Complex64::new(0.0, 0.0); }
                            for (idx, val) in v_next.iter().enumerate() {
                                if val.norm().is_infinite() {
                                    v[idx] = Complex64::new(1.0, 0.0);
                                    found = true;
                                    break;
                                }
                            }
                            if found { break; }
                        }
                        break;
                    }
                    v = v_next.iter().map(|&c| c / max_norm).collect();
                } else {
                    break;
                }
            }
            // 最終的な正規化（L2ノルム）
            let mut norm_sq = 0.0;
            for val in &v { norm_sq += val.norm_sqr(); }
            let norm = norm_sq.sqrt();
            if norm > 1e-15 {
                for val in v.iter_mut() { *val /= norm; }
            }
            for r in 0..n { p_data[r][i] = v[r]; }
        }
        let p = Matrix::new(p_data.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect());
        let d = Matrix::new(d_data.iter().map(|row| row.iter().map(|&c| EmlTree::Value(c)).collect()).collect());
        Some((p, d))
    }
}
