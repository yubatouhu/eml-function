use eml_ope::domain::eml::EmlTree;
use eml_ope::domain::vec::Matrix;
use num_complex::Complex64;

fn complex(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

fn n_tree(n: i32) -> EmlTree {
    if n == 0 {
        EmlTree::zero()
    } else if n == 1 {
        EmlTree::One
    } else if n > 1 {
        EmlTree::add(EmlTree::One, n_tree(n - 1))
    } else {
        EmlTree::neg(n_tree(-n))
    }
}

#[test]
fn test_eigenvalues() {
    let m = Matrix::new(vec![
        vec![n_tree(2), n_tree(1)],
        vec![n_tree(1), n_tree(2)],
    ]);
    let evals = m.eigenvalues(complex(0.0));
    let eval_res: Vec<f64> = evals.iter().map(|c| c.re.round()).collect();
    assert!(eval_res.contains(&3.0));
    assert!(eval_res.contains(&1.0));
}

#[test]
fn test_diagonalize_nn() {
    let m = Matrix::new(vec![
        vec![n_tree(2), n_tree(-1), n_tree(0)],
        vec![n_tree(-1), n_tree(2), n_tree(-1)],
        vec![n_tree(0), n_tree(-1), n_tree(2)],
    ]);

    let (p, d) = m.diagonalize(complex(0.0)).expect("Should diagonalize");

    // AP = PD の検証を個別の固有値方程式 Av_i = λ_i v_i として行う
    let p_eval = p.evaluate_all(complex(0.0));
    let d_eval = d.evaluate_all(complex(0.0));
    let m_eval = m.evaluate_all(complex(0.0));

    let n = 3;
    for j in 0..n {
        let lambda = d_eval[j][j];
        let mut v = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n { v[i] = p_eval[i][j]; }

        // A * v
        let mut av = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            for k in 0..n {
                av[i] += m_eval[i][k] * v[k];
            }
        }

        // λ * v
        let mut lv = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            lv[i] = lambda * v[i];
        }

        for i in 0..n {
            // 誤差の許容範囲を少し広めに設定 (1e-3)
            assert!((av[i].re - lv[i].re).abs() < 1e-3, "Failed at column {} row {}: av={} vs lv={}", j, i, av[i].re, lv[i].re);
        }
    }
}
