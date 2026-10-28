use eml_ope::domain::eml::{eml, eml_complex, Eml, EmlTree};
use num_complex::Complex64;
use std::f64::consts::{E, PI};

fn complex(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

#[test]
fn test_eml_basic() {
    // f64: eml(1.0, 1.0) = exp(1.0) - ln(1.0) = e - 0 = e
    assert!((eml(1.0, 1.0) - E).abs() < 1e-10);

    // Complex64: eml_complex(1.0, 1.0) = e
    let c1 = Complex64::new(1.0, 0.0);
    let res = eml_complex(c1, c1);
    assert!((res.re - E).abs() < 1e-10);
    assert!(res.im.abs() < 1e-10);

    // Eml trait implementation
    let trait_res = c1.eml(c1);
    assert!((trait_res.re - E).abs() < 1e-10);
    assert!(trait_res.im.abs() < 1e-10);

    let trait_f64 = 1.0.eml(1.0);
    assert!((trait_f64 - E).abs() < 1e-10);
}

#[test]
fn test_constants() {
    // e
    assert!((EmlTree::e().eval(complex(0.0), &[]).re - E).abs() < 1e-10);

    // pi
    let pi_val = EmlTree::pi().eval(complex(0.0), &[]);
    assert!((pi_val.re - PI).abs() < 1e-10);

    // i
    let i_val = EmlTree::i().eval(complex(0.0), &[]);
    assert!(i_val.re.abs() < 1e-10);
    assert!((i_val.im - 1.0).abs() < 1e-10);
}

#[test]
fn test_pow_sqrt() {
    // 2^3 = 8
    let tree = EmlTree::pow(EmlTree::two(), EmlTree::add(EmlTree::two(), EmlTree::One));
    assert!((tree.eval(complex(0.0), &[]).re - 8.0).abs() < 1e-10);

    // sqrt(9) = 3
    assert!((EmlTree::sqrt(EmlTree::InputX).eval(complex(9.0), &[]).re - 3.0).abs() < 1e-10);
}

#[test]
fn test_trig() {
    let half_pi = EmlTree::div(EmlTree::pi(), EmlTree::two());

    // sin(pi/2) = 1
    assert!((EmlTree::sin(half_pi.clone()).eval(complex(0.0), &[]).re - 1.0).abs() < 1e-10);

    // cos(pi/2) = 0
    assert!(EmlTree::cos(half_pi).eval(complex(0.0), &[]).re.abs() < 1e-10);
}

#[test]
fn test_gradient() {
    // d/dx exp(x) = exp(x)
    let exp_tree = EmlTree::exp(EmlTree::InputX);
    let x = complex(2.0);
    let (val, grad, _) = exp_tree.eval_full_gradients(x, &[]);
    assert!((val.re - x.re.exp()).abs() < 1e-10);
    assert!((grad.re - x.re.exp()).abs() < 1e-10);

    // d/dx ln(x) = 1/x
    let ln_tree = EmlTree::ln(EmlTree::InputX);
    let x = complex(2.0);
    let (val, grad, _) = ln_tree.eval_full_gradients(x, &[]);
    assert!((val.re - x.re.ln()).abs() < 1e-10);
    assert!((grad.re - 1.0/x.re).abs() < 1e-10);

    // d/dx x^2 = 2x
    let x_sq = EmlTree::pow(EmlTree::InputX, EmlTree::two());
    let x = complex(3.0);
    let (val, grad, _) = x_sq.eval_full_gradients(x, &[]);
    println!("x=3.0, val={}, grad={}", val, grad);
    assert!((val.re - 9.0).abs() < 1e-10);
    assert!((grad.re - 6.0).abs() < 1e-10);
}

#[test]
fn test_param_gradient() {
    // f(x, p0, p1) = p0 * x + p1
    let tree = EmlTree::add(
        EmlTree::mul(EmlTree::Param(0), EmlTree::InputX),
        EmlTree::Param(1)
    );
    let x = complex(2.0);
    let params = vec![complex(3.0), complex(5.0)]; // 3x + 5
    let (val, grad_x, grad_p) = tree.eval_full_gradients(x, &params);

    assert!((val.re - 11.0).abs() < 1e-10);
    assert!((grad_x.re - 3.0).abs() < 1e-10); // df/dx = p0 = 3
    assert!((grad_p[0].re - 2.0).abs() < 1e-10); // df/dp0 = x = 2
    assert!((grad_p[1].re - 1.0).abs() < 1e-10); // df/dp1 = 1
}
