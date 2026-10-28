use eml_ope::domain::eml::EmlTree;
use eml_ope::domain::vec::{Matrix, Vector};
use num_complex::Complex64;
use std::io::{self, Write};

fn main() {
    println!("=== EML Operator Calculator (arXiv:2603.21852v2) ===");
    println!("1 と eml(x, y) = exp(x) - ln(y) のみで構成された計算機");
    println!("-----------------------------------------------------");

    loop {
        println!("\n計算メニュー:");
        println!("1: exp(1) の計算 [e の導出]");
        println!("2: sin(pi/2) の計算 [三角関数の再構成]");
        println!("3: 2^10 の計算 [累乗の構成]");
        println!("4: 行列演算のデモ [[1, 2], [3, 4]] * [1, 2]");
        println!("5: 数式発見（Symbolic Regression）のデモ [f(x) = 3x + 5 の学習]");
        println!("q: 終了");
        print!("選択してください > ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        let choice = choice.trim();

        match choice {
            "1" => {
                let tree = EmlTree::e();
                let res = tree.eval(Complex64::new(0.0, 0.0), &[]);
                println!("結果: e ≈ {:.10} (re: {}, im: {})", res.re, res.re, res.im);
            }
            "2" => {
                let half_pi = EmlTree::div(EmlTree::pi(), EmlTree::two());
                let tree = EmlTree::sin(half_pi);
                let res = tree.eval(Complex64::new(0.0, 0.0), &[]);
                println!("結果: sin(pi/2) ≈ {:.10} (re: {}, im: {})", res.re, res.re, res.im);
            }
            "3" => {
                // 2^10
                let ten = (0..10).fold(EmlTree::zero(), |acc, _| EmlTree::add(acc, EmlTree::One));
                let tree = EmlTree::pow(EmlTree::two(), ten);
                let res = tree.eval(Complex64::new(0.0, 0.0), &[]);
                println!("結果: 2^10 = {:.1} (re: {}, im: {})", res.re, res.re, res.im);
            }
            "4" => {
                println!("行列演算の構成中...");
                // Matrix [[1, 2], [3, 4]]
                let m = Matrix::new(vec![
                    vec![EmlTree::One, EmlTree::two()],
                    vec![EmlTree::add(EmlTree::two(), EmlTree::One), EmlTree::add(EmlTree::two(), EmlTree::two())],
                ]);
                // Vector [1, 2]
                let v = Vector::new(vec![EmlTree::One, EmlTree::two()]);

                let res = m.mul_vec(&v);

                // 1*1 + 2*2 = 5
                // 3*1 + 4*2 = 11
                let res_0 = res.elements[0].eval(Complex64::new(0.0, 0.0), &[]);
                let res_1 = res.elements[1].eval(Complex64::new(0.0, 0.0), &[]);

                println!("結果ベクトル:");
                println!("  [ {:.1} ]", res_0.re);
                println!("  [ {:.1} ]", res_1.re);
            }
            "5" => {
                println!("--- Symbolic Regression Demo ---");
                println!("ターゲット関数: f(x) = 3x + 5");
                println!("学習モデル: p0 * x + p1");

                // データ生成
                let mut data = Vec::new();
                for i in 0..10 {
                    let x = i as f64;
                    let y = 3.0 * x + 5.0;
                    data.push((Complex64::new(x, 0.0), Complex64::new(y, 0.0)));
                }

                // モデル構成: p0 * x + p1
                let model = EmlTree::add(
                    EmlTree::mul(EmlTree::Param(0), EmlTree::InputX),
                    EmlTree::Param(1)
                );

                // 初期パラメータ
                let mut params = vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
                let learning_rate = 0.01;
                let epochs = 200;

                println!("学習開始 (LR={}, Epochs={})...", learning_rate, epochs);

                for epoch in 0..=epochs {
                    let mut total_loss = 0.0;
                    let mut grad_sum = vec![Complex64::new(0.0, 0.0); 2];

                    for (x, y_target) in &data {
                        let (y_pred, _, grad_p) = model.eval_full_gradients(*x, &params);
                        let diff = y_pred - y_target;
                        total_loss += diff.norm_sqr();

                        // MSEの勾配: d/dp 1/N * sum((y_pred - y_target)^2) = 2/N * (y_pred - y_target) * dy_pred/dp
                        for i in 0..2 {
                            grad_sum[i] += diff * grad_p[i];
                        }
                    }

                    // パラメータ更新
                    for i in 0..2 {
                        params[i] -= Complex64::new(learning_rate * 2.0 / data.len() as f64, 0.0) * grad_sum[i];
                    }

                    if epoch % 50 == 0 {
                        println!("Epoch {:3}: Loss = {:.6}, params = [p0: {:.4}, p1: {:.4}]",
                                 epoch, total_loss / data.len() as f64, params[0].re, params[1].re);
                    }
                }

                println!("学習完了！ final f(x) ≈ {:.2}x + {:.2}", params[0].re, params[1].re);
            }
            "q" => break,
            _ => println!("無効な選択です。"),
        }
    }
}
