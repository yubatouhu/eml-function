use num_complex::Complex64;

/// eml(x, y) = exp(x) - ln(y) を計算する基本関数（実数）
pub fn eml(x: f64, y: f64) -> f64 {
    x.exp() - y.ln()
}

/// 複素数に対する eml(x, y) = exp(x) - ln(y)
pub fn eml_complex(x: Complex64, y: Complex64) -> Complex64 {
    x.exp() - y.ln()
}

/// eml 演算を統一的に扱うトレイト
pub trait Eml<Rhs = Self> {
    type Output;
    fn eml(self, rhs: Rhs) -> Self::Output;
}

impl Eml for f64 {
    type Output = f64;
    fn eml(self, rhs: f64) -> f64 {
        eml(self, rhs)
    }
}

impl Eml for Complex64 {
    type Output = Complex64;
    fn eml(self, rhs: Complex64) -> Complex64 {
        eml_complex(self, rhs)
    }
}

impl Eml for EmlTree {
    type Output = EmlTree;
    fn eml(self, rhs: EmlTree) -> EmlTree {
        EmlTree::Node(Box::new(self), Box::new(rhs))
    }
}

/// EMLツリーの構成要素: S -> 1 | x | c | p_i | eml(S, S) | S+S | S-S | S*S | S/S
#[derive(Debug, Clone)]
pub enum EmlTree {
    One,                               // 定数 1
    InputX,                            // 変数 x
    Value(Complex64),                  // 固定定数
    Param(usize),                      // 最適化対象のパラメータ p_i
    Node(Box<EmlTree>, Box<EmlTree>),  // eml(left, right) = exp(left) - ln(right)
    Add(Box<EmlTree>, Box<EmlTree>),
    Sub(Box<EmlTree>, Box<EmlTree>),
    Mul(Box<EmlTree>, Box<EmlTree>),
    Div(Box<EmlTree>, Box<EmlTree>),
}

impl EmlTree {
    /// ツリーを再帰的に評価する
    pub fn eval(&self, x: Complex64, params: &[Complex64]) -> Complex64 {
        match self {
            EmlTree::One => Complex64::new(1.0, 0.0),
            EmlTree::InputX => x,
            EmlTree::Value(c) => *c,
            EmlTree::Param(i) => params[*i],
            EmlTree::Node(left, right) => {
                let eval_left = left.eval(x, params);
                let eval_right = right.eval(x, params);
                eval_left.exp() - eval_right.ln()
            }
            EmlTree::Add(left, right) => left.eval(x, params) + right.eval(x, params),
            EmlTree::Sub(left, right) => left.eval(x, params) - right.eval(x, params),
            EmlTree::Mul(left, right) => left.eval(x, params) * right.eval(x, params),
            EmlTree::Div(left, right) => left.eval(x, params) / right.eval(x, params),
        }
    }

    /// 変数 x に関する微分値と、各パラメータに関する勾配を同時に計算する
    /// 戻り値: (評価値, xに関する勾配, パラメータに関する勾配ベクトル)
    pub fn eval_full_gradients(
        &self,
        x: Complex64,
        params: &[Complex64],
    ) -> (Complex64, Complex64, Vec<Complex64>) {
        let n = params.len();
        match self {
            EmlTree::One => (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), vec![Complex64::new(0.0, 0.0); n]),
            EmlTree::InputX => (x, Complex64::new(1.0, 0.0), vec![Complex64::new(0.0, 0.0); n]),
            EmlTree::Value(c) => (*c, Complex64::new(0.0, 0.0), vec![Complex64::new(0.0, 0.0); n]),
            EmlTree::Param(i) => {
                let mut p_grads = vec![Complex64::new(0.0, 0.0); n];
                p_grads[*i] = Complex64::new(1.0, 0.0);
                (params[*i], Complex64::new(0.0, 0.0), p_grads)
            }
            EmlTree::Node(left, right) => {
                let (val_l, grad_x_l, grad_p_l) = left.eval_full_gradients(x, params);
                let (val_r, grad_x_r, grad_p_r) = right.eval_full_gradients(x, params);

                let val = val_l.exp() - val_r.ln();
                let grad_x = val_l.exp() * grad_x_l - grad_x_r / val_r;
                let grad_p = grad_p_l.iter().zip(grad_p_r.iter())
                    .map(|(gl, gr)| val_l.exp() * gl - gr / val_r)
                    .collect();
                (val, grad_x, grad_p)
            }
            EmlTree::Add(left, right) => {
                let (val_l, grad_x_l, grad_p_l) = left.eval_full_gradients(x, params);
                let (val_r, grad_x_r, grad_p_r) = right.eval_full_gradients(x, params);
                let grad_p = grad_p_l.iter().zip(grad_p_r.iter()).map(|(l, r)| l + r).collect();
                (val_l + val_r, grad_x_l + grad_x_r, grad_p)
            }
            EmlTree::Sub(left, right) => {
                let (val_l, grad_x_l, grad_p_l) = left.eval_full_gradients(x, params);
                let (val_r, grad_x_r, grad_p_r) = right.eval_full_gradients(x, params);
                let grad_p = grad_p_l.iter().zip(grad_p_r.iter()).map(|(l, r)| l - r).collect();
                (val_l - val_r, grad_x_l - grad_x_r, grad_p)
            }
            EmlTree::Mul(left, right) => {
                let (val_l, grad_x_l, grad_p_l) = left.eval_full_gradients(x, params);
                let (val_r, grad_x_r, grad_p_r) = right.eval_full_gradients(x, params);
                let grad_x = grad_x_l * val_r + val_l * grad_x_r;
                let grad_p = grad_p_l.iter().zip(grad_p_r.iter())
                    .map(|(gl, gr)| gl * val_r + val_l * gr)
                    .collect();
                (val_l * val_r, grad_x, grad_p)
            }
            EmlTree::Div(left, right) => {
                let (val_l, grad_x_l, grad_p_l) = left.eval_full_gradients(x, params);
                let (val_r, grad_x_r, grad_p_r) = right.eval_full_gradients(x, params);
                let den = val_r * val_r;
                let grad_x = (grad_x_l * val_r - val_l * grad_x_r) / den;
                let grad_p = grad_p_l.iter().zip(grad_p_r.iter())
                    .map(|(gl, gr)| (gl * val_r - val_l * gr) / den)
                    .collect();
                (val_l / val_r, grad_x, grad_p)
            }
        }
    }

    // --- 基本部品のビルダ ---

    /// eml(left, right) = Node(left, right)
    pub fn eml(left: EmlTree, right: EmlTree) -> Self {
        EmlTree::Node(Box::new(left), Box::new(right))
    }

    /// exp(x) = eml(x, 1)
    pub fn exp(inner: EmlTree) -> Self {
        EmlTree::Node(Box::new(inner), Box::new(EmlTree::One))
    }

    /// ln(x) = eml(1, eml(eml(1, x), 1))
    /// 数値的安定性のために直接 eml を使わず、理論的定義として保持しつつ計算することも可能だが、
    /// ここでは eml(1, x) = e - ln(x) なので、純粋に eml 表現に拘るなら Node を使う。
    /// ただし ln(x) は基本演算なので、専用バリアントにしても良い。
    /// 今回は Node を維持し、必要に応じて ln バリアントを追加検討する。
    pub fn ln(inner: EmlTree) -> Self {
        EmlTree::Node(
            Box::new(EmlTree::One),
            Box::new(EmlTree::Node(
                Box::new(EmlTree::Node(Box::new(EmlTree::One), Box::new(inner))),
                Box::new(EmlTree::One),
            )),
        )
    }

    pub fn sub(x: EmlTree, y: EmlTree) -> Self {
        EmlTree::Sub(Box::new(x), Box::new(y))
    }

    pub fn add(x: EmlTree, y: EmlTree) -> Self {
        EmlTree::Add(Box::new(x), Box::new(y))
    }

    pub fn mul(x: EmlTree, y: EmlTree) -> Self {
        EmlTree::Mul(Box::new(x), Box::new(y))
    }

    pub fn div(x: EmlTree, y: EmlTree) -> Self {
        EmlTree::Div(Box::new(x), Box::new(y))
    }

    pub fn zero() -> Self {
        EmlTree::Value(Complex64::new(0.0, 0.0))
    }

    pub fn neg(y: EmlTree) -> Self {
        Self::sub(Self::zero(), y)
    }

    // --- 定数のビルダ ---

    /// 2 = 1 + 1
    pub fn two() -> Self {
        EmlTree::Value(Complex64::new(2.0, 0.0))
    }

    /// e = exp(1)
    pub fn e() -> Self {
        Self::exp(EmlTree::One)
    }

    /// i = exp(ln(-1) / 2)
    pub fn i() -> Self {
        let neg_one = Self::neg(EmlTree::One);
        let i_pi = Self::ln(neg_one);
        Self::exp(Self::div(i_pi, Self::two()))
    }

    /// pi = ln(-1) / i
    pub fn pi() -> Self {
        let neg_one = Self::neg(EmlTree::One);
        let i_pi = Self::ln(neg_one);
        Self::div(i_pi, Self::i())
    }

    // --- 高度な演算子のビルダ ---

    pub fn pow(x: EmlTree, y: EmlTree) -> Self {
        Self::exp(Self::mul(y, Self::ln(x)))
    }

    pub fn sqrt(x: EmlTree) -> Self {
        Self::pow(x, Self::div(EmlTree::One, Self::two()))
    }

    pub fn log_base(base: EmlTree, x: EmlTree) -> Self {
        Self::div(Self::ln(x), Self::ln(base))
    }

    // --- 三角関数のビルダ ---

    pub fn sin(x: EmlTree) -> Self {
        let ix = Self::mul(Self::i(), x);
        let num = Self::sub(Self::exp(ix.clone()), Self::exp(Self::neg(ix)));
        let den = Self::mul(Self::two(), Self::i());
        Self::div(num, den)
    }

    pub fn cos(x: EmlTree) -> Self {
        let ix = Self::mul(Self::i(), x);
        let num = Self::add(Self::exp(ix.clone()), Self::exp(Self::neg(ix)));
        Self::div(num, Self::two())
    }

    pub fn tan(x: EmlTree) -> Self {
        Self::div(Self::sin(x.clone()), Self::cos(x))
    }

    // --- 双曲線関数のビルダ ---

    pub fn sinh(x: EmlTree) -> Self {
        let num = Self::sub(Self::exp(x.clone()), Self::exp(Self::neg(x)));
        Self::div(num, Self::two())
    }

    pub fn cosh(x: EmlTree) -> Self {
        let num = Self::add(Self::exp(x.clone()), Self::exp(Self::neg(x)));
        Self::div(num, Self::two())
    }
}
