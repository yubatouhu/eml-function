# Project Overview: eml_ope

関数 `eml(x,y) = exp(x) - ln(y)` と定数 `1` のみを用いた、すべての初等関数の再帰的表現と、それを利用した数式発見（Symbolic Regression）を Rust で実現するプロジェクト。

## Core Mandates & Principles

- **Mathematical Integrity**: `eml(x,y) = exp(x) - ln(y)` の定義と、論文 (arXiv:2603.21852v2) に基づく数学的性質を厳密に扱う。
- **Learning Priority**: 実装だけでなく、数理科学的背景（文脈自由文法としての数式など）を理解し、学習プロセスを重視する。
- **Language**: 全ての対話とドキュメントは日本語で行う。
- **Knowledge Base**: 習得した知識は `obsidian\valt` ディレクトリに整理して蓄積する。

## Technical Stack

- **Language**: Rust (Edition 2024)
- **Build Tool**: Cargo
- **Scientific Computing**: `num-complex` を利用した複素数平面での計算。
- **Structure**: `S -> 1 | x | eml(S, S)` という文法に基づくバイナリツリー構造。

## Directory Structure

- `src/lib.rs`: ライブラリのエントリポイント。
- `src/main.rs`: 実行バイナリのエントリポイント。
- `src/domain/`: `eml` 関数のコアロジック。
    - `eml.rs`: `eml` 演算（実数・複素数、`Eml` トレイト）および `EmlTree` 列挙型による再帰的表現と評価。
    - `vec.rs`: `EmlTree` を要素に持つ行列 (`Matrix`) とベクトル (`Vector`) の演算。
- `tests/`: 統合テスト。
    - `eml_tests.rs`: `eml` 演算、`EmlTree` と各初等関数の数学的検証。
    - `vec_tests.rs`: 行列・ベクトル演算の検証。
- `obsidian/valt/`: 論文の要約、数学的証明、Rust 実装の知見。

## Building and Running

- **ビルド**: `cargo build`
- **実行**: `cargo run`
- **テスト**: `cargo test`

## Development Workflow

1.  **Research & Plan**: 論文に基づく `eml()` を用いた初等関数の導出アルゴリズムを整理する。
2.  **Implementation**: `EmlTree` 構造の実装と、行列演算への拡張。
3.  **Documentation**: 数理的背景を Obsidian に記録し、`GEMINI.md` を更新。
4.  **Optimization**: 勾配降下法による重み最適化（Symbolic Regression）の検討。

## Current Status
- **ライブラリ基盤構築完了**: `lib.rs` を通じたモジュール公開。
- **EmlTree 構成完了**: `1` と `eml` のみで構成される科学用計算機機能の実装。
- **行列演算の実装完了**: `EmlTree` を要素に持つ行列の積、行列とベクトルの積のサポート。
- **勾配計算（自動微分）の実装完了**: 変数 $x$ に関する微分の再帰的計算をサポート。
- **テストの充実**: 各初等関数および行列演算、勾配計算の統合テストを完備。
