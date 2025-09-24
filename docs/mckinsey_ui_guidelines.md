# McKinsey風UIルック＆フィール規範

経営コンサルタントが安心して使える「知性・簡潔・信頼」を訴求するために、配色・タイポグラフィ・レイアウト・コンポーネントのガイドラインを以下にまとめる。

## 1. デザイン・トークン

| カテゴリー | トークン名 | 値・定義 | 用途 |
| --- | --- | --- | --- |
| カラー | primaryColor | #0B1F3B（濃紺） | ブランドの核となる色。KPIカードの数値やアクションボタンに使用。 |
|  | secondaryColor | #5A6B7A（ミディアムグレー） | 補助的なテキストや境界線、二次アクションに使用。 |
|  | accentColor | #1E88E5（ブルーアクセント） | ハイライトや選択状態の強調。棒グラフの主要系列に使用。 |
|  | backgroundColor | #F7F8FA（オフホワイト） | ダッシュボード全体の背景色。 |
|  | cardBackground | #FFFFFF（白） | カード・テーブル・フォームの背景。 |
|  | successColor | #2E7D32（グリーン、彩度20%減） | 前期比がプラスのときの矢印・数値。 |
|  | warningColor | #FFB300（アンバー、彩度20%減） | 閾値超過の警告、欠品アラート。 |
|  | errorColor | #C62828（レッド、彩度20%減） | 赤字やエラー時の強調。 |
| フォント | baseFontFamily | "Inter", "Source Sans 3", sans-serif | 本文・数字ともに読みやすさを重視。数字には等幅フォントを使用。 |
|  | headingSize | 22–28px | H1、H2に適用。 |
|  | bodySize | 14–16px | 本文・テーブルセルに適用。行間は1.4–1.6。 |
|  | metricSize | 28–32px | KPIカードの数値。 |
| レイアウト | gridColumns | 12カラム相当 | Streamlitの`st.columns()`で `[3,1]` 等の比率で主従構造を作る。 |
|  | borderRadius | 8–12px | カード・入力ボックスの角丸。 |
|  | shadow | 0px 2px 4px rgba(0,0,0,0.06) | カードにごく薄いドロップシャドウを付与。 |
|  | spacingUnit | 8px | 余白の基本単位。16px、24pxなど倍数で統一。 |

## 2. Streamlitテーマ設定例（.streamlit/config.toml）

```toml
[theme]
primaryColor = "#0B1F3B"
backgroundColor = "#F7F8FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1A1A1A"
font = "sans serif" # Inter/Source Sans 3 を含む
accentColor = "#1E88E5"

[server]
headless = true
enableXsrfProtection = true
```

- `secondaryBackgroundColor` はカードやテーブルの背景。
- `textColor` は基本文字色（ほぼ黒 #1A1A1A）。
- `accentColor` はリンクや選択されたタブに反映され、ユーザーの視線を誘導する。
- カラーバリアフリーを考慮し、彩度・明度差が一定となるよう調整する。

## 3. コンポーネントのスタイリング

- **KPIカード**: `st.metric()` を独自ラップし、カード背景に`cardBackground`、角丸8px、内側余白16pxを設定。`delta_color`オプションでプラス→`successColor`、マイナス→`errorColor` を自動適用。
- **セグメントコントロール**: 期間や店舗の切替には`st.radio()`や`st.tabs()`を利用。選択状態は`accentColor`で強調し、未選択は`secondaryColor`で落ち着いた印象に。
- **トグル／チェックボックス**: 色味を抑えたグレー基調（ON時にアクセント色が点灯）。説明文には`st.tooltip()`を追加し、専門用語の補足を表示。
- **テーブル**: `st.dataframe()`の`column_config`で桁区切り（`format="*,d"`）、単位（`suffix=" 円"`）、条件付き書式（例：粗利率<0.3なら背景を薄赤）を設定。ヘッダーはセンター揃え、縞模様行（striped）を採用し可読性を高める。
- **グラフ**: Plotlyベースでタイトルを必ず付け、系列色は`primaryColor`と`accentColor`を中心に使用。凡例は右上、縦グリッド線は薄グレーに統一。円グラフは極力避け、棒・折れ線・面グラフを用いる。

## 4. レスポンシブ設計と視線誘導

- `layout="wide"` をデフォルトとし、PCでは3カラム、モバイルでは1カラムへ自動落とし込む。`st.columns()` のブレークポイントは `[6,6]→[12]` のように設定。
- 重要指標は左上から右下へと情報密度が高くなるよう配置。例：売上→利益→在庫→シミュレーションの順で下方向へ遷移させ、視線の流れを自然にする。
- フォーカスすべき数値は文字サイズ・太字で強調し、色だけに頼らないデザインとする（アクセシビリティ対応）。

## 5. 効果見込み（フェルミ）

- 統一感のある色・フォントにより、認知負荷を約30％削減。ユーザーがどこを見れば良いか迷う時間を平均5秒短縮。
- 条件付き書式やカラートークンで異常値が即座に視覚化され、誤判断率を現状比50％以上低減。
- 簡潔なカード・余白設計により、モバイル閲覧時のスクロール回数を1画面平均4回→2回に削減。
