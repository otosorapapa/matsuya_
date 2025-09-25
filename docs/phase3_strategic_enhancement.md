# Phase 3 長期的改善ロードマップ

## 1. 現状診断
- 過去データの可視化が中心で、将来予測やシナリオ比較は限定的。
- ユーザーごとのカスタマイズ、複数店舗間のベンチマーク、経営戦略サポート機能が不足している。

## 2. 課題抽出
| 観点 | 課題 |
| --- | --- |
| 実用性 | 売上予測・需要予測・最適発注量など経営判断を支援する機能が存在しない。 |
| ユーザーフレンドリー性 | シナリオ比較やシミュレーション結果を保存・共有する仕組みが未実装。 |
| デザイン性 | モバイル対応・レスポンシブデザインが不足し、利用シーンが限定されている。 |
| 操作負荷 | 複数店舗データの横断分析やKPIダッシュボード構築の集計が手作業で負担。 |
| 実装可能性 | 外部サービス連携や機械学習を組み込むための設計変更が必要。 |

## 3. 改善提案
1. **需要予測・発注最適化モデルの組み込み**
   - 過去の売上、天候、イベント情報を取り込み、SARIMA や Meta Prophet などの時系列モデルで需要を予測。
   - 予測結果を EOQ（Economic Order Quantity）モデルに入力して最適発注量を算出。
   - Streamlit のバックエンドに `scikit-learn` や `prophet` を追加し、`analytics/forecast.py`（新規）などでモデルを提供する。
2. **KPI ベンチマーク機能**
   - 複数店舗の売上高・粗利率・在庫回転率などをレーダーチャートやランキングで比較。
   - 低パフォーマンス店舗の改善ポイントをヒートマップや条件付きコメントとして提示。
3. **シナリオ管理・共有**
   - 資金シミュレーションや発注計画をユーザー単位で保存し、名称を付けて一覧化。
   - `st.experimental_set_query_params` を利用して URL にシナリオパラメータを含め、共有リンクを発行する。
4. **データ統合基盤との連携**
   - POS・会計・在庫システムと API 連携する「データ自動更新モード」を実装。
   - 例: Google Cloud Functions を経由して BigQuery からデータを取得し、日次で更新。
5. **ユーザー権限とログ管理**
   - 閲覧可能店舗や機能をロールごとに制御。
   - シナリオ作成・更新操作を監査ログとして記録し、Auth0 や `st.secrets` を活用した認証基盤を整備。
6. **デザイン高度化**
   - `streamlit-extras` や Bootstrap ベースのテーマを採用し、レスポンシブデザインに対応。
   - モバイル・タブレットでも KPI カードやグラフが最適なレイアウトで表示されるよう調整。

## 4. 実装サンプル
### 4.1 需要予測の例 (Prophet)
```python
from prophet import Prophet


def forecast_demand(df_sales, periods: int = 30):
    m = Prophet()
    m.fit(df_sales.rename(columns={"date": "ds", "sales": "y"}))
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


forecast_df = forecast_demand(df_sales)
st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
```

### 4.2 シナリオ保存と URL 共有
```python
import json
import urllib.parse

scenario = {
    "gross_margin": gross_margin,
    "fixed_cost": fixed_cost,
    "target_profit": target_profit,
}
encoded = urllib.parse.quote(json.dumps(scenario))
share_url = f"{st.get_url()}?scenario={encoded}"
st.button(
    "シェア用URLコピー",
    on_click=st.copy_to_clipboard,
    args=(share_url,),
)
```

## 5. 成長ストーリー
- Phase 1 で表示エラーや UI/UX のバグを解消し、基礎的な指標を安定的に閲覧できる状態を整える。
- Phase 2 で分析ロジックとユーザー導線を再設計し、迷わない操作と KPI 集約を実現する。
- Phase 3 では需要予測、シナリオ管理、API 連携、レスポンシブデザインなどを拡張し、経営コンサルティングのフレームワークにも耐える高度な意思決定支援ツールへ進化させる。

長期的には、予測モデルやシナリオ管理、外部データ連携を統合し、松屋の経営判断を継続的に後押しする戦略ダッシュボードを目指す。
