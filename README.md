# 松屋 計数管理アプリ設計リポジトリ

本リポジトリでは、株式会社松屋向けに構想したStreamlitベースの「計数管理」Webアプリの設計書およびプロトタイプ実装を管理しています。戦略バランスト・スコアカード (BSC)、経営シミュレーション、AI レコメンド、権限管理など Phase 3 向けの高度化プランも含め、詳細な UI/UX、データ構造、実装モジュール案は下記ドキュメントを参照してください。

- [松屋向け「計数管理」Streamlitアプリ設計書](docs/streamlit_design.md)
- [Phase 2 情報設計・導線再設計ガイド](docs/phase2_information_architecture.md)
- [Phase 3 長期的改善ロードマップ](docs/phase3_strategic_enhancement.md)

## Streamlitアプリ概要

`streamlit_app/` 配下に設計書をもとにしたダッシュボードアプリを実装しました。売上分析・商品別分析・損益管理・在庫分析・経営シミュレーションの5タブ構成で、サイドバーからフィルタや帳票出力操作を行えます。`assets/sample_data/` には動作確認用のサンプルCSVを同梱しています。

### 動かし方

1. 依存ライブラリをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

2. Streamlitでアプリを起動します。

   ```bash
   streamlit run streamlit_app/app.py
   ```

3. ブラウザで表示されたアプリ上で、サイドバーのサンプルデータを利用するか独自CSVをアップロードして指標を確認します。

帳票出力（CSV/PDF）はサイドバーのチェックボックスでON/OFFを切り替え、各集計結果を即時ダウンロードできます。今後、プロトタイプ実装や運用マニュアルなどの成果物もここに蓄積していく予定です。
