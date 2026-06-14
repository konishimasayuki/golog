# Golog（ゴーログ）

GPS距離・スイング解析・スコア管理ができるゴルフ統合アプリ。
React + Vite + Vercel + Upstash Redis（いつものスタック）。

## ログイン

- 一般ユーザー: `a` / `a`
- 管理者: `b` / `b`

## デプロイ手順（GitHub Web → Vercel）

1. このフォルダ一式を GitHub リポジトリにアップロード（`konishimasayuki/golog` など）
2. Vercel で New Project → 該当リポジトリを Import
3. Framework Preset は **Vite** が自動検出される
4. Deploy 押すだけ（ビルドコマンド `npm run build`、出力 `dist` は自動）

## 機能

- **ラウンド**: ラウンド開始 → 設定（日付/天気/コース/同伴者）→ 記録中（GPS自動ON・残り距離・スコア入力）
- **練習**: スイング撮影・AI解析・骨格表示・スロー再生・描画・2画面比較
- **練習記録**: スイングスコア推移・カレンダー・解析履歴
- **コーチ**: AIチャット相談（Claude API）・お手本
- **設定**: プロフィール・クラブ・通知・ログアウト
- **管理画面（b）**: ゴルフ場登録（各ホールPAR/ヤード）・グリーン位置ピン留め・ユーザー管理

## 次の実装（実機/Vercelで動かす分）

### 1. Upstash Redis 接続
`api/score.js` は用意済み。有効化するには:
- `package.json` の dependencies に `"@upstash/redis": "^1.34.0"` を追加
- Vercel の環境変数に `UPSTASH_REDIS_REST_URL` と `UPSTASH_REDIS_REST_TOKEN` を設定
- App.jsx 側で fetch("/api/score") を呼ぶように差し替え（現状はセッション内保存）

### 2. MediaPipe（実写骨格）
`IMPLEMENTATION-GUIDE.md` 参照。`@mediapipe/tasks-vision` を追加し、
練習タブの SwingStage を実カメラ + 骨格描画に置き換える。

### 3. コース地図を実Leaflet+OpenStreetMapに
管理画面のグリーン登録の疑似衛星写真を、実際の地図タイルに差し替え。

## 注意

- ログイン認証は現状フロントのみのデモ（a/a・b/b ハードコード）。
  本番は管理画面で登録したユーザー（ID/PW）を Redis 照合に変更すること。
- GPS・カメラは HTTPS 必須（Vercelは自動HTTPS）。iOS Safari はユーザー操作起点で起動。
