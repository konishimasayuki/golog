// api/score.js — スコア保存・取得（Upstash Redis）
// Vercel環境変数に UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN を設定してください。
// 使うには package.json に "@upstash/redis" を追加します。

import { Redis } from "@upstash/redis";

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});

export default async function handler(req, res) {
  const userId = req.query.userId || "demo";

  if (req.method === "POST") {
    const round = req.body; // { date, course, score, putt, holes:[...] }
    const id = `round:${userId}:${Date.now()}`;
    await redis.set(id, JSON.stringify(round));
    await redis.lpush(`rounds:${userId}`, id);
    return res.status(200).json({ ok: true, id });
  }

  if (req.method === "GET") {
    const ids = await redis.lrange(`rounds:${userId}`, 0, 30);
    const rounds = await Promise.all(ids.map((i) => redis.get(i)));
    return res.status(200).json({
      rounds: rounds.map((r) => (typeof r === "string" ? JSON.parse(r) : r)),
    });
  }

  res.status(405).json({ error: "Method not allowed" });
}
