import { useState, useRef, useEffect, useCallback } from "react";
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* ============================================================
   MediaPipe 骨格検出ユーティリティ
   ============================================================ */
const POSE_CONNECTIONS = [
  [11, 13], [13, 15], [12, 14], [14, 16],   // 腕
  [11, 12], [23, 24], [11, 23], [12, 24],   // 胴体
  [23, 25], [25, 27], [24, 26], [26, 28],   // 脚
  [27, 31], [28, 32],                       // 足
  [15, 17], [16, 18],                       // 手
];

// PoseLandmarker をシングルトンで生成
let _poseLandmarker = null;
async function getPoseLandmarker() {
  if (_poseLandmarker) return _poseLandmarker;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  _poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  return _poseLandmarker;
}

// 動画の全フレームから骨格を抽出（解析時に1度だけ実行してキャッシュ）
async function extractPoseFrames(videoEl, onProgress) {
  const lm = await getPoseLandmarker();
  const duration = videoEl.duration;
  const FPS = 30;
  const step = 1 / FPS;
  const frames = [];
  for (let t = 0; t < duration; t += step) {
    videoEl.currentTime = t;
    await new Promise((res) => { videoEl.onseeked = res; });
    const result = lm.detectForVideo(videoEl, performance.now());
    frames.push(result.landmarks?.[0] || null);
    if (onProgress) onProgress(Math.min(1, t / duration));
  }
  return frames;
}

// 3点の角度（度）
function angleAt(a, b, c) {
  if (!a || !b || !c) return null;
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const m1 = Math.hypot(v1.x, v1.y), m2 = Math.hypot(v2.x, v2.y);
  return Math.round((Math.acos(dot / (m1 * m2)) * 180) / Math.PI);
}

// 骨格データから解析値を算出
function analyzePoseFrames(frames, fps = 30) {
  const valid = frames.filter(Boolean);
  if (!valid.length) return null;
  // 手首(15/16)の最大速度 → ヘッドスピード推定
  let maxV = 0, impactIdx = 0;
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i]?.[15], b = frames[i - 1]?.[15];
    if (!a || !b) continue;
    const v = Math.hypot(a.x - b.x, a.y - b.y) * fps;
    if (v > maxV) { maxV = v; impactIdx = i; }
  }
  // 背骨の前傾（肩中点-腰中点の鉛直からの傾き）をアドレス時に測る
  const f0 = frames.find(Boolean);
  let spineAngle = null;
  if (f0) {
    const sh = { x: (f0[11].x + f0[12].x) / 2, y: (f0[11].y + f0[12].y) / 2 };
    const hp = { x: (f0[23].x + f0[24].x) / 2, y: (f0[23].y + f0[24].y) / 2 };
    spineAngle = Math.round((Math.atan2(hp.x - sh.x, hp.y - sh.y) * 180) / Math.PI);
  }
  // px/frame正規化値 → m/s推定（キャリブレーション係数は要調整）
  const CALIB = 95; // 暫定。身長基準でキャリブレーションするとより正確
  const headSpeed = +(maxV * CALIB).toFixed(1);
  const distance = Math.round(headSpeed * 4.7); // ヘッドスピード×係数
  return { headSpeed, distance, impactIdx, impactFrame: impactIdx / frames.length, spineAngle, totalFrames: frames.length };
}


/* ============================================================
   SwingLab v3 — "Atelier Green"
   アイボリー × フォレストグリーン × くすみゴールド
   セリフ見出し / モノスペース数値 / SVG線画アイコン
   ============================================================ */

const C = {
  bg: "#F0F4F7",        // 薄いグレーブルー
  card: "#FFFFFF",      // 白カード
  cardAlt: "#EAF1F6",   // 沈んだ面（薄青）
  ink: "#1A1A1A",       // 文字
  sub: "#666666",       // サブテキスト
  faint: "#9AA8B2",     // 極薄
  line: "#E2E8ED",      // ヘアライン
  green: "#2E9BE0",     // メイン青（主役・トークン名は据え置き）
  greenLite: "#5BB4E5",
  greenSoft: "#E8F4FB",
  greenDark: "#1B6FA8", // 濃い青
  gold: "#8BC34A",      // アクセント（パット等の緑）
  goldSoft: "#EAF6DD",
  red: "#E0533F",
  bone: "#7FD0F5",      // 骨格ライン（水色）
  goal: "#F5D547",      // 目標ライン黄
};

const serif = "-apple-system,'Hiragino Sans',sans-serif"; // GDO流：セリフ廃止
const mono = "ui-monospace,'SF Mono','Menlo',monospace";

/* ハバーサイン公式：2点間の距離(ヤード) */
function distanceYards(lat1, lng1, lat2, lng2) {
  const R = 6371000, toRad = (d) => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1), dLng = toRad(lng2 - lng1);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return Math.round(R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) / 0.9144);
}

/* サンプルコース（実際はRedisに保存／衛星写真からピン留め登録） */
const SAMPLE_COURSE = {
  name: "サンプルカントリークラブ",
  holes: Array.from({ length: 18 }, (_, i) => {
    const par = [4, 5, 3, 4, 4, 3, 5, 4, 4, 4, 3, 5, 4, 4, 3, 4, 5, 4][i];
    const yard = par === 3 ? 150 + i * 2 : par === 5 ? 480 + i : 360 + i * 3;
    // グリーン3点（デモ用座標。基準点から少しずらす）
    const baseLat = 35.6812 + i * 0.003, baseLng = 139.7671 + i * 0.002;
    return { no: i + 1, par, yard,
      green: { front: { lat: baseLat, lng: baseLng }, center: { lat: baseLat + 0.00012, lng: baseLng + 0.00008 }, back: { lat: baseLat + 0.00024, lng: baseLng + 0.00016 } } };
  }),
};

const CLUBS = [
  { id: "dr", s: "DR", n: "ドライバー", dist: 230 },
  { id: "w3", s: "3W", n: "3番ウッド", dist: 210 },
  { id: "u4", s: "4U", n: "ユーティリティ", dist: 190 },
  { id: "i5", s: "5I", n: "5番アイアン", dist: 175 },
  { id: "i7", s: "7I", n: "7番アイアン", dist: 150 },
  { id: "i9", s: "9I", n: "9番アイアン", dist: 125 },
  { id: "pw", s: "PW", n: "ピッチング", dist: 110 },
  { id: "sw", s: "SW", n: "サンド", dist: 80 },
];

/* ---------- SVG線画アイコン ---------- */
const Icon = ({ d, size = 22, stroke = "currentColor", sw = 1.6, fill = "none", children }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {d ? <path d={d} /> : children}
  </svg>
);
const IconCamera = (p) => <Icon {...p}><path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L17 6h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><circle cx="12" cy="12.5" r="3.5" /></Icon>;
const IconPlay = (p) => <Icon {...p}><polygon points="6 4 20 12 6 20 6 4" fill="currentColor" stroke="none" /></Icon>;
const IconChart = (p) => <Icon {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></Icon>;
const IconChat = (p) => <Icon {...p}><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.5 8.5 0 1 1 21 11.5z" /></Icon>;
const IconAward = (p) => <Icon {...p}><circle cx="12" cy="9" r="6" /><path d="M9 14.5 7.5 22 12 19.5 16.5 22 15 14.5" /></Icon>;
const IconTarget = (p) => <Icon {...p}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1.2" fill="currentColor" /></Icon>;
const IconFlame = (p) => <Icon {...p}><path d="M12 2s5 4 5 9a5 5 0 0 1-10 0c0-1.5.7-2.8 1.5-3.5C8 9 9 11 9 11s-1-4 3-9z" /></Icon>;
const IconTripod = (p) => <Icon {...p}><rect x="8.5" y="3" width="7" height="6" rx="1" /><path d="M12 9v6M12 15l-4 6M12 15l4 6M9 19h6" /></Icon>;
const IconBolt = (p) => <Icon {...p}><polygon points="13 2 4 14 11 14 10 22 19 10 12 10 13 2" /></Icon>;
const IconSound = (p) => <Icon {...p}><path d="M11 5 6 9H2v6h4l5 4z" /><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" /></Icon>;
const IconPen = (p) => <Icon {...p}><path d="M12 19l7-7-3-3-7 7-1 4z" /><path d="M16 9l3 3" /></Icon>;
const IconCompare = (p) => <Icon {...p}><rect x="3" y="4" width="7" height="16" rx="1" /><rect x="14" y="4" width="7" height="16" rx="1" /></Icon>;
const IconBack = (p) => <Icon {...p}><polyline points="15 18 9 12 15 6" /></Icon>;
const IconNext = (p) => <Icon {...p}><polyline points="9 18 15 12 9 6" /></Icon>;
const IconSkeleton = (p) => <Icon {...p}><circle cx="12" cy="4.5" r="2" /><path d="M12 6.5v6M12 8l-4 2M12 8l4 2M12 12.5l-3 5M12 12.5l3 5" /></Icon>;
const IconFlag = (p) => <Icon {...p}><path d="M5 21V4M5 4l11 3-3 4 3 4-11 2" /></Icon>;
const IconPin = (p) => <Icon {...p}><path d="M12 21s7-6.5 7-12a7 7 0 0 0-14 0c0 5.5 7 12 7 12z" /><circle cx="12" cy="9" r="2.5" /></Icon>;
const IconPlus = (p) => <Icon {...p}><path d="M12 5v14M5 12h14" /></Icon>;
const IconMinus = (p) => <Icon {...p}><path d="M5 12h14" /></Icon>;
const IconCheck = (p) => <Icon {...p}><polyline points="4 12 9 17 20 6" /></Icon>;
const IconLayers = (p) => <Icon {...p}><polygon points="12 2 22 8.5 12 15 2 8.5 12 2" /><polyline points="2 15.5 12 22 22 15.5" /></Icon>;
const IconGear = (p) => <Icon {...p}><circle cx="12" cy="12" r="3" /><path d="M12 2v3M12 19v3M5 5l2 2M17 17l2 2M2 12h3M19 12h3M5 19l2-2M17 7l2-2" /></Icon>;
const IconUsers = (p) => <Icon {...p}><circle cx="9" cy="8" r="3.5" /><path d="M3 20a6 6 0 0 1 12 0" /><path d="M16 5a3 3 0 0 1 0 6M18 20a6 6 0 0 0-3-5" /></Icon>;
const IconCourse = (p) => <Icon {...p}><path d="M5 21V4M5 4l11 3-3 4 3 4-11 2" /><circle cx="5" cy="21" r="1.5" fill="currentColor" stroke="none" /></Icon>;
const IconLogout = (p) => <Icon {...p}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" /></Icon>;
const IconTrash = (p) => <Icon {...p}><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" /></Icon>;

const S = {
  app: { minHeight: "100vh", background: C.bg, color: C.ink,
    fontFamily: "-apple-system,'Hiragino Sans',sans-serif", maxWidth: 440, margin: "0 auto",
    position: "relative", paddingBottom: 90, overflowX: "hidden" },
  topbar: { position: "sticky", top: 0, zIndex: 50, background: "rgba(255,255,255,0.88)",
    backdropFilter: "blur(14px)", borderBottom: `1px solid ${C.line}`, padding: "16px 20px",
    display: "flex", alignItems: "center", justifyContent: "space-between" },
  content: { padding: 20 },
  card: { background: C.card, borderRadius: 20, padding: 20, marginBottom: 16,
    boxShadow: "0 1px 2px rgba(42,38,32,0.04), 0 8px 24px rgba(42,38,32,0.04)" },
  // セクション見出し（セリフ + 細いラベル）
  eyebrow: { fontSize: 11, fontWeight: 600, letterSpacing: "2px", textTransform: "uppercase", color: C.gold },
  heading: { fontFamily: serif, fontSize: 20, fontWeight: 600, color: C.ink, letterSpacing: "-0.2px" },
  nav: { position: "fixed", bottom: 0, left: "50%", transform: "translateX(-50%)", width: "100%",
    maxWidth: 440, background: "rgba(251,249,244,0.94)", backdropFilter: "blur(14px)",
    borderTop: `1px solid ${C.line}`, display: "flex", justifyContent: "space-around",
    padding: "9px 0 calc(9px + env(safe-area-inset-bottom))", zIndex: 50 },
  navItem: (on) => ({ display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
    color: on ? C.green : C.faint, fontSize: 10, fontWeight: on ? 700 : 500, cursor: "pointer", flex: 1, transition: "color .2s" }),
};

/* ====== 骨格オーバーレイ（繊細版） ====== */
function Skeleton({ frame, color = C.bone }) {
  const t = frame / 100;
  let arm; if (t < 0.5) arm = 80 - t * 2 * 200; else arm = -120 + (t - 0.5) * 2 * 300;
  const sx = 50, sy = 38, hipx = 50, hipy = 58, handLen = 22;
  const hx = sx + Math.cos(arm * Math.PI / 180) * handLen, hy = sy - Math.sin(arm * Math.PI / 180) * handLen;
  const joints = [{ x: sx, y: sy }, { x: hipx, y: hipy }, { x: sx - 9, y: sy }, { x: sx + 9, y: sy }, { x: hipx - 8, y: 88 }, { x: hipx + 8, y: 88 }, { x: hx, y: hy }, { x: sx, y: sy - 12 }];
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
      <g opacity="0.92">
        <circle cx={sx} cy={sy - 12} r="5" fill="none" stroke={color} strokeWidth="0.9" />
        <line x1={sx} y1={sy} x2={hipx} y2={hipy} stroke={color} strokeWidth="1" />
        <line x1={sx - 9} y1={sy} x2={sx + 9} y2={sy} stroke={color} strokeWidth="1" />
        <line x1={hipx - 7} y1={hipy} x2={hipx + 7} y2={hipy} stroke={color} strokeWidth="1" />
        <line x1={hipx} y1={hipy} x2={hipx - 8} y2={88} stroke={color} strokeWidth="1" />
        <line x1={hipx} y1={hipy} x2={hipx + 8} y2={88} stroke={color} strokeWidth="1" />
        <line x1={hipx - 8} y1={88} x2={hipx - 11} y2={96} stroke={color} strokeWidth="1" />
        <line x1={hipx + 8} y1={88} x2={hipx + 11} y2={96} stroke={color} strokeWidth="1" />
        <line x1={sx + 9} y1={sy} x2={hx} y2={hy} stroke={color} strokeWidth="1" />
        <line x1={hx} y1={hy} x2={hx + Math.cos(arm * Math.PI / 180) * 18} y2={hy - Math.sin(arm * Math.PI / 180) * 18} stroke={C.gold} strokeWidth="0.8" strokeDasharray="2,1.5" />
        {joints.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r="1.3" fill={C.card} stroke={color} strokeWidth="0.7" />)}
      </g>
    </svg>
  );
}

/* スイング映像（品のあるグラデ） */
function SwingStage({ frame, withSkeleton }) {
  const t = frame / 100;
  let a; if (t < 0.5) a = 90 - t * 2 * 220; else a = -130 + (t - 0.5) * 2 * 320;
  const cx = 100, cy = 130, len = 55;
  const hx = cx + Math.cos(a * Math.PI / 180) * len, hy = cy - Math.sin(a * Math.PI / 180) * len;
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <svg viewBox="0 0 200 220" style={{ width: "100%", height: "100%" }} preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="stage" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2C4A63" /><stop offset="55%" stopColor="#1F3A4F" /><stop offset="100%" stopColor="#162B3B" />
          </linearGradient>
          <radialGradient id="spot" cx="50%" cy="42%" r="55%">
            <stop offset="0%" stopColor="#3D6889" stopOpacity="0.6" /><stop offset="100%" stopColor="#162B3B" stopOpacity="0" />
          </radialGradient>
        </defs>
        <rect width="200" height="220" fill="url(#stage)" />
        <rect width="200" height="220" fill="url(#spot)" />
        <line x1="0" y1="200" x2="200" y2="200" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
        {/* 人物 */}
        <g opacity="0.9">
          <circle cx={cx} cy={75} r="13" fill="rgba(255,255,255,0.85)" />
          <line x1={cx} y1={88} x2={cx} y2={cy} stroke="rgba(255,255,255,0.85)" strokeWidth="9" strokeLinecap="round" />
          <line x1={cx} y1={cy} x2={cx - 14} y2={195} stroke="rgba(255,255,255,0.85)" strokeWidth="8" strokeLinecap="round" />
          <line x1={cx} y1={cy} x2={cx + 14} y2={195} stroke="rgba(255,255,255,0.85)" strokeWidth="8" strokeLinecap="round" />
          <line x1={cx} y1={105} x2={hx} y2={hy} stroke="rgba(255,255,255,0.92)" strokeWidth="5" strokeLinecap="round" />
          <circle cx={hx} cy={hy} r="4" fill={C.goal} />
        </g>
      </svg>
      {withSkeleton && <Skeleton frame={frame} />}
    </div>
  );
}

/* ====== レーダーチャート（細密） ====== */
function Radar({ you, target, pro, labels }) {
  const cx = 110, cy = 108, R = 76, N = labels.length;
  const pt = (vals) => vals.map((v, i) => { const ang = (Math.PI * 2 * i) / N - Math.PI / 2; const r = (v / 100) * R; return [cx + Math.cos(ang) * r, cy + Math.sin(ang) * r]; });
  const poly = (pts) => pts.map((p) => p.join(",")).join(" ");
  return (
    <svg viewBox="0 0 220 215" style={{ width: "100%" }}>
      {[0.25, 0.5, 0.75, 1].map((s, i) => (
        <polygon key={i} points={poly(Array.from({ length: N }, (_, j) => { const ang = (Math.PI * 2 * j) / N - Math.PI / 2; return [cx + Math.cos(ang) * R * s, cy + Math.sin(ang) * R * s]; }))} fill="none" stroke={C.line} strokeWidth="0.8" />
      ))}
      {labels.map((_, i) => { const ang = (Math.PI * 2 * i) / N - Math.PI / 2; return <line key={i} x1={cx} y1={cy} x2={cx + Math.cos(ang) * R} y2={cy + Math.sin(ang) * R} stroke={C.line} strokeWidth="0.8" />; })}
      <polygon points={poly(pt(pro))} fill="none" stroke={C.faint} strokeWidth="1.2" strokeDasharray="2,2" />
      <polygon points={poly(pt(target))} fill="none" stroke={C.gold} strokeWidth="1.3" strokeDasharray="4,2.5" />
      <polygon points={poly(pt(you))} fill={`${C.green}1F`} stroke={C.green} strokeWidth="2.2" />
      {pt(you).map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="2.6" fill={C.green} />)}
      {labels.map((l, i) => { const ang = (Math.PI * 2 * i) / N - Math.PI / 2; const lx = cx + Math.cos(ang) * (R + 16), ly = cy + Math.sin(ang) * (R + 16);
        return <g key={i}><text x={lx} y={ly - 2} textAnchor="middle" fontSize="9.5" fontWeight="600" fill={C.sub}>{l}</text>
          <text x={lx} y={ly + 9} textAnchor="middle" fontSize="11" fontWeight="700" fill={C.green} fontFamily={mono}>{you[i]}</text></g>; })}
    </svg>
  );
}

/* ====== 描画キャンバス ====== */
function DrawCanvas({ enabled, color, tool, clearSig }) {
  const ref = useRef(null), shapes = useRef([]), drawing = useRef(false), start = useRef(null), freePts = useRef([]);
  const redraw = useCallback(() => {
    const cv = ref.current; if (!cv) return; const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height);
    shapes.current.forEach((s) => {
      ctx.strokeStyle = s.color; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.lineJoin = "round";
      if (s.tool === "free") { ctx.beginPath(); s.pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke(); }
      else if (s.tool === "circle") { const r = Math.hypot(s.x2 - s.x1, s.y2 - s.y1); ctx.beginPath(); ctx.arc(s.x1, s.y1, r, 0, 6.3); ctx.stroke(); }
      else { ctx.beginPath(); ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2); ctx.stroke();
        if (s.tool === "angle") { const ang = Math.round(Math.atan2(-(s.y2 - s.y1), s.x2 - s.x1) * 180 / Math.PI); ctx.fillStyle = s.color; ctx.font = "bold 14px sans-serif"; ctx.fillText(`${ang}°`, s.x2 + 6, s.y2); } }
    });
  }, []);
  useEffect(() => { shapes.current = []; redraw(); }, [clearSig, redraw]);
  useEffect(() => { const cv = ref.current; if (!cv) return; const rz = () => { cv.width = cv.offsetWidth; cv.height = cv.offsetHeight; redraw(); }; rz(); window.addEventListener("resize", rz); return () => window.removeEventListener("resize", rz); }, [redraw]);
  const pos = (e) => { const r = ref.current.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e; return { x: t.clientX - r.left, y: t.clientY - r.top }; };
  const down = (e) => { if (!enabled) return; drawing.current = true; start.current = pos(e); freePts.current = [pos(e)]; };
  const move = (e) => { if (!enabled || !drawing.current) return; e.preventDefault(); const p = pos(e); redraw();
    const ctx = ref.current.getContext("2d"); ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.lineJoin = "round";
    if (tool === "free") { freePts.current.push(p); ctx.beginPath(); freePts.current.forEach((q, i) => i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)); ctx.stroke(); }
    else if (tool === "circle") { const r = Math.hypot(p.x - start.current.x, p.y - start.current.y); ctx.beginPath(); ctx.arc(start.current.x, start.current.y, r, 0, 6.3); ctx.stroke(); }
    else { ctx.beginPath(); ctx.moveTo(start.current.x, start.current.y); ctx.lineTo(p.x, p.y); ctx.stroke(); } };
  const up = (e) => { if (!enabled || !drawing.current) return; drawing.current = false;
    const p = pos(e.changedTouches ? { clientX: e.changedTouches[0].clientX, clientY: e.changedTouches[0].clientY } : e);
    if (tool === "free") shapes.current.push({ tool, color, pts: [...freePts.current] });
    else shapes.current.push({ tool, color, x1: start.current.x, y1: start.current.y, x2: p.x, y2: p.y }); redraw(); };
  return <canvas ref={ref} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: enabled ? "auto" : "none", touchAction: "none", cursor: "crosshair" }}
    onMouseDown={down} onMouseMove={move} onMouseUp={up} onTouchStart={down} onTouchMove={move} onTouchEnd={up} />;
}

function Toggle({ on, onClick }) {
  return <div onClick={onClick} style={{ width: 46, height: 27, borderRadius: 14, padding: 3, cursor: "pointer", background: on ? C.green : C.cardAlt, display: "flex", justifyContent: on ? "flex-end" : "flex-start", transition: "background .2s" }}>
    <div style={{ width: 21, height: 21, borderRadius: "50%", background: C.card, boxShadow: "0 1px 3px rgba(0,0,0,0.18)" }} /></div>;
}
function chip(on) {
  return { padding: "7px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600, border: `1px solid ${on ? C.green : C.line}`, background: on ? C.green : C.card, color: on ? C.card : C.sub, cursor: "pointer", display: "flex", alignItems: "center", gap: 5 };
}

/* ====== プレイヤー ====== */
function Player({ proMode, compact }) {
  const [playing, setPlaying] = useState(false), [frame, setFrame] = useState(0), [speed, setSpeed] = useState(0.5);
  const [skel, setSkel] = useState(true), [drawOn, setDrawOn] = useState(false);
  const [tool, setTool] = useState("free"), [dColor, setDColor] = useState(C.gold), [clearSig, setClearSig] = useState(0);
  const timer = useRef(null);
  const MARKS = [{ k: "A", f: 8 }, { k: "T", f: 50 }, { k: "I", f: 80 }, { k: "F", f: 96 }];
  useEffect(() => { if (playing) timer.current = setInterval(() => setFrame((f) => f >= 100 ? 0 : f + 1), 45 / speed); else clearInterval(timer.current); return () => clearInterval(timer.current); }, [playing, speed]);
  const phase = frame < 20 ? "Address" : frame < 45 ? "Backswing" : frame < 55 ? "Top" : frame < 75 ? "Downswing" : frame < 85 ? "Impact" : "Follow";
  return (
    <div>
      <div style={{ position: "relative", borderRadius: 16, overflow: "hidden", aspectRatio: compact ? "16/11" : "9/13", maxHeight: compact ? 230 : 420 }}>
        <SwingStage frame={frame} withSkeleton={skel} />
        <DrawCanvas enabled={drawOn} color={dColor} tool={tool} clearSig={clearSig} />
        <div style={{ position: "absolute", top: 12, left: 12, background: "rgba(22,43,59,0.5)", color: "#fff", fontSize: 11, fontWeight: 600, padding: "4px 11px", borderRadius: 14, backdropFilter: "blur(4px)", fontFamily: mono, letterSpacing: "0.5px" }}>{phase}</div>
        {proMode && <div style={{ position: "absolute", top: 12, right: 12, background: C.gold, color: C.ink, fontSize: 10, fontWeight: 700, padding: "4px 10px", borderRadius: 14, letterSpacing: "1px" }}>PRO</div>}
      </div>
      <div style={{ padding: "16px 4px 4px" }}>
        <input type="range" min={0} max={100} value={frame} onChange={(e) => { setFrame(+e.target.value); setPlaying(false); }} style={{ width: "100%", accentColor: C.green, height: 4 }} />
        <div style={{ position: "relative", height: 24, marginTop: 4 }}>
          {MARKS.map((m) => (
            <div key={m.k} onClick={() => { setFrame(m.f); setPlaying(false); }} style={{ position: "absolute", left: `${m.f}%`, transform: "translateX(-50%)", cursor: "pointer" }}>
              <div style={{ width: 20, height: 20, borderRadius: "50%", border: `1px solid ${Math.abs(frame - m.f) < 5 ? C.green : C.line}`, background: Math.abs(frame - m.f) < 5 ? C.green : C.card, color: Math.abs(frame - m.f) < 5 ? C.card : C.sub, fontSize: 10, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: mono }}>{m.k}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 18, padding: "8px 0 14px" }}>
        <div onClick={() => setFrame((f) => Math.max(0, f - 1))} style={ctrlBtn}><IconBack size={16} /></div>
        <div onClick={() => setPlaying((p) => !p)} style={{ width: 54, height: 54, borderRadius: "50%", background: C.green, color: C.card, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", boxShadow: `0 6px 18px ${C.green}44` }}>
          {playing ? <Icon size={20}><rect x="6" y="5" width="4" height="14" fill="currentColor" stroke="none" /><rect x="14" y="5" width="4" height="14" fill="currentColor" stroke="none" /></Icon> : <IconPlay size={20} />}
        </div>
        <div onClick={() => setFrame((f) => Math.min(100, f + 1))} style={ctrlBtn}><IconNext size={16} /></div>
      </div>
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 14, flexWrap: "wrap" }}>
        {[0.25, 0.5, 1].map((s) => <button key={s} onClick={() => setSpeed(s)} style={chip(speed === s)}>{s === 1 ? "等速" : `${s}x`}</button>)}
        <button onClick={() => setSkel((v) => !v)} style={chip(skel)}><IconSkeleton size={14} />骨格</button>
      </div>
      <div style={{ background: C.cardAlt, borderRadius: 14, padding: 12 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: drawOn ? 12 : 0 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: C.green, display: "flex", alignItems: "center", gap: 7 }}><IconPen size={16} stroke={C.green} />描画ツール</span>
          <Toggle on={drawOn} onClick={() => setDrawOn((d) => !d)} />
        </div>
        {drawOn && (
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", alignItems: "center" }}>
            {[["free", "手描き"], ["line", "直線"], ["angle", "角度"], ["circle", "円"]].map(([t, l]) => (
              <button key={t} onClick={() => setTool(t)} style={{ padding: "6px 11px", borderRadius: 9, fontSize: 11, fontWeight: 600, border: `1px solid ${tool === t ? C.green : C.line}`, background: tool === t ? C.card : "transparent", color: tool === t ? C.green : C.sub, cursor: "pointer" }}>{l}</button>
            ))}
            <div style={{ display: "flex", gap: 5, marginLeft: "auto" }}>
              {[C.red, C.gold, C.green, "#fff"].map((col) => <div key={col} onClick={() => setDColor(col)} style={{ width: 22, height: 22, borderRadius: "50%", background: col, border: dColor === col ? `2px solid ${C.ink}` : `1px solid ${C.line}`, cursor: "pointer" }} />)}
            </div>
            <button onClick={() => setClearSig((s) => s + 1)} style={{ padding: "6px 11px", borderRadius: 9, fontSize: 11, fontWeight: 600, border: `1px solid ${C.line}`, background: C.card, color: C.red, cursor: "pointer" }}>消去</button>
          </div>
        )}
      </div>
    </div>
  );
}
const ctrlBtn = { width: 42, height: 42, borderRadius: "50%", border: `1px solid ${C.line}`, background: C.card, display: "flex", alignItems: "center", justifyContent: "center", color: C.sub, cursor: "pointer" };

/* ====== 実写動画 + MediaPipe骨格 の再生プレイヤー ====== */
function PoseReviewPlayer({ videoUrl, poseFrames }) {
  const videoRef = useRef(null), canvasRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [skel, setSkel] = useState(true);
  const [dur, setDur] = useState(0);
  const [cur, setCur] = useState(0);
  const rafRef = useRef(null);
  const MARKS = [{ k: "A", t: 0.08 }, { k: "T", t: 0.5 }, { k: "I", t: 0.8 }, { k: "F", t: 0.96 }];

  // 現在時刻に対応する骨格を描画
  const draw = useCallback(() => {
    const v = videoRef.current, cv = canvasRef.current;
    if (!v || !cv || !poseFrames?.length) return;
    cv.width = v.videoWidth || cv.offsetWidth;
    cv.height = v.videoHeight || cv.offsetHeight;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!skel) return;
    const idx = Math.min(poseFrames.length - 1, Math.floor((v.currentTime / (v.duration || 1)) * poseFrames.length));
    const lm = poseFrames[idx];
    if (!lm) return;
    ctx.strokeStyle = "#7FD0F5"; ctx.lineWidth = Math.max(2, cv.width / 180);
    POSE_CONNECTIONS.forEach(([a, b]) => {
      if (!lm[a] || !lm[b]) return;
      ctx.beginPath();
      ctx.moveTo(lm[a].x * cv.width, lm[a].y * cv.height);
      ctx.lineTo(lm[b].x * cv.width, lm[b].y * cv.height);
      ctx.stroke();
    });
    ctx.fillStyle = "#fff";
    lm.forEach((p) => { ctx.beginPath(); ctx.arc(p.x * cv.width, p.y * cv.height, Math.max(3, cv.width / 240), 0, 6.3); ctx.fill(); });
  }, [poseFrames, skel]);

  useEffect(() => {
    const loop = () => { draw(); if (videoRef.current) setCur(videoRef.current.currentTime); rafRef.current = requestAnimationFrame(loop); };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = speed; }, [speed]);

  const toggle = () => { const v = videoRef.current; if (!v) return; if (playing) v.pause(); else v.play(); setPlaying(!playing); };
  const seek = (t) => { if (videoRef.current) { videoRef.current.currentTime = t; videoRef.current.pause(); setPlaying(false); } };
  const stepFrame = (dir) => { if (videoRef.current) { videoRef.current.currentTime = Math.max(0, Math.min(dur, videoRef.current.currentTime + dir / 30)); videoRef.current.pause(); setPlaying(false); } };
  const phase = (() => { const r = dur ? cur / dur : 0; return r < 0.2 ? "Address" : r < 0.45 ? "Backswing" : r < 0.55 ? "Top" : r < 0.75 ? "Downswing" : r < 0.85 ? "Impact" : "Follow"; })();

  return (
    <div>
      <div style={{ position: "relative", borderRadius: 16, overflow: "hidden", background: "#162B3B", aspectRatio: "9/13", maxHeight: 420 }}>
        <video ref={videoRef} src={videoUrl} muted playsInline onLoadedMetadata={(e) => setDur(e.target.duration)} onEnded={() => setPlaying(false)} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
        <div style={{ position: "absolute", top: 12, left: 12, background: "rgba(22,43,59,0.55)", color: "#fff", fontSize: 11, fontWeight: 600, padding: "4px 11px", borderRadius: 14, fontFamily: mono }}>{phase}</div>
      </div>
      <div style={{ padding: "16px 4px 4px" }}>
        <input type="range" min={0} max={dur || 1} step="0.01" value={cur} onChange={(e) => seek(+e.target.value)} style={{ width: "100%", accentColor: C.green, height: 4 }} />
        <div style={{ position: "relative", height: 24, marginTop: 4 }}>
          {MARKS.map((m) => (
            <div key={m.k} onClick={() => seek(m.t * dur)} style={{ position: "absolute", left: `${m.t * 100}%`, transform: "translateX(-50%)", cursor: "pointer" }}>
              <div style={{ width: 20, height: 20, borderRadius: "50%", border: `1px solid ${C.line}`, background: C.card, color: C.sub, fontSize: 10, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: mono }}>{m.k}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 18, padding: "8px 0 14px" }}>
        <div onClick={() => stepFrame(-1)} style={ctrlBtn}><IconBack size={16} /></div>
        <div onClick={toggle} style={{ width: 54, height: 54, borderRadius: "50%", background: C.green, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", boxShadow: `0 6px 18px ${C.green}44` }}>
          {playing ? <Icon size={20}><rect x="6" y="5" width="4" height="14" fill="currentColor" stroke="none" /><rect x="14" y="5" width="4" height="14" fill="currentColor" stroke="none" /></Icon> : <IconPlay size={20} />}
        </div>
        <div onClick={() => stepFrame(1)} style={ctrlBtn}><IconNext size={16} /></div>
      </div>
      <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
        {[0.25, 0.5, 1].map((s) => <button key={s} onClick={() => setSpeed(s)} style={chip(speed === s)}>{s === 1 ? "等速" : `${s}x`}</button>)}
        <button onClick={() => setSkel((v) => !v)} style={chip(skel)}><IconSkeleton size={14} />骨格</button>
      </div>
    </div>
  );
}

/* ============================================================ MAIN */
function UserApp({ onLogout }) {
  const [tab, setTab] = useState("round");
  const [club, setClub] = useState("dr");
  const [autoRec, setAutoRec] = useState(true);
  const [voice, setVoice] = useState(true);
  const [cameraOn, setCameraOn] = useState(false);
  const [recState, setRecState] = useState("idle");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const videoRef = useRef(null), streamRef = useRef(null), recTimer = useRef(null), detectTimer = useRef(null);
  const [recTime, setRecTime] = useState(0);
  // MediaPipe関連
  const recorderRef = useRef(null), chunksRef = useRef([]);
  const [videoUrl, setVideoUrl] = useState(null);
  const [poseFrames, setPoseFrames] = useState(null);
  const [poseProgress, setPoseProgress] = useState(0);
  const hiddenVideoRef = useRef(null);
  const clubObj = CLUBS.find((c) => c.id === club);

  // ラウンド用
  const [roundActive, setRoundActive] = useState(false); // ラウンド記録中か
  const [hole, setHole] = useState(0); // index 0-17
  const [gps, setGps] = useState(null); // {lat,lng}
  const [gpsState, setGpsState] = useState("idle"); // idle/locating/ok/error
  const [scores, setScores] = useState(Array(18).fill(null));
  const [putts, setPutts] = useState(Array(18).fill(null));
  const [practiceView, setPracticeView] = useState("record"); // record / review
  const [coachView, setCoachView] = useState("chat"); // chat / pro
  const watchId = useRef(null);
  const course = SAMPLE_COURSE;
  const holeObj = course.holes[hole];

  const startGPS = () => {
    if (!navigator.geolocation) { setGpsState("error"); return; }
    setGpsState("locating");
    watchId.current = navigator.geolocation.watchPosition(
      (pos) => { setGps({ lat: pos.coords.latitude, lng: pos.coords.longitude }); setGpsState("ok"); },
      () => setGpsState("error"),
      { enableHighAccuracy: true, maximumAge: 1000, timeout: 10000 }
    );
  };
  const stopGPS = () => { if (watchId.current != null) { navigator.geolocation.clearWatch(watchId.current); watchId.current = null; } setGpsState("idle"); setGps(null); };
  useEffect(() => () => { if (watchId.current != null) navigator.geolocation.clearWatch(watchId.current); }, []);

  // ラウンド開始：設定確定 → 記録スタート + GPS自動ON
  const beginRound = () => {
    setRoundActive(true);
    setHole(0);
    setShowSetup(false);
    startGPS(); // 記録中はGPS常時ON
  };
  const endRound = () => {
    setRoundActive(false);
    stopGPS();
  };

  // 残り距離（GPSがあれば実測、なければデモ値）
  const distTo = (pt) => {
    if (gps) return distanceYards(gps.lat, gps.lng, pt.lat, pt.lng);
    return null;
  };
  const demoDist = { front: holeObj.yard - 8, center: holeObj.yard, back: holeObj.yard + 9 };
  const dFront = distTo(holeObj.green.front) ?? demoDist.front;
  const dCenter = distTo(holeObj.green.center) ?? demoDist.center;
  const dBack = distTo(holeObj.green.back) ?? demoDist.back;

  const setScore = (delta) => setScores((s) => { const n = [...s]; n[hole] = Math.max(1, (n[hole] || holeObj.par) + delta); return n; });
  const setPutt = (delta) => setPutts((p) => { const n = [...p]; n[hole] = Math.max(0, (n[hole] || 2) + delta); return n; });
  const totalScore = scores.reduce((a, b) => a + (b || 0), 0);
  const playedPar = course.holes.reduce((a, h, i) => a + (scores[i] != null ? h.par : 0), 0);
  const scoreVsPar = totalScore - playedPar;

  // AIチャット
  const [chatMsgs, setChatMsgs] = useState([{ role: "ai", text: "こんにちは！専属AIコーチです。スイング解析やスコアについて、何でも聞いてください。" }]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading) return;
    const next = [...chatMsgs, { role: "user", text: chatInput }];
    setChatMsgs(next); setChatInput(""); setChatLoading(true);
    try {
      const ctx = result ? `直近の解析: スコア${result.score}, ${clubObj.n}, 軌道${result.pathType}, 課題=${result.issues?.map((i) => i.title).join("、")}` : "解析データなし";
      const r = await fetch("https://api.anthropic.com/v1/messages", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "claude-sonnet-4-6", max_tokens: 1000,
          system: `あなたはゴルフ専属AIコーチ。親身で具体的。${ctx}。2-4文でアドバイス。`,
          messages: next.map((m) => ({ role: m.role === "ai" ? "assistant" : "user", content: m.text })) }) });
      const d = await r.json();
      setChatMsgs((p) => [...p, { role: "ai", text: d.content?.[0]?.text || "もう一度お願いします。" }]);
    } catch {
      setChatMsgs((p) => [...p, { role: "ai", text: "通信エラーが起きました。少し待ってから試してください。" }]);
    } finally { setChatLoading(false); }
  };

  // ラウンド設定
  const [showSetup, setShowSetup] = useState(false);
  const [weather, setWeather] = useState("sunny");
  const [members, setMembers] = useState(["小西 公幸"]);
  const [memo, setMemo] = useState("");
  const [puttInput, setPuttInput] = useState(true);

  // コース登録（ピン留め）
  const [showCourseReg, setShowCourseReg] = useState(false);
  const [pins, setPins] = useState({ front: null, center: null, back: null });
  const [pinTarget, setPinTarget] = useState("center");

  const startCamera = async () => {
    try { const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = stream; if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play(); }
      setCameraOn(true); if (autoRec) beginDetect();
    } catch { alert("カメラを起動できませんでした。アクセスを許可してください。"); }
  };
  const beginDetect = () => { setRecState("detecting"); detectTimer.current = setTimeout(() => startRec(), 2500); };
  const startRec = () => {
    setRecState("recording"); setRecTime(0); setVideoUrl(null); setPoseFrames(null);
    // 実録画
    try {
      chunksRef.current = [];
      const rec = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });
      rec.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoUrl(URL.createObjectURL(blob));
      };
      rec.start();
      recorderRef.current = rec;
    } catch (e) { /* 非対応端末はデモ動作 */ }
    recTimer.current = setInterval(() => setRecTime((t) => { if (t >= 3) { stopRec(); return t; } return t + 1; }), 700);
  };
  const stopRec = () => {
    clearInterval(recTimer.current);
    if (recorderRef.current && recorderRef.current.state !== "inactive") recorderRef.current.stop();
    setRecState("done");
  };
  useEffect(() => () => { clearInterval(recTimer.current); clearTimeout(detectTimer.current); if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop()); }, []);

  const analyze = async () => {
    setAnalyzing(true); setResult(null); setPoseProgress(0);
    // 1) MediaPipeで全フレームの骨格を抽出
    let poseStats = null;
    if (videoUrl && hiddenVideoRef.current) {
      try {
        const v = hiddenVideoRef.current;
        v.src = videoUrl;
        await new Promise((res) => { v.onloadedmetadata = res; });
        const frames = await extractPoseFrames(v, (p) => setPoseProgress(p));
        setPoseFrames(frames);
        poseStats = analyzePoseFrames(frames);
      } catch (e) { /* 抽出失敗時はAI解析のみ */ }
    }
    // 2) 骨格データ（あれば）をAIに渡して講評
    try {
      const ctx = poseStats ? `骨格解析結果: 推定ヘッドスピード${poseStats.headSpeed}m/s, 推定飛距離${poseStats.distance}y, 背骨前傾角${poseStats.spineAngle}度。` : "骨格データなし。";
      const r = await fetch("https://api.anthropic.com/v1/messages", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "claude-sonnet-4-6", max_tokens: 1000,
          system: `ゴルフスイング解析AI。${clubObj.n}のスイング。${ctx}これを踏まえJSONのみ返す:
{"speed":数値,"distance":数値,"pathAngle":数値-8〜8,"pathType":"インサイドアウト"|"アウトサイドイン"|"スクエア","score":数値60-95,"radar":{"tech":1-100,"stability":1-100,"power":1-100,"rhythm":1-100,"linkage":1-100},"rating":"S"|"A"|"B+"|"B"|"C","issues":[{"phase":"アドレス"|"トップ"|"ダウン"|"インパクト","title":"課題(15字)","desc":"説明(30字)"}],"advice":["30字以内","30字以内","30字以内"]}
speed/distanceは骨格解析結果があればそれを優先。issuesは2-3個。余計な文字不要。`,
          messages: [{ role: "user", content: `${clubObj.n}でのスイング解析をお願いします。` }] }) });
      const d = await r.json(); const txt = (d.content?.[0]?.text || "").replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(txt);
      if (poseStats) { parsed.speed = poseStats.headSpeed; parsed.distance = poseStats.distance; }
      setResult(parsed);
    } catch {
      setResult({ speed: poseStats?.headSpeed || 43.2, distance: poseStats?.distance || 205, pathAngle: -2.4, pathType: "インサイドアウト", score: 84,
        radar: { tech: 82, stability: 76, power: 88, rhythm: 79, linkage: 85 }, rating: "A",
        issues: [{ phase: "トップ", title: "オーバースイング", desc: "トップでクラブが寝すぎています" }, { phase: "ダウン", title: "右肩の突っ込み", desc: "切り返しで右肩が前に出ています" }],
        advice: ["トップでの間を作ると軌道が安定します", "下半身リードの切り返しを意識しましょう", "フォローを高く取ると飛距離が伸びます"] });
    } finally { setAnalyzing(false); }
  };
  const ratingColor = (r) => r === "S" ? C.gold : r === "A" ? C.green : r?.startsWith("B") ? C.greenLite : C.faint;

  return (
    <div style={S.app}>
      <style>{`
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
        @keyframes fade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        @keyframes scan{0%{top:0}100%{top:100%}}
        @keyframes slideUp{from{transform:translateY(100%)}to{transform:translateY(0)}}
        *{box-sizing:border-box;margin:0;padding:0}
        body{background:${C.bg}}
        input[type=range]{-webkit-appearance:none;background:${C.line};border-radius:4px}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:${C.green};cursor:pointer}
        ::-webkit-scrollbar{display:none}
      `}</style>

      <div style={{ ...S.topbar, background: `linear-gradient(135deg,${C.green},${C.greenDark})`, borderBottom: "none" }}>
        <div>
          <div style={{ fontSize: 21, fontWeight: 800, letterSpacing: "0.5px", lineHeight: 1, color: "#fff" }}>Golog</div>
          <div style={{ fontSize: 9, letterSpacing: "3px", color: "rgba(255,255,255,0.7)", marginTop: 3, textTransform: "uppercase" }}>Golf Tracker</div>
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 7, padding: "7px 14px", borderRadius: 16, background: "rgba(255,255,255,0.2)" }}>
          <span style={{ fontSize: 13, fontWeight: 800, color: "#fff", fontFamily: mono }}>{clubObj.s}</span>
          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.8)", fontFamily: mono }}>{clubObj.dist}y</span>
        </div>
      </div>

      <div style={S.content}>
        {/* ===== ラウンド ===== */}
        {tab === "round" && !roundActive && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ marginBottom: 16 }}><div style={S.eyebrow}>Score</div><div style={S.heading}>ラウンド記録</div></div>

            {/* ラウンド開始ボタン */}
            <button onClick={() => setShowSetup(true)} style={{ width: "100%", padding: "18px", borderRadius: 16, background: `linear-gradient(135deg,${C.green},${C.greenDark})`, color: "#fff", border: "none", fontSize: 16, fontWeight: 800, cursor: "pointer", marginBottom: 18, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, boxShadow: `0 8px 24px ${C.green}44` }}>
              <IconFlag size={20} stroke="#fff" />ラウンド開始
            </button>

            {/* GDO風スコア統計ダッシュボード */}
            <div style={{ background: `linear-gradient(180deg,${C.green},${C.greenDark})`, borderRadius: 18, padding: 18, marginBottom: 16, color: "#fff" }}>
              <div style={{ display: "flex", justifyContent: "space-around", marginBottom: 16 }}>
                {[["ベストスコア", "82"], ["平均スコア", "91.9"], ["平均パット数", "33.1"]].map(([l, v], i) => (
                  <div key={i} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 28, fontWeight: 800, lineHeight: 1 }}>{v}</div>
                    <div style={{ fontSize: 10, color: "rgba(255,255,255,0.85)", marginTop: 4 }}>{l}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "center", gap: 14, fontSize: 10, marginBottom: 8 }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 14, borderTop: `2px solid ${C.goal}` }} />目標</span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 9, height: 9, background: C.greenLite, borderRadius: 2 }} />スコア</span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}><span style={{ width: 9, height: 9, background: C.gold, borderRadius: 2 }} />パット</span>
              </div>
              <StackedBars />
            </div>

            {/* ドーナツ分析 */}
            <div style={S.card}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>スコア分析<span style={{ fontSize: 11, color: C.faint, fontWeight: 400, marginLeft: 6 }}>過去10回</span></div>
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 14 }}><Donut /></div>
              <div style={{ display: "flex", justifyContent: "space-between", borderTop: `1px solid ${C.line}`, paddingTop: 12 }}>
                {[["イーグル以上", "0.0", "#E0A82E"], ["バーディ", "3.7", C.red], ["パー", "26.5", C.green], ["ボギー", "35.8", C.gold], ["+2以上", "34.0", C.faint]].map(([k, v, c], i) => (
                  <div key={i} style={{ textAlign: "center", flex: 1 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: c, margin: "0 auto 5px" }} />
                    <div style={{ fontSize: 9, color: C.sub }}>{k}</div>
                    <div style={{ fontSize: 12, fontWeight: 700, fontFamily: mono }}>{v}%</div>
                  </div>
                ))}
              </div>
            </div>

            {/* 履歴 */}
            <div style={{ ...S.card, background: C.greenSoft, padding: "11px 16px", display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: C.sub }}>期間中ラウンド数</span><b style={{ color: C.green }}>10</b>
            </div>
            {[["2026年6月13日(土)", "リバーサイドパーク長門石ゴルフ場", 47, 21], ["2026年5月25日(月)", "リバーサイドパーク長門石ゴルフ場", 82, 34], ["2026年5月9日(土)", "リバーサイドパーク長門石ゴルフ場", 87, 33], ["2026年5月8日(金)", "リバーサイドパーク長門石ゴルフ場", 84, 35]].map(([date, course2, sc, pt], i) => (
              <div key={i} style={{ ...S.card, padding: 14, marginBottom: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: C.faint, marginBottom: 3 }}>{date}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 5 }}>{course2}</div>
                  <div style={{ fontSize: 12, color: C.sub }}>スコア：{sc}　パット：{pt}</div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
                  <span style={{ fontSize: 10, color: C.sub, background: C.cardAlt, padding: "3px 8px", borderRadius: 6 }}>✓ 同期済</span>
                  <IconNext size={15} stroke={C.green} />
                </div>
              </div>
            ))}

            {/* 管理者専用は別画面なのでコース登録導線はここから除外 */}
            {/* ラウンド設定モーダル（開始フロー） */}
            {showSetup && (
              <Modal title="ラウンド設定" onClose={() => setShowSetup(false)}>
                <Row label="ラウンド日"><span style={{ fontSize: 13 }}>2026年6月13日(土)</span></Row>
                <Row label="天気">
                  <div style={{ display: "flex", gap: 6 }}>
                    {[["sunny", "☀"], ["cloudy", "☁"], ["rainy", "☂"], ["snow", "⛄"]].map(([k, ic]) => (
                      <div key={k} onClick={() => setWeather(k)} style={{ width: 38, height: 34, borderRadius: 9, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, cursor: "pointer", background: weather === k ? C.green : C.cardAlt, border: `1px solid ${weather === k ? C.green : C.line}` }}>{ic}</div>
                    ))}
                  </div>
                </Row>
                <Row label="コース"><span style={{ fontSize: 13 }}>{course.name}</span></Row>
                <Row label="ティー"><span style={{ fontSize: 13, fontFamily: mono }}>OUT REG</span></Row>
                <Row label="パット入力"><Toggle on={puttInput} onClick={() => setPuttInput((v) => !v)} /></Row>
                <div style={{ fontSize: 12, fontWeight: 700, margin: "14px 0 8px" }}>同伴者</div>
                {members.map((m, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <input value={m} onChange={(e) => setMembers((p) => p.map((x, j) => j === i ? e.target.value : x))} style={{ flex: 1, padding: "10px 12px", borderRadius: 10, border: `1px solid ${C.line}`, fontSize: 13, outline: "none" }} />
                    {i > 0 && <div onClick={() => setMembers((p) => p.filter((_, j) => j !== i))} style={{ color: C.red, cursor: "pointer", fontSize: 18 }}>×</div>}
                  </div>
                ))}
                {members.length < 4 && <button onClick={() => setMembers((p) => [...p, ""])} style={{ width: "100%", padding: 10, borderRadius: 10, background: C.greenSoft, color: C.green, border: "none", fontSize: 12, fontWeight: 700, cursor: "pointer", marginBottom: 12 }}>+ 同伴者を追加</button>}
                <div style={{ fontSize: 12, fontWeight: 700, margin: "8px 0" }}>ラウンドメモ</div>
                <textarea value={memo} onChange={(e) => setMemo(e.target.value)} placeholder="メモを入力..." rows={2} style={{ width: "100%", padding: 12, borderRadius: 10, border: `1px solid ${C.line}`, fontSize: 13, outline: "none", resize: "none", fontFamily: "inherit" }} />
                <button onClick={beginRound} style={{ ...pBtn(false), marginTop: 14 }}>この内容でラウンド開始</button>
              </Modal>
            )}
          </div>
        )}

        {/* ===== ラウンド記録中（GPS距離 + スコア） ===== */}
        {tab === "round" && roundActive && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.red, animation: "pulse 1s infinite" }} />
                <div style={{ fontSize: 16, fontWeight: 800 }}>ラウンド記録中</div>
              </div>
              <button onClick={endRound} style={{ padding: "8px 14px", borderRadius: 11, background: C.card, border: `1px solid ${C.red}`, color: C.red, fontSize: 12, fontWeight: 700, cursor: "pointer" }}>終了</button>
            </div>

            {/* ホールセレクタ */}
            <div style={{ ...S.card, padding: "14px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div onClick={() => setHole((h) => Math.max(0, h - 1))} style={ctrlBtn}><IconBack size={16} /></div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 10, letterSpacing: "2px", color: C.gold }}>HOLE</div>
                <div style={{ fontFamily: serif, fontSize: 32, fontWeight: 600, lineHeight: 1 }}>{holeObj.no}</div>
                <div style={{ fontSize: 12, color: C.sub, fontFamily: mono }}>PAR {holeObj.par} · {holeObj.yard}y</div>
              </div>
              <div onClick={() => setHole((h) => Math.min(17, h + 1))} style={ctrlBtn}><IconNext size={16} /></div>
            </div>

            {/* GPS距離（前/中/奥） */}
            <div style={{ ...S.card, background: C.green, color: "#fff", position: "relative", overflow: "hidden" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <div style={{ fontSize: 11, letterSpacing: "2px", color: C.gold, textTransform: "uppercase" }}>グリーンまで</div>
                <div style={{ fontSize: 10, display: "flex", alignItems: "center", gap: 5, color: "rgba(255,255,255,0.8)" }}>
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: gpsState === "ok" ? "#7BD4A0" : gpsState === "locating" ? C.gold : "rgba(255,255,255,0.4)" }} />
                  {gpsState === "ok" ? "GPS測位中" : gpsState === "locating" ? "測位中..." : gpsState === "error" ? "GPS不可" : "GPS未開始"}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-around" }}>

                <DistCol label="前" value={dFront} />
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.7)", marginBottom: 4 }}>センター</div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 3, justifyContent: "center" }}>
                    <span style={{ fontSize: 56, fontWeight: 700, fontFamily: serif, lineHeight: 0.9, color: "#fff" }}>{dCenter}</span>
                  </div>
                  <div style={{ fontSize: 11, color: C.gold, fontFamily: mono }}>yard</div>
                </div>
                <DistCol label="奥" value={dBack} />
              </div>
              {!gps && <div style={{ fontSize: 10, color: "rgba(255,255,255,0.6)", textAlign: "center", marginTop: 14 }}>※ プレビューではGPSが取得できないためデモ値。実機では自動で実測されます</div>}
            </div>

            {/* スコア入力 */}
            <div style={S.card}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 16 }}>スコア入力</div>
              <div style={{ display: "flex", gap: 12 }}>
                <Stepper label="ストローク" value={scores[hole] ?? holeObj.par} onMinus={() => setScore(-1)} onPlus={() => setScore(1)} highlight />
                <Stepper label="パット" value={putts[hole] ?? 2} onMinus={() => setPutt(-1)} onPlus={() => setPutt(1)} />
              </div>
              {scores[hole] != null && (
                <div style={{ textAlign: "center", marginTop: 14, fontSize: 13, fontWeight: 700, color: scores[hole] - holeObj.par <= 0 ? C.green : C.red }}>
                  {(() => { const v = scores[hole] - holeObj.par;
                    return v === 0 ? "パー" : v === -1 ? "バーディー 🐦" : v <= -2 ? "イーグル 🦅" : v === 1 ? "ボギー" : `+${v}`; })()}
                </div>
              )}
            </div>

            {/* スコアカード集計 */}
            <div style={S.card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                <span style={{ fontSize: 13, fontWeight: 700 }}>スコアカード</span>
                <span style={{ fontSize: 13, fontWeight: 700, fontFamily: mono, color: scoreVsPar <= 0 ? C.green : C.red }}>
                  {totalScore > 0 ? `${totalScore} (${scoreVsPar >= 0 ? "+" : ""}${scoreVsPar})` : "—"}
                </span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(9,1fr)", gap: 3 }}>
                {course.holes.slice(0, 9).map((h, i) => <ScoreCell key={i} no={h.no} par={h.par} score={scores[i]} active={i === hole} onClick={() => setHole(i)} />)}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(9,1fr)", gap: 3, marginTop: 3 }}>
                {course.holes.slice(9, 18).map((h, i) => <ScoreCell key={i} no={h.no} par={h.par} score={scores[i + 9]} active={i + 9 === hole} onClick={() => setHole(i + 9)} />)}
              </div>
            </div>
          </div>
        )}

        {/* ===== 練習（撮影 / 再生） ===== */}
        {tab === "practice" && (
          <div style={{ marginBottom: 18 }}>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => setPracticeView("record")} style={chip(practiceView === "record")}>撮影・解析</button>
              <button onClick={() => setPracticeView("review")} style={chip(practiceView === "review")}>スイング再生</button>
            </div>
          </div>
        )}

        {/* ===== 撮影 ===== */}
        {tab === "practice" && practiceView === "record" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ marginBottom: 18 }}>
              <div style={S.eyebrow}>Session</div>
              <div style={S.heading}>スイングを記録する</div>
            </div>

            {/* クラブグリッド */}
            <div style={S.card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: C.ink }}>クラブ選択</span>
                <span style={{ fontSize: 11, color: C.faint }}>{clubObj.n}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
                {CLUBS.map((c) => {
                  const on = club === c.id;
                  return <div key={c.id} onClick={() => setClub(c.id)} style={{ padding: "13px 4px", borderRadius: 13, border: `1px solid ${on ? C.green : C.line}`, background: on ? C.green : C.card, cursor: "pointer", textAlign: "center", transition: "all .12s" }}>
                    <div style={{ fontSize: 15, fontWeight: 700, color: on ? C.card : C.ink, fontFamily: mono }}>{c.s}</div>
                    <div style={{ fontSize: 9, color: on ? "rgba(255,255,255,0.75)" : C.faint, marginTop: 3, fontFamily: mono }}>{c.dist}y</div>
                  </div>;
                })}
              </div>
            </div>

            {/* カメラ */}
            <div style={{ position: "relative", borderRadius: 20, overflow: "hidden", aspectRatio: "9/13", maxHeight: 440, boxShadow: "0 8px 30px rgba(42,38,32,0.12)" }}>
              {cameraOn ? <video ref={videoRef} muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} /> :
                <div style={{ position: "absolute", inset: 0 }}><SwingStage frame={8} withSkeleton={false} />
                  <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, color: "#fff" }}>
                    <IconCamera size={40} stroke="#fff" sw={1.4} /><div style={{ fontSize: 13, opacity: 0.85 }}>タップしてカメラを起動</div>
                  </div></div>}
              {cameraOn && <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} viewBox="0 0 90 130" preserveAspectRatio="none">
                <rect x="30" y="18" width="30" height="98" rx="3" stroke={recState === "recording" ? C.red : "rgba(255,255,255,0.6)"} strokeWidth="0.5" strokeDasharray="3,2" fill="none" />
                <line x1="45" y1="18" x2="45" y2="116" stroke="rgba(255,255,255,0.22)" strokeWidth="0.4" /></svg>}
              {recState === "detecting" && <div style={badgeMid}><span style={dot(C.gold)} />構えを検出中</div>}
              {recState === "recording" && <div style={{ position: "absolute", top: 14, left: 14, background: C.red, color: C.card, fontSize: 12, fontWeight: 700, padding: "6px 13px", borderRadius: 16, display: "flex", alignItems: "center", gap: 7, fontFamily: mono }}><span style={dot(C.card)} />REC {recTime}.0s</div>}
              {recState === "done" && <div style={{ position: "absolute", top: 14, left: 14, background: C.green, color: C.card, fontSize: 12, fontWeight: 700, padding: "6px 13px", borderRadius: 16 }}>✓ 撮影完了</div>}
              {analyzing && <div style={{ position: "absolute", inset: 0, background: "rgba(22,43,59,0.84)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
                <div style={{ width: 44, height: 44, border: "2.5px solid rgba(255,255,255,0.2)", borderTop: `2.5px solid ${C.gold}`, borderRadius: "50%", animation: "spin .8s linear infinite" }} />
                <div style={{ color: "#fff", fontSize: 12, letterSpacing: "2px", fontFamily: mono }}>{poseProgress > 0 && poseProgress < 1 ? `骨格抽出 ${Math.round(poseProgress * 100)}%` : "ANALYZING"}</div>
                <div style={{ position: "absolute", left: 0, right: 0, height: 1.5, background: `linear-gradient(90deg,transparent,${C.gold},transparent)`, animation: "scan 1.5s linear infinite" }} /></div>}
            </div>
            {/* 骨格抽出用の隠し動画 */}
            <video ref={hiddenVideoRef} muted playsInline style={{ display: "none" }} />

            {/* 三脚ガイド */}
            <div style={{ ...S.card, marginTop: 16, background: C.cardAlt }}>
              <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                <div style={{ color: C.green, flexShrink: 0 }}><IconTripod size={36} stroke={C.green} sw={1.3} /></div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>三脚セットガイド</div>
                  <div style={{ fontSize: 12, color: C.sub, lineHeight: 1.6 }}>ボールの2〜3m後方・腰の高さ。ターゲットラインに垂直に設置し、スイング全体を収めます。</div>
                </div>
              </div>
            </div>

            {/* 設定 */}
            <div style={{ ...S.card, padding: "6px 20px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "13px 0", borderBottom: `1px solid ${C.line}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: 11 }}><IconBolt size={18} stroke={C.green} /><div><div style={{ fontSize: 14, fontWeight: 600 }}>自動録画</div><div style={{ fontSize: 11, color: C.faint }}>構えを検出して自動開始</div></div></div>
                <Toggle on={autoRec} onClick={() => setAutoRec((a) => !a)} />
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "13px 0" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 11 }}><IconSound size={18} stroke={C.green} /><div><div style={{ fontSize: 14, fontWeight: 600 }}>音声フィードバック</div><div style={{ fontSize: 11, color: C.faint }}>結果を読み上げ</div></div></div>
                <Toggle on={voice} onClick={() => setVoice((v) => !v)} />
              </div>
            </div>

            {!cameraOn ? <button onClick={startCamera} style={pBtn(false)}>カメラを起動</button> :
              recState === "done" ? <div style={{ display: "flex", gap: 10 }}>
                <button onClick={() => { setRecState("idle"); setResult(null); if (autoRec) beginDetect(); }} style={{ ...pBtn(false), flex: 1, background: C.card, color: C.green, border: `1px solid ${C.green}` }}>撮り直す</button>
                <button onClick={analyze} disabled={analyzing} style={{ ...pBtn(analyzing), flex: 2 }}>{analyzing ? "解析中..." : "スイングを解析"}</button>
              </div> : recState === "recording" ? <button onClick={stopRec} style={{ ...pBtn(false), background: C.red }}>停止</button> :
                !autoRec ? <button onClick={startRec} style={pBtn(false)}>録画開始</button> : <button disabled style={pBtn(true)}>構えてください...</button>}

            {result && (
              <div style={{ animation: "fade .4s ease", marginTop: 22 }}>
                <div style={{ marginBottom: 16 }}><div style={S.eyebrow}>Report</div><div style={S.heading}>解析レポート</div></div>

                {/* スコア大表示（スポーツの大胆さ） */}
                <div style={{ ...S.card, background: C.green, color: "#fff", position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "relative", zIndex: 1, display: "flex", alignItems: "flex-end", justifyContent: "space-between" }}>
                    <div>
                      <div style={{ fontSize: 11, letterSpacing: "2px", color: C.gold, textTransform: "uppercase", marginBottom: 6 }}>Swing Score</div>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                        <span style={{ fontSize: 64, fontWeight: 700, fontFamily: serif, lineHeight: 0.9 }}>{result.score}</span>
                        <span style={{ fontSize: 18, color: "rgba(255,255,255,0.6)" }}>/100</span>
                      </div>
                      <div style={{ fontSize: 13, marginTop: 8, color: "rgba(255,255,255,0.85)" }}>{result.pathType} · {clubObj.n}</div>
                    </div>
                    <div style={{ width: 56, height: 56, borderRadius: "50%", border: `2px solid ${C.gold}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, fontWeight: 700, fontFamily: serif, color: C.gold }}>{result.rating}</div>
                  </div>
                  <div style={{ position: "absolute", right: -20, bottom: -20, opacity: 0.08 }}><IconAward size={140} stroke="#fff" sw={1} /></div>
                </div>

                {/* 計測値（モノスペース大） */}
                <div style={{ display: "flex", gap: 14, marginBottom: 16 }}>
                  <BigMetric label="HEAD SPEED" value={result.speed} unit="m/s" />
                  <BigMetric label="DISTANCE" value={result.distance} unit="yd" />
                </div>

                {/* レーダー */}
                <div style={S.card}>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>スイング能力</div>
                  <Radar you={[result.radar.tech, result.radar.stability, result.radar.power, result.radar.rhythm, result.radar.linkage]} target={[88, 85, 90, 86, 88]} pro={[95, 94, 96, 93, 95]} labels={["技術", "安定性", "出力", "リズム", "連動性"]} />
                  <div style={{ display: "flex", justifyContent: "center", gap: 18, fontSize: 11, marginTop: 4 }}>
                    <Leg c={C.green} t="あなた" /><Leg c={C.gold} t="目標" dash /><Leg c={C.faint} t="プロ" dash />
                  </div>
                </div>

                {/* 課題 */}
                <div style={S.card}>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14 }}>改善ポイント<span style={{ fontSize: 11, color: C.faint, fontWeight: 400, marginLeft: 6 }}>優先度順</span></div>
                  {result.issues?.map((iss, i) => (
                    <div key={i} style={{ display: "flex", gap: 13, padding: "12px 0", borderTop: i ? `1px solid ${C.line}` : "none" }}>
                      <div style={{ fontFamily: serif, fontSize: 20, fontWeight: 600, color: C.gold, lineHeight: 1, width: 22 }}>{i + 1}</div>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                          <span style={{ fontSize: 14, fontWeight: 700 }}>{iss.title}</span>
                          <span style={{ fontSize: 10, color: C.green, border: `1px solid ${C.green}44`, padding: "1px 8px", borderRadius: 9 }}>{iss.phase}</span>
                        </div>
                        <div style={{ fontSize: 12, color: C.sub }}>{iss.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* AIアドバイス */}
                <div style={{ ...S.card, background: C.goldSoft }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 13 }}>
                    <div style={{ width: 32, height: 32, borderRadius: 10, background: C.gold, display: "flex", alignItems: "center", justifyContent: "center" }}><IconChat size={17} stroke={C.card} /></div>
                    <div style={{ fontSize: 13, fontWeight: 700 }}>AIコーチからの助言</div>
                  </div>
                  {result.advice?.map((a, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, marginBottom: 9, alignItems: "flex-start" }}>
                      <span style={{ color: C.gold, fontWeight: 700, fontFamily: serif }}>—</span>
                      <div style={{ fontSize: 13, lineHeight: 1.6 }}>{a}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===== 再生 ===== */}
        {tab === "practice" && practiceView === "review" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 16 }}>
              <div><div style={S.eyebrow}>Review</div><div style={S.heading}>{compareMode ? "プロと比較" : "スイング再生"}</div></div>
              <button onClick={() => setCompareMode((c) => !c)} style={chip(compareMode)}><IconCompare size={14} stroke={compareMode ? C.card : C.sub} />比較</button>
            </div>
            {compareMode && <div style={{ ...S.card, padding: 16 }}><div style={{ fontSize: 12, fontWeight: 700, color: C.gold, marginBottom: 10 }}>お手本プロ</div><Player proMode compact /></div>}
            <div style={S.card}>
              {compareMode && <div style={{ fontSize: 12, fontWeight: 700, color: C.green, marginBottom: 10 }}>あなた</div>}
              {videoUrl && poseFrames
                ? <PoseReviewPlayer videoUrl={videoUrl} poseFrames={poseFrames} />
                : <>
                    {!videoUrl && <div style={{ fontSize: 12, color: C.faint, textAlign: "center", marginBottom: 10 }}>「撮影・解析」でスイングを撮ると、実写＋骨格で再生できます（プレビュー版はデモ表示）</div>}
                    <Player compact={compareMode} />
                  </>}
            </div>
          </div>
        )}

        {/* ===== 練習記録（スイング解析の履歴） ===== */}
        {tab === "log" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ marginBottom: 16 }}><div style={S.eyebrow}>Practice</div><div style={S.heading}>練習記録</div></div>

            {/* サマリー */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
              <Mini icon={<IconFlame size={20} stroke={C.gold} />} label="連続日数" value="4" unit="日" />
              <Mini icon={<IconTarget size={20} stroke={C.green} />} label="今週の練習" value="6" unit="回" />
              <Mini icon={<IconCamera size={20} stroke={C.green} />} label="総スイング" value="128" unit="" />
              <Mini icon={<IconAward size={20} stroke={C.gold} />} label="最高スコア" value="84" unit="点" />
            </div>

            {/* スイングスコア推移 */}
            <div style={S.card}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>スイングスコア推移</div>
              <ScoreGraph data={[71, 74, 76, 79, 84]} />
            </div>

            {/* 練習カレンダー */}
            <div style={S.card}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>練習カレンダー<span style={{ fontSize: 11, color: C.faint, fontWeight: 400, marginLeft: 6 }}>2026年6月</span></div>
              <Calendar />
            </div>

            {/* スイング解析の履歴 */}
            <div style={{ fontSize: 11, letterSpacing: "1px", color: C.faint, marginBottom: 10, textTransform: "uppercase" }}>解析履歴</div>
            {[["06/13", "DRV", 43.2, 205, "A"], ["06/12", "7I", 38.0, 152, "B+"], ["06/10", "DRV", 41.5, 198, "B"], ["06/08", "DRV", 40.1, 192, "B"]].map(([d, club2, sp, dist, rate], i) => (
              <div key={i} style={{ ...S.card, padding: 14, marginBottom: 8, display: "flex", alignItems: "center", gap: 13 }}>
                <div style={{ width: 42, height: 42, borderRadius: 11, background: C.greenSoft, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 800, color: C.green, fontFamily: mono }}>{club2}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: C.faint, marginBottom: 3 }}>2026/{d}</div>
                  <div style={{ display: "flex", gap: 12, fontSize: 13 }}>
                    <span>HS <b style={{ color: C.green, fontFamily: mono }}>{sp}</b></span>
                    <span>飛距離 <b style={{ color: C.greenDark, fontFamily: mono }}>{dist}</b>y</span>
                  </div>
                </div>
                <div style={{ fontSize: 18, fontWeight: 900, color: C.green, fontFamily: mono }}>{rate}</div>
              </div>
            ))}
          </div>
        )}

        {/* ===== コーチ（チャット / お手本） ===== */}
        {tab === "coach" && (
          <div style={{ marginBottom: 18 }}>
            <div style={{ marginBottom: 16 }}><div style={S.eyebrow}>Coaching</div><div style={S.heading}>AIコーチ</div></div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => setCoachView("chat")} style={chip(coachView === "chat")}>チャット相談</button>
              <button onClick={() => setCoachView("pro")} style={chip(coachView === "pro")}>お手本スイング</button>
            </div>
          </div>
        )}

        {tab === "coach" && coachView === "chat" && (
          <div style={{ animation: "fade .35s ease", display: "flex", flexDirection: "column", height: "calc(100vh - 260px)", minHeight: 380 }}>
            <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, paddingBottom: 12 }}>
              {chatMsgs.map((m, i) => (
                <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
                  <div style={{ maxWidth: "82%", padding: "11px 14px", borderRadius: 16, fontSize: 13, lineHeight: 1.6,
                    background: m.role === "user" ? C.green : C.card, color: m.role === "user" ? "#fff" : C.ink,
                    border: m.role === "user" ? "none" : `1px solid ${C.line}`,
                    borderBottomRightRadius: m.role === "user" ? 4 : 16, borderBottomLeftRadius: m.role === "user" ? 16 : 4 }}>{m.text}</div>
                </div>
              ))}
              {chatLoading && (
                <div style={{ display: "flex", justifyContent: "flex-start" }}>
                  <div style={{ padding: "12px 16px", borderRadius: 16, background: C.card, border: `1px solid ${C.line}`, display: "flex", gap: 4 }}>
                    {[0, 1, 2].map((d) => <span key={d} style={{ width: 7, height: 7, borderRadius: "50%", background: C.faint, animation: `pulse 1s ${d * 0.2}s infinite` }} />)}
                  </div>
                </div>
              )}
            </div>
            {!result && <div style={{ fontSize: 11, color: C.faint, marginBottom: 8, textAlign: "center" }}>「練習」タブでスイングを解析すると、より具体的にアドバイスできます。</div>}
            <div style={{ display: "flex", gap: 8 }}>
              <input value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendChat()}
                placeholder="質問を入力..." style={{ flex: 1, padding: "13px 16px", borderRadius: 22, border: `1px solid ${C.line}`, fontSize: 14, outline: "none", background: C.card }} />
              <button onClick={sendChat} disabled={chatLoading} style={{ width: 48, height: 48, borderRadius: "50%", background: C.green, color: "#fff", border: "none", fontSize: 18, cursor: "pointer", flexShrink: 0 }}>↑</button>
            </div>
          </div>
        )}

        {/* ===== お手本 ===== */}
        {tab === "coach" && coachView === "pro" && (
          <div style={{ animation: "fade .35s ease" }}>
            {[{ n: "ドライバー基本", s: "正面 · 骨格解説", t: "DR", l: "初級" }, { n: "アイアンのダウンブロー", s: "上から捉える", t: "7I", l: "中級" }, { n: "ドロー回転", s: "インサイドアウト", t: "DR", l: "上級" }].map((p, i) => (
              <div key={i} style={{ ...S.card, padding: 0, overflow: "hidden" }}>
                <div style={{ position: "relative", aspectRatio: "16/9" }}>
                  <SwingStage frame={50} withSkeleton />
                  <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <div style={{ width: 46, height: 46, borderRadius: "50%", background: "rgba(251,249,244,0.92)", display: "flex", alignItems: "center", justifyContent: "center", color: C.green }}><IconPlay size={17} /></div>
                  </div>
                  <div style={{ position: "absolute", top: 10, left: 10, background: "rgba(22,43,59,0.5)", color: "#fff", fontSize: 10, fontWeight: 700, padding: "3px 9px", borderRadius: 10, fontFamily: mono }}>{p.t}</div>
                  <div style={{ position: "absolute", top: 10, right: 10, background: C.gold, color: C.ink, fontSize: 10, fontWeight: 700, padding: "3px 9px", borderRadius: 10 }}>{p.l}</div>
                </div>
                <div style={{ padding: 16 }}><div style={{ fontSize: 15, fontWeight: 700 }}>{p.n}</div><div style={{ fontSize: 12, color: C.faint, marginTop: 3 }}>{p.s}</div></div>
              </div>
            ))}
          </div>
        )}

        {/* ===== 設定 ===== */}
        {tab === "settings" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ marginBottom: 16 }}><div style={S.eyebrow}>Settings</div><div style={S.heading}>設定</div></div>

            {/* プロフィール */}
            <div style={{ ...S.card, display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ width: 54, height: 54, borderRadius: "50%", background: `linear-gradient(135deg,${C.green},${C.greenDark})`, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 22, fontWeight: 800 }}>小</div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 16, fontWeight: 700 }}>小西 公幸</div>
                <div style={{ fontSize: 12, color: C.faint }}>一般ユーザー · ID: a</div>
              </div>
              <IconNext size={18} stroke={C.faint} />
            </div>

            {/* 各種設定 */}
            <div style={{ ...S.card, padding: "4px 18px" }}>
              {[["クラブセッティング", "14本"], ["目標スコア", "85"], ["ホームコース", "未設定"], ["使用ボール", "未選択"]].map(([l, v], i, arr) => (
                <div key={l} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 0", borderBottom: i < arr.length - 1 ? `1px solid ${C.line}` : "none" }}>
                  <span style={{ fontSize: 14 }}>{l}</span>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span style={{ fontSize: 13, color: C.faint }}>{v}</span><IconNext size={16} stroke={C.faint} /></div>
                </div>
              ))}
            </div>

            {/* 通知トグル */}
            <div style={{ ...S.card, padding: "4px 18px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 0", borderBottom: `1px solid ${C.line}` }}>
                <div><div style={{ fontSize: 14, fontWeight: 600 }}>LINE通知</div><div style={{ fontSize: 11, color: C.faint }}>練習リマインド等</div></div>
                <Toggle on={true} onClick={() => {}} />
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 0" }}>
                <div><div style={{ fontSize: 14, fontWeight: 600 }}>メール通知</div><div style={{ fontSize: 11, color: C.faint }}>月次レポート</div></div>
                <Toggle on={false} onClick={() => {}} />
              </div>
            </div>

            {/* プラン */}
            <div style={{ ...S.card, background: `linear-gradient(135deg,${C.green},${C.greenDark})`, color: "#fff" }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>プレミアムプラン</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.85)", marginBottom: 12 }}>GPS無制限・コースマップ・解析履歴保存</div>
              <button style={{ width: "100%", padding: 11, borderRadius: 11, background: "#fff", color: C.green, border: "none", fontSize: 13, fontWeight: 700, cursor: "pointer" }}>プランを見る</button>
            </div>

            <button onClick={onLogout} style={{ width: "100%", padding: 15, borderRadius: 14, background: C.card, border: `1px solid ${C.line}`, color: C.red, fontSize: 14, fontWeight: 700, cursor: "pointer" }}>ログアウト</button>
            <div style={{ textAlign: "center", fontSize: 11, color: C.faint, marginTop: 16 }}>Golog v1.0.0</div>
          </div>
        )}
      </div>

      <div style={S.nav}>
        {[{ k: "round", I: IconFlag, l: roundActive ? "記録中" : "ラウンド" }, { k: "practice", I: IconCamera, l: "練習" }, { k: "log", I: IconChart, l: "練習記録" }, { k: "coach", I: IconChat, l: "コーチ" }, { k: "settings", I: IconGear, l: "設定" }].map((n) => {
          const on = tab === n.k; const I = n.I;
          const recording = n.k === "round" && roundActive;
          return <div key={n.k} style={S.navItem(on)} onClick={() => setTab(n.k)}>
            <div style={{ position: "relative" }}>
              <I size={21} stroke={recording ? C.red : on ? C.green : C.faint} />
              {recording && <span style={{ position: "absolute", top: -2, right: -3, width: 7, height: 7, borderRadius: "50%", background: C.red, animation: "pulse 1s infinite" }} />}
            </div>
            <span style={{ color: recording ? C.red : undefined }}>{n.l}</span>
          </div>;
        })}
      </div>
    </div>
  );
}

function pBtn(disabled) { return { width: "100%", padding: 17, marginTop: 16, borderRadius: 15, background: disabled ? C.cardAlt : C.green, color: disabled ? C.faint : "#fff", fontSize: 15, fontWeight: 700, border: "none", cursor: disabled ? "default" : "pointer", letterSpacing: "0.5px", fontFamily: serif }; }
function Modal({ title, onClose, children }) {
  return <div style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.4)", display: "flex", alignItems: "flex-end", justifyContent: "center" }} onClick={onClose}>
    <div onClick={(e) => e.stopPropagation()} style={{ width: "100%", maxWidth: 440, maxHeight: "88vh", overflowY: "auto", background: C.bg, borderRadius: "22px 22px 0 0", padding: 20, animation: "slideUp .3s ease" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
        <div style={{ fontSize: 17, fontWeight: 800 }}>{title}</div>
        <div onClick={onClose} style={{ width: 30, height: 30, borderRadius: "50%", background: C.cardAlt, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", fontSize: 16, color: C.sub }}>×</div>
      </div>
      {children}
    </div>
  </div>;
}
function Row({ label, children }) {
  return <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 0", borderBottom: `1px solid ${C.line}` }}>
    <span style={{ fontSize: 13, color: C.sub }}>{label}</span>{children}
  </div>;
}
function StackedBars() {
  const rounds = [{ d: "4/17", s: 104, p: 40 }, { d: "4/19", s: 108, p: 38 }, { d: "4/21", s: 93, p: 36 }, { d: "4/27", s: 86, p: 34 }, { d: "5/4", s: 85, p: 31 }, { d: "5/5", s: 98, p: 30 }, { d: "5/8", s: 84, p: 35 }, { d: "5/9", s: 87, p: 33 }, { d: "5/25", s: 82, p: 34 }];
  const maxS = 115, goal = 100, W = 340, H = 175, padB = 24, padT = 12, gap = W / rounds.length, bw = gap * 0.56;
  const gy = padT + (1 - goal / maxS) * (H - padT - padB);
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
    <line x1="0" y1={gy} x2={W} y2={gy} stroke={C.goal} strokeWidth="1.5" strokeDasharray="4,3" />
    {rounds.map((r, i) => {
      const x = gap * i + (gap - bw) / 2, tH = (r.s / maxS) * (H - padT - padB), pH = (r.p / maxS) * (H - padT - padB), y = H - padB - tH;
      return <g key={i}>
        <rect x={x} y={y} width={bw} height={tH - pH} rx="3" fill={C.greenLite} />
        <rect x={x} y={H - padB - pH} width={bw} height={pH} rx="3" fill={C.gold} />
        <text x={x + bw / 2} y={y - 4} textAnchor="middle" fontSize="10" fontWeight="700" fill="#fff" fontFamily={mono}>{r.s}</text>
        <text x={x + bw / 2} y={H - 7} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.8)">{r.d}</text>
      </g>;
    })}
  </svg>;
}
function Donut() {
  const segs = [[0.0, "#E0A82E"], [3.7, C.red], [26.5, C.green], [35.8, C.gold], [34.0, C.faint]];
  const cx = 90, cy = 90, r = 62, sw = 26, circ = 2 * Math.PI * r; let acc = 0;
  return <svg viewBox="0 0 180 180" style={{ width: 160, height: 160 }}>
    {segs.map(([v, c], i) => { const frac = v / 100, dash = frac * circ;
      const el = <circle key={i} cx={cx} cy={cy} r={r} fill="none" stroke={c} strokeWidth={sw} strokeDasharray={`${dash} ${circ - dash}`} strokeDashoffset={-acc * circ} transform={`rotate(-90 ${cx} ${cy})`} />;
      acc += frac; return el; })}
    <text x={cx} y={cy - 2} textAnchor="middle" fontSize="32" fontWeight="800" fill={C.ink} fontFamily={mono}>90.5</text>
    <text x={cx} y={cy + 16} textAnchor="middle" fontSize="10" fill={C.faint} letterSpacing="1">SCORE</text>
  </svg>;
}
function DistCol({ label, value }) {
  return <div style={{ textAlign: "center", paddingBottom: 6 }}>
    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.7)", marginBottom: 4 }}>{label}</div>
    <div style={{ fontSize: 28, fontWeight: 700, fontFamily: mono, color: "rgba(255,255,255,0.92)", lineHeight: 1 }}>{value}</div>
  </div>;
}
function Stepper({ label, value, onMinus, onPlus, highlight }) {
  return <div style={{ flex: 1, background: C.cardAlt, borderRadius: 14, padding: 14, textAlign: "center" }}>
    <div style={{ fontSize: 11, color: C.faint, marginBottom: 10 }}>{label}</div>
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
      <div onClick={onMinus} style={stepBtn}><IconMinus size={16} /></div>
      <span style={{ fontSize: 30, fontWeight: 700, fontFamily: mono, color: highlight ? C.green : C.ink, minWidth: 40 }}>{value}</span>
      <div onClick={onPlus} style={stepBtn}><IconPlus size={16} /></div>
    </div>
  </div>;
}
const stepBtn = { width: 36, height: 36, borderRadius: "50%", background: C.card, border: `1px solid ${C.line}`, display: "flex", alignItems: "center", justifyContent: "center", color: C.green, cursor: "pointer" };
function ScoreCell({ no, par, score, active, onClick }) {
  const diff = score != null ? score - par : null;
  const bg = score == null ? C.card : diff < 0 ? C.greenSoft : diff === 0 ? C.card : C.goldSoft;
  return <div onClick={onClick} style={{ aspectRatio: "0.72", borderRadius: 7, border: `1px solid ${active ? C.green : C.line}`, background: bg, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", cursor: "pointer", padding: "2px 0" }}>
    <span style={{ fontSize: 8, color: C.faint, fontFamily: mono }}>{no}</span>
    <span style={{ fontSize: 13, fontWeight: 700, fontFamily: mono, color: score == null ? C.faint : diff < 0 ? C.green : C.ink }}>{score ?? "·"}</span>
  </div>;
}
const badgeMid = { position: "absolute", top: 14, left: "50%", transform: "translateX(-50%)", background: "rgba(22,43,59,0.55)", color: "#F4F0E8", fontSize: 12, fontWeight: 600, padding: "6px 14px", borderRadius: 16, backdropFilter: "blur(4px)", display: "flex", alignItems: "center", gap: 7 };
const dot = (c) => ({ width: 7, height: 7, borderRadius: "50%", background: c, animation: "pulse 1s infinite", display: "inline-block" });
function BigMetric({ label, value, unit }) {
  return <div style={{ flex: 1, background: C.card, borderRadius: 18, padding: 18, boxShadow: "0 1px 2px rgba(42,38,32,0.04),0 8px 24px rgba(42,38,32,0.04)" }}>
    <div style={{ fontSize: 10, letterSpacing: "1.5px", color: C.faint, marginBottom: 8 }}>{label}</div>
    <div style={{ display: "flex", alignItems: "baseline", gap: 3 }}>
      <span style={{ fontSize: 30, fontWeight: 700, color: C.green, fontFamily: mono, lineHeight: 1 }}>{value}</span>
      <span style={{ fontSize: 12, color: C.faint, fontFamily: mono }}>{unit}</span></div>
  </div>;
}
function Mini({ icon, label, value, unit }) {
  return <div style={{ ...S.card, marginBottom: 0, padding: 16, display: "flex", alignItems: "center", gap: 12 }}>
    {icon}<div><div style={{ fontSize: 11, color: C.faint }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: mono }}>{value}<span style={{ fontSize: 11, color: C.faint, marginLeft: 2 }}>{unit}</span></div></div>
  </div>;
}
function Leg({ c, t, dash }) { return <div style={{ display: "flex", alignItems: "center", gap: 5 }}><span style={{ width: 14, borderTop: `2px ${dash ? "dashed" : "solid"} ${c}` }} /><span style={{ color: C.sub }}>{t}</span></div>; }
function ScoreGraph({ data }) {
  const min = Math.min(...data) - 5, max = Math.max(...data) + 3, W = 300, H = 120, pad = 14;
  const pts = data.map((v, i) => [pad + (i / (data.length - 1)) * (W - pad * 2), H - pad - ((v - min) / (max - min)) * (H - pad * 2)]);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
    <defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={C.green} stopOpacity="0.14" /><stop offset="100%" stopColor={C.green} stopOpacity="0" /></linearGradient></defs>
    <path d={`${path} L${pts[pts.length - 1][0]},${H} L${pts[0][0]},${H} Z`} fill="url(#sg)" />
    <path d={path} stroke={C.green} strokeWidth="2.2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    {pts.map((p, i) => <g key={i}><circle cx={p[0]} cy={p[1]} r={i === pts.length - 1 ? 4 : 3} fill={i === pts.length - 1 ? C.green : C.card} stroke={C.green} strokeWidth="2" /><text x={p[0]} y={p[1] - 9} textAnchor="middle" fontSize="10" fontWeight="700" fill={C.ink} fontFamily={mono}>{data[i]}</text></g>)}
  </svg>;
}
function Calendar() {
  const days = ["日", "月", "火", "水", "木", "金", "土"];
  const practiced = { 1: 32, 2: 41, 4: 27, 5: 52, 6: 38, 8: 45, 10: 29, 12: 25, 13: "today" };
  const cells = []; for (let d = 1; d <= 30; d++) cells.push(d);
  return <div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 4, marginBottom: 8 }}>{days.map((d) => <div key={d} style={{ textAlign: "center", fontSize: 10, color: C.faint }}>{d}</div>)}</div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 4 }}>
      {cells.map((d) => { const p = practiced[d]; const today = p === "today"; const has = p && !today;
        return <div key={d} style={{ aspectRatio: "1", borderRadius: 9, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: today ? C.green : has ? C.greenSoft : "transparent" }}>
          <span style={{ fontSize: 11, fontWeight: has || today ? 700 : 400, color: today ? C.card : C.ink, fontFamily: mono }}>{d}</span>
          {has && <span style={{ fontSize: 8, color: C.green, fontWeight: 700, fontFamily: mono }}>{p}</span>}
        </div>; })}
    </div>
  </div>;
}

/* ============================================================
   管理画面（管理者 b 専用）
   ============================================================ */
function AdminApp({ onLogout }) {
  const [atab, setAtab] = useState("courses");
  // ゴルフ場
  const [courses, setCourses] = useState([
    { id: 1, name: "リバーサイドパーク長門石ゴルフ場", holes: 18, greens: 18 },
    { id: 2, name: "みやきリンクス愛しとーとゴルフクラブ", holes: 18, greens: 6 },
  ]);
  const [showCourseForm, setShowCourseForm] = useState(false);
  const defaultHoles = (n) => Array.from({ length: n }, (_, i) => ({ no: i + 1, par: 4, yard: 360 }));
  const [newCourse, setNewCourse] = useState({ name: "", holes: 18, holeData: defaultHoles(18) });
  // グリーン登録
  const [greenCourse, setGreenCourse] = useState(null);
  const [greenHole, setGreenHole] = useState(1);
  const [pins, setPins] = useState({ front: null, center: null, back: null });
  const [pinTarget, setPinTarget] = useState("center");
  // ユーザー
  const [users, setUsers] = useState([
    { id: "a", name: "小西 公幸", pw: "a", role: "user", rounds: 10 },
    { id: "b", name: "管理者", pw: "b", role: "admin", rounds: 0 },
    { id: "tanaka", name: "田中 太郎", pw: "pass123", role: "user", rounds: 24 },
  ]);
  const [showUserForm, setShowUserForm] = useState(false);
  const [newUser, setNewUser] = useState({ id: "", name: "", pw: "", role: "user" });

  const setHoles = (n) => setNewCourse((p) => ({ ...p, holes: n, holeData: defaultHoles(n) }));
  const editHole = (idx, field, val) => setNewCourse((p) => ({ ...p, holeData: p.holeData.map((h, i) => i === idx ? { ...h, [field]: val } : h) }));
  const addCourse = () => {
    if (!newCourse.name.trim()) return;
    setCourses((c) => [...c, { id: Date.now(), name: newCourse.name, holes: newCourse.holes, greens: 0, holeData: newCourse.holeData }]);
    setNewCourse({ name: "", holes: 18, holeData: defaultHoles(18) }); setShowCourseForm(false);
  };
  const addUser = () => {
    if (!newUser.id.trim() || !newUser.name.trim() || !newUser.pw.trim()) return;
    setUsers((u) => [...u, { ...newUser, rounds: 0 }]);
    setNewUser({ id: "", name: "", pw: "", role: "user" }); setShowUserForm(false);
  };

  return (
    <div style={{ ...S.app }}>
      <style>{`
        @keyframes fade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        @keyframes slideUp{from{transform:translateY(100%)}to{transform:translateY(0)}}
        *{box-sizing:border-box;margin:0;padding:0}body{background:${C.bg}}
        ::-webkit-scrollbar{display:none}
      `}</style>

      {/* ヘッダー（管理者は濃色で区別） */}
      <div style={{ ...S.topbar, background: `linear-gradient(135deg,${C.greenDark},#0E3A56)`, borderBottom: "none" }}>
        <div>
          <div style={{ fontSize: 21, fontWeight: 800, letterSpacing: "0.5px", lineHeight: 1, color: "#fff" }}>Golog</div>
          <div style={{ fontSize: 9, letterSpacing: "3px", color: C.goal, marginTop: 3, textTransform: "uppercase" }}>Admin Console</div>
        </div>
        <div onClick={onLogout} style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 16, background: "rgba(255,255,255,0.2)", cursor: "pointer" }}>
          <IconLogout size={15} stroke="#fff" /><span style={{ fontSize: 12, fontWeight: 700, color: "#fff" }}>ログアウト</span>
        </div>
      </div>

      <div style={S.content}>
        {/* ゴルフ場登録 */}
        {atab === "courses" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 16 }}>
              <div><div style={S.eyebrow}>Admin</div><div style={S.heading}>ゴルフ場管理</div></div>
              <button onClick={() => setShowCourseForm(true)} style={{ padding: "9px 14px", borderRadius: 11, background: C.green, color: "#fff", border: "none", fontSize: 12, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", gap: 5 }}><IconPlus size={14} stroke="#fff" />新規</button>
            </div>
            {courses.map((c) => (
              <div key={c.id} style={{ ...S.card, padding: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>{c.name}</div>
                    <div style={{ display: "flex", gap: 12, fontSize: 12, color: C.sub }}>
                      <span>{c.holes}ホール</span>
                      <span>グリーン登録 <b style={{ color: c.greens === c.holes ? C.green : C.gold, fontFamily: mono }}>{c.greens}/{c.holes}</b></span>
                    </div>
                  </div>
                  <IconTrash size={16} stroke={C.faint} />
                </div>
                <button onClick={() => { setGreenCourse(c); setAtab("greens"); }} style={{ width: "100%", marginTop: 12, padding: 10, borderRadius: 10, background: C.greenSoft, color: C.green, border: "none", fontSize: 12, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}><IconPin size={14} stroke={C.green} />グリーン位置を登録</button>
              </div>
            ))}
            {showCourseForm && (
              <Modal title="ゴルフ場を登録" onClose={() => setShowCourseForm(false)}>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>コース名</div>
                <input value={newCourse.name} onChange={(e) => setNewCourse((p) => ({ ...p, name: e.target.value }))} placeholder="〇〇カントリークラブ" style={inp} />
                <div style={{ fontSize: 12, fontWeight: 700, margin: "14px 0 6px" }}>ホール数</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {[9, 18, 27, 36].map((h) => (
                    <button key={h} onClick={() => setHoles(h)} style={{ flex: 1, padding: 12, borderRadius: 10, fontSize: 13, fontWeight: 700, border: `1px solid ${newCourse.holes === h ? C.green : C.line}`, background: newCourse.holes === h ? C.green : C.card, color: newCourse.holes === h ? "#fff" : C.sub, cursor: "pointer" }}>{h}</button>
                  ))}
                </div>
                {/* 各ホールのPAR・ヤード */}
                <div style={{ fontSize: 12, fontWeight: 700, margin: "16px 0 8px" }}>各ホール設定</div>
                <div style={{ display: "flex", fontSize: 10, color: C.faint, padding: "0 4px 6px" }}>
                  <span style={{ width: 44 }}>ホール</span><span style={{ flex: 1, textAlign: "center" }}>PAR</span><span style={{ flex: 1, textAlign: "center" }}>ヤード</span>
                </div>
                <div style={{ maxHeight: 240, overflowY: "auto", border: `1px solid ${C.line}`, borderRadius: 10 }}>
                  {newCourse.holeData.map((h, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 8px", borderBottom: i < newCourse.holeData.length - 1 ? `1px solid ${C.line}` : "none" }}>
                      <span style={{ width: 36, fontSize: 12, fontWeight: 700, fontFamily: mono }}>{h.no}</span>
                      <div style={{ flex: 1, display: "flex", gap: 4, justifyContent: "center" }}>
                        {[3, 4, 5].map((p) => (
                          <button key={p} onClick={() => editHole(i, "par", p)} style={{ width: 30, height: 28, borderRadius: 7, fontSize: 12, fontWeight: 700, fontFamily: mono, border: `1px solid ${h.par === p ? C.green : C.line}`, background: h.par === p ? C.green : C.card, color: h.par === p ? "#fff" : C.sub, cursor: "pointer" }}>{p}</button>
                        ))}
                      </div>
                      <input type="number" value={h.yard} onChange={(e) => editHole(i, "yard", +e.target.value)} style={{ flex: 1, width: "100%", padding: "6px 8px", borderRadius: 7, border: `1px solid ${C.line}`, fontSize: 12, fontFamily: mono, textAlign: "center", outline: "none" }} />
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11, color: C.faint, marginTop: 8, textAlign: "center" }}>合計PAR {newCourse.holeData.reduce((a, h) => a + h.par, 0)} · 全長 {newCourse.holeData.reduce((a, h) => a + h.yard, 0)}y</div>
                <button onClick={addCourse} style={{ ...pBtn(!newCourse.name.trim()) }}>登録する</button>
              </Modal>
            )}
          </div>
        )}

        {/* グリーン位置登録 */}
        {atab === "greens" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ marginBottom: 16 }}>
              <div style={S.eyebrow}>Admin</div>
              <div style={S.heading}>グリーン位置登録</div>
              <div style={{ fontSize: 12, color: C.faint, marginTop: 4 }}>{greenCourse?.name || "コース未選択"}</div>
            </div>
            {!greenCourse ? (
              <div style={{ ...S.card, textAlign: "center", padding: 30, color: C.faint }}>ゴルフ場管理から対象コースを選んでください</div>
            ) : (
              <>
                {/* ホール選択 */}
                <div style={{ ...S.card, padding: "12px 14px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div onClick={() => { setGreenHole((h) => Math.max(1, h - 1)); setPins({ front: null, center: null, back: null }); }} style={ctrlBtn}><IconBack size={16} /></div>
                  <div style={{ textAlign: "center" }}><div style={{ fontSize: 10, letterSpacing: "2px", color: C.gold }}>HOLE</div><div style={{ fontSize: 28, fontWeight: 800, lineHeight: 1 }}>{greenHole}</div></div>
                  <div onClick={() => { setGreenHole((h) => Math.min(greenCourse.holes, h + 1)); setPins({ front: null, center: null, back: null }); }} style={ctrlBtn}><IconNext size={16} /></div>
                </div>
                {/* ピン種別 */}
                <div style={{ display: "flex", gap: 6, margin: "14px 0" }}>
                  {[["front", "前"], ["center", "中"], ["back", "奥"]].map(([k, l]) => (
                    <button key={k} onClick={() => setPinTarget(k)} style={{ flex: 1, padding: "9px", borderRadius: 9, fontSize: 12, fontWeight: 700, border: `1px solid ${pinTarget === k ? C.green : C.line}`, background: pinTarget === k ? C.green : C.card, color: pinTarget === k ? "#fff" : C.sub, cursor: "pointer" }}>{l}{pins[k] ? " ✓" : ""}</button>
                  ))}
                </div>
                {/* 衛星写真ピン留め */}
                <div onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setPins((p) => ({ ...p, [pinTarget]: { x: ((e.clientX - r.left) / r.width * 100).toFixed(1), y: ((e.clientY - r.top) / r.height * 100).toFixed(1) } })); }}
                  style={{ position: "relative", aspectRatio: "1", borderRadius: 14, overflow: "hidden", cursor: "crosshair", background: "radial-gradient(circle at 50% 45%,#4a7c3a,#2f5226 60%,#243f1d)" }}>
                  <div style={{ position: "absolute", top: "32%", left: "38%", width: "26%", height: "20%", background: "radial-gradient(circle,#7bb85f,#5a9442)", borderRadius: "50%" }} />
                  <div style={{ position: "absolute", top: "12%", right: "14%", width: "16%", height: "12%", background: "#c9b884", borderRadius: "40%", opacity: 0.7 }} />
                  {Object.entries(pins).map(([k, p]) => p && (
                    <div key={k} style={{ position: "absolute", left: `${p.x}%`, top: `${p.y}%`, transform: "translate(-50%,-100%)", textAlign: "center" }}>
                      <IconPin size={26} stroke="#fff" fill={k === "center" ? C.green : k === "front" ? C.gold : C.red} />
                      <div style={{ fontSize: 9, fontWeight: 700, color: "#fff", background: "rgba(0,0,0,0.5)", borderRadius: 6, padding: "1px 5px", marginTop: -4 }}>{k === "front" ? "前" : k === "center" ? "中" : "奥"}</div>
                    </div>
                  ))}
                  <div style={{ position: "absolute", bottom: 8, left: 8, fontSize: 9, color: "rgba(255,255,255,0.7)" }}>© OpenStreetMap（デモ）</div>
                </div>
                <button onClick={() => { setPins({ front: null, center: null, back: null }); }} disabled={!pins.center} style={{ ...pBtn(!pins.center) }}>{pins.center ? "保存して次のホールへ" : "中央を登録してください"}</button>
              </>
            )}
          </div>
        )}

        {/* ユーザー管理 */}
        {atab === "users" && (
          <div style={{ animation: "fade .35s ease" }}>
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 16 }}>
              <div><div style={S.eyebrow}>Admin</div><div style={S.heading}>ユーザー管理</div></div>
              <button onClick={() => setShowUserForm(true)} style={{ padding: "9px 14px", borderRadius: 11, background: C.green, color: "#fff", border: "none", fontSize: 12, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", gap: 5 }}><IconPlus size={14} stroke="#fff" />追加</button>
            </div>
            <div style={{ ...S.card, background: C.greenSoft, padding: "11px 16px", display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: C.sub }}>登録ユーザー数</span><b style={{ color: C.green }}>{users.length}</b>
            </div>
            {users.map((u) => (
              <div key={u.id} style={{ ...S.card, padding: 14, display: "flex", alignItems: "center", gap: 13 }}>
                <div style={{ width: 42, height: 42, borderRadius: "50%", background: u.role === "admin" ? C.greenDark : C.greenSoft, color: u.role === "admin" ? "#fff" : C.green, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 800 }}>{u.name[0]}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, display: "flex", alignItems: "center", gap: 7 }}>{u.name}
                    {u.role === "admin" && <span style={{ fontSize: 9, fontWeight: 700, color: "#fff", background: C.greenDark, padding: "1px 7px", borderRadius: 8 }}>管理者</span>}
                  </div>
                  <div style={{ fontSize: 11, color: C.faint }}>ID: {u.id} · {u.rounds}ラウンド</div>
                </div>
                <IconTrash size={16} stroke={C.faint} />
              </div>
            ))}
            {showUserForm && (
              <Modal title="ユーザーを追加" onClose={() => setShowUserForm(false)}>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>ユーザーID（ログイン用）</div>
                <input value={newUser.id} onChange={(e) => setNewUser((p) => ({ ...p, id: e.target.value }))} placeholder="login_id" autoCapitalize="none" style={inp} />
                <div style={{ fontSize: 12, fontWeight: 700, margin: "14px 0 6px" }}>パスワード</div>
                <input value={newUser.pw} onChange={(e) => setNewUser((p) => ({ ...p, pw: e.target.value }))} placeholder="パスワード" style={inp} />
                <div style={{ fontSize: 12, fontWeight: 700, margin: "14px 0 6px" }}>表示名</div>
                <input value={newUser.name} onChange={(e) => setNewUser((p) => ({ ...p, name: e.target.value }))} placeholder="山田 太郎" style={inp} />
                <div style={{ fontSize: 12, fontWeight: 700, margin: "14px 0 6px" }}>権限</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {[["user", "一般ユーザー"], ["admin", "管理者"]].map(([k, l]) => (
                    <button key={k} onClick={() => setNewUser((p) => ({ ...p, role: k }))} style={{ flex: 1, padding: 12, borderRadius: 10, fontSize: 13, fontWeight: 700, border: `1px solid ${newUser.role === k ? C.green : C.line}`, background: newUser.role === k ? C.green : C.card, color: newUser.role === k ? "#fff" : C.sub, cursor: "pointer" }}>{l}</button>
                  ))}
                </div>
                <button onClick={addUser} style={{ ...pBtn(!newUser.id.trim() || !newUser.name.trim() || !newUser.pw.trim()) }}>追加する</button>
              </Modal>
            )}
          </div>
        )}
      </div>

      <div style={S.nav}>
        {[{ k: "courses", I: IconCourse, l: "ゴルフ場" }, { k: "greens", I: IconPin, l: "グリーン" }, { k: "users", I: IconUsers, l: "ユーザー" }].map((n) => {
          const on = atab === n.k; const I = n.I;
          return <div key={n.k} style={S.navItem(on)} onClick={() => setAtab(n.k)}><I size={21} stroke={on ? C.green : C.faint} /><span>{n.l}</span></div>;
        })}
      </div>
    </div>
  );
}
const inp = { width: "100%", padding: "12px 14px", borderRadius: 10, border: `1px solid ${C.line}`, fontSize: 14, outline: "none", background: C.card };

/* ============================================================
   ログイン + ルーティング
   ============================================================ */
export default function SwingLab() {
  const [session, setSession] = useState(null); // null / "user" / "admin"
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [err, setErr] = useState("");

  const login = () => {
    if (id === "a" && pw === "a") { setSession("user"); setErr(""); }
    else if (id === "b" && pw === "b") { setSession("admin"); setErr(""); }
    else setErr("IDまたはパスワードが違います");
  };
  const logout = () => { setSession(null); setId(""); setPw(""); };

  if (session === "user") return <UserApp onLogout={logout} />;
  if (session === "admin") return <AdminApp onLogout={logout} />;

  // ログイン画面
  return (
    <div style={{ minHeight: "100vh", maxWidth: 440, margin: "0 auto", background: `linear-gradient(160deg,${C.green},${C.greenDark} 70%,#0E3A56)`, display: "flex", flexDirection: "column", justifyContent: "center", padding: 28, fontFamily: "-apple-system,'Hiragino Sans',sans-serif" }}>
      <style>{`*{box-sizing:border-box;margin:0;padding:0}body{background:${C.greenDark}}`}</style>
      <div style={{ textAlign: "center", marginBottom: 36 }}>
        <div style={{ fontSize: 38, fontWeight: 800, color: "#fff", letterSpacing: "1px" }}>Golog</div>
        <div style={{ fontSize: 11, letterSpacing: "4px", color: "rgba(255,255,255,0.6)", marginTop: 6, textTransform: "uppercase" }}>ゴーログ · Golf Log</div>
      </div>
      <div style={{ background: "rgba(255,255,255,0.97)", borderRadius: 22, padding: 26 }}>
        <div style={{ fontSize: 18, fontWeight: 800, marginBottom: 20, textAlign: "center" }}>ログイン</div>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.sub, marginBottom: 6 }}>ユーザーID</div>
        <input value={id} onChange={(e) => setId(e.target.value)} placeholder="ID" autoCapitalize="none" style={{ ...inp, marginBottom: 14 }} />
        <div style={{ fontSize: 12, fontWeight: 700, color: C.sub, marginBottom: 6 }}>パスワード</div>
        <input value={pw} onChange={(e) => setPw(e.target.value)} onKeyDown={(e) => e.key === "Enter" && login()} type="password" placeholder="パスワード" style={inp} />
        {err && <div style={{ fontSize: 12, color: C.red, marginTop: 12, textAlign: "center" }}>{err}</div>}
        <button onClick={login} style={{ width: "100%", marginTop: 20, padding: 15, borderRadius: 13, background: C.green, color: "#fff", border: "none", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>ログイン</button>
        <div style={{ marginTop: 18, padding: 12, background: C.cardAlt, borderRadius: 11, fontSize: 11, color: C.sub, lineHeight: 1.7 }}>
          <b>テスト用アカウント</b><br />
          一般ユーザー: <b style={{ fontFamily: mono }}>a / a</b><br />
          管理者: <b style={{ fontFamily: mono }}>b / b</b>
        </div>
      </div>
    </div>
  );
}
