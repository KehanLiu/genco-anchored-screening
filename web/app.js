"use strict";

// Map canvas in viewBox units; bus coordinates from the API are in [0, 1].
const W = 1000, H = 720, PAD = 44;
const PARALLEL_GAP = 7;
const LABEL_ZOOM = 2;

// One-hue ramps, low to high. Branch quantities are blue, bus quantities orange.
// Dark mode anchors the low end at the dark step so "near zero" recedes.
const RAMPS = {
  light: {
    branch: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
    bus: ["#ee9670", "#eb6834", "#c9521f", "#9c3d14", "#6a280b"],
  },
  dark: {
    branch: ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
    bus: ["#8f3712", "#b84a1c", "#d95926", "#f09a73", "#f9d2c0"],
  },
};

// ---------- text, in English and Chinese ----------

const TEXT = {
  en: {
    brand: "GENCO what-if",
    mapShows: "Map shows",
    viewGenco: "GENCO",
    viewAc: "AC solver",
    viewDiff: "Difference",
    viewResidual: "GENCO residual",
    load: "Load",
    solving: "Solving…",
    gridMap: "Grid map",
    hint: "Click a line to trip it, click again to restore. Scroll to zoom, drag to pan.",
    scenario: "Scenario",
    linesOut: "Lines out",
    reset: "Reset scenario",
    advisor: "GENCO advisor",
    agreement: "Checked against the AC solver",
    ranked: "Largest line disagreements",
    solveTime: "Solve time",
    line: (n) => `Line ${n}`,
    bus: (n) => `Bus ${n}`,
    lines: (list) => `line ${list}`,
    span: (a, b) => `${a} to ${b}`,
    loadingGenco: "Line loading, GENCO",
    loadingAc: "Line loading, AC solver",
    voltageGenco: "Bus voltage, GENCO (p.u.)",
    voltageAc: "Bus voltage, AC solver (p.u.)",
    flowError: "Line flow error, GENCO vs AC (MW)",
    voltageError: "Bus voltage error, GENCO vs AC (p.u.)",
    residualTitle: "GENCO power-balance residual (MW)",
    busPQ: "Load bus",
    busPV: "Generator bus",
    busREF: "Reference bus",
    overloaded: "Overloaded",
    uncertainKey: (lo, hi) => `Uncertain (${lo}–${hi})`,
    outOfService: "Out of service",
    kindLine: "Line",
    kindTransformer: "Transformer",
    branchSub: (kind, span, rating) => `${kind}, ${span}, ${rating}`,
    rated: (mva) => `rated ${mva} MVA`,
    noRating: "no rating",
    clickToRestore: "Out of service. Click to restore.",
    acCol: "AC",
    activeFlow: "Active flow (MW)",
    reactiveFlow: "Reactive flow (MVAr)",
    loadingPct: "Loading (%)",
    dcApprox: (mw) => `DC approximation: ${mw} MW`,
    verdictLabel: "GENCO verdict",
    verdict_safe: "safe",
    verdict_uncertain: "uncertain, too close to the limit to call",
    verdict_overload: "overloaded",
    acSays: "AC solver",
    acOver: "overloaded",
    acOk: "within limit",
    confidence: "Confidence",
    conf_high: "high",
    conf_medium: "medium",
    conf_low: "low",
    confDetail: (mw) => `self-check residual ${mw} MW`,
    confLow: "Re-check this state with the AC solver.",
    callsOverload: (n) => `${n} line${n === 1 ? "" : "s"} overloaded`,
    callsUncertain: (n) => `${n} line${n === 1 ? "" : "s"} uncertain`,
    callsNone: "No overloads predicted",
    confirmed: (n) => `${n} overload call${n === 1 ? "" : "s"} confirmed`,
    uncertainResolved: (n, k) => `${k} of ${n} uncertain line${n === 1 ? "" : "s"} overloaded`,
    bridge: "Cannot be tripped: it is the only connection for part of the grid.",
    bridgeNotice: (line) => `${line} cannot be tripped: it is the only connection for part of the grid.`,
    busSub: (shape, kv) => `${shape}, ${kv} kV`,
    busLoad: (p, q) => `, load ${p} MW / ${q} MVAr`,
    voltage: "Voltage (p.u.)",
    angle: "Angle (°)",
    genP: "Generation (MW)",
    genQ: "Generation (MVAr)",
    residualLine: (p, q) => `GENCO residual: ${p} MW, ${q} MVAr`,
    missed: (n) => `${n} overload${n === 1 ? "" : "s"} GENCO called safe`,
    falseAlarms: (n) => `${n} false alarm${n === 1 ? "" : "s"} from GENCO`,
    noOverloads: "No overloads per AC solver",
    maxVoltageErr: "Largest voltage error",
    maxAngleErr: "Largest angle error",
    maxFlowErr: "Largest line flow error",
    timeAc: "AC solver",
    timeGenco: "GENCO",
    timeDc: "DC approximation",
    timeOpf: "Re-dispatch (OPF)",
    cached: "cached",
    restore: (line) => `Restore ${line}`,
    unreachable: "Cannot reach the solver. Is server.py running?",
    retry: "Retry",
    islanded: "Tripping these lines splits the grid into separate islands.",
    opf_failed: "No feasible generation dispatch at this load level.",
    ac_failed: "The classical AC power flow did not converge for this state.",
    // staged check
    viewNewton: "After one Newton step",
    loadingNewton: "Line loading, after one Newton step",
    voltageNewton: "Bus voltage, after one Newton step (p.u.)",
    alphaLabel: "Allowed miss rate",
    alphaTip: "Width of the conformal band at stage 1: the share of real overloads the band is allowed to leave unflagged, on average over outage sets like the calibration ones. Smaller means a wider band.",
    staged: "Staged check of this state",
    stageName1: "GENCO",
    stageName2: "Newton step",
    stageName3: "Full AC",
    stageDecided: "decided here",
    stagePassed: "passed on",
    stageIdle: "not needed",
    stageUnused: "not used",
    head_genco: "Secure: accepted at stage 1",
    head_newton: "Secure: accepted after the Newton step",
    head_ac_overloaded: (n) => `Sent to full AC: ${n} line${n === 1 ? "" : "s"} overloaded`,
    head_ac_secure: "Sent to full AC: no overload",
    head_anchor_secure: "Base case, solved with full AC: no overload",
    head_anchor_overloaded: (n) => `Base case, solved with full AC: ${n} line${n === 1 ? "" : "s"} overloaded`,
    why_band_clear: "No line is inside the conformal band, so GENCO's prediction is accepted.",
    why_newton_clear: (n, pctText, line) => `${n} line${n === 1 ? " is" : "s are"} inside the conformal band, so the state is not accepted on GENCO alone. After the Newton step the highest loading is ${pctText} (${line}) and the self-check passes.`,
    why_newton_overload: (line, pctText, more) => `${line} at ${pctText} after the Newton step${more ? `, ${more} more over the limit` : ""}.`,
    why_selfcheck: (mw, limit) => `Self-check failed: ${mw} MW of power mismatch left after the Newton step (limit ${limit} MW).`,
    why_anchor: "No outage: this is the base case of this load level. Its AC solution is the anchor that stage 1 adds GENCO's predicted change to.",
    row1: "Stage 1",
    row2: "Stage 2",
    row3: "Stage 3",
    s1None: "no line inside the band",
    s1Flagged: (n) => `${n} line${n === 1 ? "" : "s"} possibly overloaded`,
    s2Worst: (pctText, line) => `highest loading ${pctText}, ${line}`,
    s2Over: (n) => `${n} line${n === 1 ? "" : "s"} over the limit`,
    s2Check: (mw, limit) => `self-check ${mw} MW, limit ${limit}`,
    s2CheckFail: (mw, limit) => `self-check failed: ${mw} MW, limit ${limit}`,
    forReference: "not needed, shown for reference",
    acNeeded: "AC needed",
    acNotNeeded: "AC not needed, shown for reference",
    acFinds: (n) => `${n} line${n === 1 ? "" : "s"} overloaded`,
    acFindsNone: "no overload",
    checkLabel: "Check against AC",
    check_correct: "correct, the staged verdict agrees with AC",
    check_missed: (n) => `missed overload: accepted as secure, AC finds ${n} overloaded line${n === 1 ? "" : "s"}`,
    check_false_alarm: "false alarm: sent to AC, which finds no overload",
    check_anchor: "base case, AC is the verdict",
    session: "This session",
    sessionReset: "Reset counter",
    sessStates: "States examined",
    sessStage1: "Accepted at stage 1",
    sessStage2: "Accepted after the Newton step",
    sessNoAc: "Decided without full AC",
    sessAc: "Needed full AC",
    sessAcDetail: (n) => `${n} base case${n === 1 ? "" : "s"}`,
    sessIter: "Newton iterations saved",
    sessIterValue: (saved, flat) => `${saved} of ${flat}`,
    sessIterTip: "Counted with one plain Newton solver (tolerance 1e-8 p.u.). Reference: every state solved from a flat start. Staged: 0 iterations for a state accepted at stage 1, 1 for a state accepted after the Newton step, and for a state sent to full AC the iterations Newton needs from GENCO's voltages (the stage-2 step included). A base case counts as a flat-start solve. GENCO's own inference is not counted and is not free.",
    sessCheck: "Against AC",
    sessCheckTip: "States with an outage only. Missed: accepted as secure although AC finds an overload. False: a false alarm, sent to full AC although AC finds no overload.",
    sessCheckValue: (ok, missed, alarms) => `${ok} correct · ${missed} missed · ${alarms} false`,
    possibleKey: "Possibly overloaded (inside the band)",
    overNewtonKey: "Over the limit after the step",
    clearedKey: "Inside the band, cleared by the step",
    newtonPct: "After one Newton step",
    tipStage1: (pred, limit) => `Stage 1: anchored prediction ${pred}, flagged from ${limit}`,
    tip_possible: "possibly overloaded",
    tip_safe: "safe",
  },
  zh: {
    brand: "GENCO 假设分析",
    mapShows: "地图显示",
    viewGenco: "GENCO",
    viewAc: "交流潮流",
    viewDiff: "差值",
    viewResidual: "GENCO 残差",
    load: "负荷",
    solving: "求解中…",
    gridMap: "电网图",
    hint: "点击线路使其断开，再次点击恢复。滚轮缩放，拖动平移。",
    scenario: "场景",
    linesOut: "断开线路",
    reset: "重置场景",
    advisor: "GENCO 助手",
    agreement: "交流潮流核对结果",
    ranked: "分歧最大的线路",
    solveTime: "求解耗时",
    line: (n) => `线路 ${n}`,
    bus: (n) => `节点 ${n}`,
    lines: (list) => `线路 ${list}`,
    span: (a, b) => `${a} 至 ${b}`,
    loadingGenco: "线路负载率 · GENCO",
    loadingAc: "线路负载率 · 交流潮流",
    voltageGenco: "节点电压 · GENCO (p.u.)",
    voltageAc: "节点电压 · 交流潮流 (p.u.)",
    flowError: "线路潮流误差 · GENCO 对比交流潮流 (MW)",
    voltageError: "节点电压误差 · GENCO 对比交流潮流 (p.u.)",
    residualTitle: "GENCO 功率平衡残差 (MW)",
    busPQ: "负荷节点",
    busPV: "发电机节点",
    busREF: "平衡节点",
    overloaded: "过载",
    uncertainKey: (lo, hi) => `不确定 (${lo}–${hi})`,
    outOfService: "已断开",
    kindLine: "线路",
    kindTransformer: "变压器",
    branchSub: (kind, span, rating) => `${kind}，${span}，${rating}`,
    rated: (mva) => `额定 ${mva} MVA`,
    noRating: "无额定容量",
    clickToRestore: "已断开。点击恢复。",
    acCol: "交流",
    activeFlow: "有功潮流 (MW)",
    reactiveFlow: "无功潮流 (MVAr)",
    loadingPct: "负载率 (%)",
    dcApprox: (mw) => `直流近似：${mw} MW`,
    verdictLabel: "GENCO 判断",
    verdict_safe: "安全",
    verdict_uncertain: "不确定，离限值太近无法判断",
    verdict_overload: "过载",
    acSays: "交流潮流",
    acOver: "过载",
    acOk: "未超限",
    confidence: "置信度",
    conf_high: "高",
    conf_medium: "中",
    conf_low: "低",
    confDetail: (mw) => `自检残差 ${mw} MW`,
    confLow: "建议用交流潮流复核该状态。",
    callsOverload: (n) => `${n} 条线路过载`,
    callsUncertain: (n) => `${n} 条线路不确定`,
    callsNone: "未预测到过载",
    confirmed: (n) => `${n} 条过载判断得到确认`,
    uncertainResolved: (n, k) => `${n} 条不确定线路中有 ${k} 条实际过载`,
    bridge: "无法断开：它是部分电网的唯一连接。",
    bridgeNotice: (line) => `${line} 无法断开：它是部分电网的唯一连接。`,
    busSub: (shape, kv) => `${shape}，${kv} kV`,
    busLoad: (p, q) => `，负荷 ${p} MW / ${q} MVAr`,
    voltage: "电压 (p.u.)",
    angle: "相角 (°)",
    genP: "发电有功 (MW)",
    genQ: "发电无功 (MVAr)",
    residualLine: (p, q) => `GENCO 残差：${p} MW，${q} MVAr`,
    missed: (n) => `${n} 处过载被 GENCO 判为安全`,
    falseAlarms: (n) => `GENCO 误报 ${n} 处过载`,
    noOverloads: "交流潮流未发现过载",
    maxVoltageErr: "最大电压误差",
    maxAngleErr: "最大相角误差",
    maxFlowErr: "最大线路潮流误差",
    timeAc: "交流潮流求解器",
    timeGenco: "GENCO",
    timeDc: "直流近似",
    timeOpf: "重新调度 (OPF)",
    cached: "已缓存",
    restore: (line) => `恢复 ${line}`,
    unreachable: "无法连接求解器。server.py 在运行吗？",
    retry: "重试",
    islanded: "断开这些线路会使电网分裂成孤岛。",
    opf_failed: "在该负荷水平下找不到可行的发电调度。",
    ac_failed: "该状态下传统交流潮流不收敛。",
    // 分级校核
    viewNewton: "一步牛顿之后",
    loadingNewton: "线路负载率 · 一步牛顿之后",
    voltageNewton: "节点电压 · 一步牛顿之后 (p.u.)",
    alphaLabel: "允许漏报率",
    alphaTip: "第 1 级保形带的宽度：在与校准集同类的断线组合上，平均允许有多大比例的真实过载未被标记。数值越小，带越宽。",
    staged: "该状态的分级校核",
    stageName1: "GENCO",
    stageName2: "一步牛顿",
    stageName3: "完整交流",
    stageDecided: "在此判定",
    stagePassed: "转下一级",
    stageIdle: "无需执行",
    stageUnused: "未使用",
    head_genco: "安全：第 1 级即接受",
    head_newton: "安全：一步牛顿后接受",
    head_ac_overloaded: (n) => `转交完整交流潮流：${n} 条线路过载`,
    head_ac_secure: "转交完整交流潮流：无过载",
    head_anchor_secure: "基态，由完整交流潮流求解：无过载",
    head_anchor_overloaded: (n) => `基态，由完整交流潮流求解：${n} 条线路过载`,
    why_band_clear: "没有线路落在保形带内，因此接受 GENCO 的预测。",
    why_newton_clear: (n, pctText, line) => `${n} 条线路落在保形带内，不能仅凭 GENCO 接受该状态。一步牛顿之后最高负载率为 ${pctText}（${line}），且自检通过。`,
    why_newton_overload: (line, pctText, more) => `一步牛顿之后${line}负载率为 ${pctText}${more ? `，另有 ${more} 条线路超限` : ""}。`,
    why_selfcheck: (mw, limit) => `自检未通过：一步牛顿之后仍有 ${mw} MW 的功率不平衡量（限值 ${limit} MW）。`,
    why_anchor: "没有断线：这是该负荷水平的基态。它的交流潮流解是锚点，第 1 级在其上叠加 GENCO 预测的变化量。",
    row1: "第 1 级",
    row2: "第 2 级",
    row3: "第 3 级",
    s1None: "没有线路落在带内",
    s1Flagged: (n) => `${n} 条线路可能过载`,
    s2Worst: (pctText, line) => `最高负载率 ${pctText}，${line}`,
    s2Over: (n) => `${n} 条线路超限`,
    s2Check: (mw, limit) => `自检 ${mw} MW，限值 ${limit}`,
    s2CheckFail: (mw, limit) => `自检未通过：${mw} MW，限值 ${limit}`,
    forReference: "无需执行，仅供参考",
    acNeeded: "需要交流潮流",
    acNotNeeded: "无需交流潮流，仅供参考",
    acFinds: (n) => `${n} 条线路过载`,
    acFindsNone: "无过载",
    checkLabel: "与交流潮流核对",
    check_correct: "正确，分级结论与交流潮流一致",
    check_missed: (n) => `漏报：已判为安全，交流潮流发现 ${n} 条线路过载`,
    check_false_alarm: "误报：已转交交流潮流，但其未发现过载",
    check_anchor: "基态，以交流潮流为准",
    session: "本次会话",
    sessionReset: "重置计数",
    sessStates: "已检查状态数",
    sessStage1: "第 1 级接受",
    sessStage2: "一步牛顿后接受",
    sessNoAc: "无需完整交流潮流",
    sessAc: "需要完整交流潮流",
    sessAcDetail: (n) => `其中基态 ${n} 个`,
    sessIter: "节省的牛顿迭代次数",
    sessIterValue: (saved, flat) => `${saved} / ${flat}`,
    sessIterTip: "均用同一个普通牛顿法求解器计数（收敛精度 1e-8 p.u.）。基准：每个状态都从平启动求解。分级流程：第 1 级接受的状态计 0 次，一步牛顿后接受的状态计 1 次，转交完整交流潮流的状态计从 GENCO 电压出发收敛所需的次数（含第 2 级那一步）。基态按一次平启动求解计。GENCO 自身的推理未计入，而它并非没有开销。",
    sessCheck: "与交流潮流对比",
    sessCheckTip: "仅统计有断线的状态。漏报：已判为安全，但交流潮流发现过载。误报：已转交完整交流潮流，但其未发现过载。",
    sessCheckValue: (ok, missed, alarms) => `${ok} 正确 · ${missed} 漏报 · ${alarms} 误报`,
    possibleKey: "可能过载（带内）",
    overNewtonKey: "一步牛顿之后超限",
    clearedKey: "带内，经一步牛顿排除",
    newtonPct: "一步牛顿之后",
    tipStage1: (pred, limit) => `第 1 级：锚定预测 ${pred}，自 ${limit} 起标记`,
    tip_possible: "可能过载",
    tip_safe: "安全",
  },
};

function storedLang() {
  try {
    return localStorage.getItem("lang");
  } catch {
    return null;
  }
}

const QUERY = new URLSearchParams(location.search);
let lang = QUERY.get("lang") || storedLang() || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en");
if (!TEXT[lang]) lang = "en";

function t(key, ...args) {
  const entry = TEXT[lang][key];
  return typeof entry === "function" ? entry(...args) : entry;
}

const pct = (v) => `${Math.round(v * 100)}%`;
const absDiff = (a, b) => (a == null || b == null ? null : Math.abs(a - b));

const VIEWS = {
  genco: {
    branch: { value: (b) => b.loading_genco, domain: [0, 1], title: "loadingGenco", ticks: ["0%", "50%", "100%"] },
    bus: { value: (n) => n.vm_genco, domain: [0.94, 1.06], title: "voltageGenco", ticks: ["0.94", "1.00", "1.06"] },
    // With the staged check the map marks the lines inside the conformal band; without it, the fixed band.
    status: (row, sol) => (sol.staged
      ? (row.stage1 === "possible" ? "uncertain" : "")
      : row.verdict === "overload" ? "over" : row.verdict === "uncertain" ? "uncertain" : ""),
    keys: (sol) => (sol && sol.staged ? ["possible"] : ["over", "uncertain"]),
  },
  newton: {
    branch: { value: (b) => b.loading_newton, domain: [0, 1], title: "loadingNewton", ticks: ["0%", "50%", "100%"] },
    bus: { value: (n) => n.vm_newton, domain: [0.94, 1.06], title: "voltageNewton", ticks: ["0.94", "1.00", "1.06"] },
    status: (row) => (row.stage2 === "over" ? "over" : row.stage1 === "possible" ? "cleared" : ""),
    keys: () => ["overNewton", "cleared"],
  },
  ac: {
    branch: { value: (b) => b.loading_ac, domain: [0, 1], title: "loadingAc", ticks: ["0%", "50%", "100%"] },
    bus: { value: (n) => n.vm_ac, domain: [0.94, 1.06], title: "voltageAc", ticks: ["0.94", "1.00", "1.06"] },
    status: (row, sol) => (sol.summary.overloads_ac.includes(row.id) ? "over" : ""),
    keys: () => ["over"],
  },
  diff: {
    branch: { value: (b) => absDiff(b.p_genco, b.p_ac), domain: [0, 10], title: "flowError", ticks: ["0", "5", "10+"] },
    bus: { value: (n) => absDiff(n.vm_genco, n.vm_ac), domain: [0, 0.01], title: "voltageError", ticks: ["0", "0.005", "0.01+"] },
  },
  residual: {
    bus: { value: (n) => Math.abs(n.res_p), domain: [0, 1], title: "residualTitle", ticks: ["0", "0.5", "1+"] },
  },
};

const BUS_SHAPES = {
  PQ: { symbol: d3.symbolCircle, size: 70, label: "busPQ" },
  PV: { symbol: d3.symbolSquare, size: 90, label: "busPV" },
  REF: { symbol: d3.symbolDiamond, size: 200, label: "busREF" },
};

const state = {
  grid: null,
  sol: null,
  scenario: null, // last scenario the solver accepted
  view: "genco",
  alpha: null,
  session: new Map(), // state key -> staged outcome, so a state looked at twice counts once
  selected: null,
  transform: d3.zoomIdentity,
};

const $ = (id) => document.getElementById(id);
const svg = d3.select("#map").attr("viewBox", `0 0 ${W} ${H}`).attr("preserveAspectRatio", "xMidYMid meet");
const layers = {};
for (const name of ["halo", "lines", "hits", "buses", "labels"]) layers[name] = svg.append("g");

const dark = window.matchMedia("(prefers-color-scheme: dark)");
const ramps = () => RAMPS[dark.matches ? "dark" : "light"];

function colorScale(domain, ramp) {
  const [lo, hi] = domain;
  const stops = ramp.map((_, i) => lo + ((hi - lo) * i) / (ramp.length - 1));
  return d3.scaleLinear().domain(stops).range(ramp).clamp(true);
}

// ---------- formatting ----------

function fmt(v, digits = 1) {
  if (v == null) return "–";
  return v.toFixed(digits).replace("-", "−");
}
const lineName = (id) => t("line", id + 1);
const busName = (id) => t("bus", id + 1);
const busSpan = (branch) => t("span", busName(branch.from), busName(branch.to));
const lineList = (ids) => t("lines", ids.map((id) => id + 1).join(", "));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- geometry ----------

function basePoint(bus) {
  return [PAD + bus.x * (W - 2 * PAD), PAD + bus.y * (H - 2 * PAD)];
}

// Parallel circuits share both end buses; spread them sideways so each is clickable.
function assignOffsets(branches) {
  const groups = d3.group(branches, (b) => `${Math.min(b.from, b.to)}-${Math.max(b.from, b.to)}`);
  for (const group of groups.values()) {
    group.forEach((b, i) => {
      b.offset = (i - (group.length - 1) / 2) * PARALLEL_GAP;
    });
  }
}

function segment(branch) {
  const tf = state.transform;
  // Take the perpendicular from the lower-id bus so parallel circuits offset to opposite sides.
  const flip = branch.from > branch.to;
  const a = tf.apply(basePoint(state.grid.buses[flip ? branch.to : branch.from]));
  const c = tf.apply(basePoint(state.grid.buses[flip ? branch.from : branch.to]));
  const dx = c[0] - a[0], dy = c[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const ox = (-dy / len) * branch.offset, oy = (dx / len) * branch.offset;
  return { x1: a[0] + ox, y1: a[1] + oy, x2: c[0] + ox, y2: c[1] + oy };
}

// ---------- drawing ----------

function drawGrid() {
  const { buses, branches } = state.grid;
  assignOffsets(branches);

  layers.halo.selectAll("line").data(branches).join("line").attr("class", "branch-halo");
  layers.lines.selectAll("line").data(branches).join("line").attr("class", "branch");
  layers.hits.selectAll("line").data(branches).join("line")
    .attr("class", (b) => `branch-hit${b.bridge ? " bridge" : ""}`)
    .on("click", (event, b) => toggleBranch(b))
    .on("mousemove", (event, b) => showTooltip(event, branchTooltip(b)))
    .on("mouseleave", hideTooltip);

  layers.buses.selectAll("path").data(buses).join("path")
    .attr("class", (n) => `bus${n.type === "REF" ? " ref" : ""}`)
    .attr("d", (n) => d3.symbol(BUS_SHAPES[n.type].symbol, BUS_SHAPES[n.type].size)())
    .on("mousemove", (event, n) => showTooltip(event, busTooltip(n)))
    .on("mouseleave", hideTooltip);

  layers.labels.selectAll("text").data(buses).join("text")
    .attr("class", "bus-label")
    .attr("dx", 9).attr("dy", 3)
    .text((n) => n.id + 1);

  svg.call(
    d3.zoom().scaleExtent([1, 10]).translateExtent([[0, 0], [W, H]]).on("zoom", (event) => {
      state.transform = event.transform;
      position();
    }),
  ).on("dblclick.zoom", null);

  position();
}

function position() {
  const tf = state.transform;
  for (const layer of [layers.halo, layers.lines, layers.hits]) {
    layer.selectAll("line").each(function (b) {
      const s = segment(b);
      this.setAttribute("x1", s.x1);
      this.setAttribute("y1", s.y1);
      this.setAttribute("x2", s.x2);
      this.setAttribute("y2", s.y2);
    });
  }
  layers.buses.selectAll("path").attr("transform", (n) => `translate(${tf.apply(basePoint(n))})`);
  layers.labels.attr("display", tf.k >= LABEL_ZOOM ? null : "none");
  layers.labels.selectAll("text").attr("transform", (n) => `translate(${tf.apply(basePoint(n))})`);
}

function paint() {
  if (!state.sol) return;
  const view = VIEWS[state.view];
  const sol = state.sol;
  const branchColor = view.branch && colorScale(view.branch.domain, ramps().branch);
  const busColor = colorScale(view.bus.domain, ramps().bus);

  layers.lines.selectAll("line").each(function (b) {
    const row = sol.branches[b.id];
    const value = view.branch && row.in_service ? view.branch.value(row) : null;
    const status = row.in_service && view.status ? view.status(row, sol) : "";
    this.setAttribute("class", `branch${row.in_service ? "" : " out"}${status ? ` ${status}` : ""}${value == null ? " plain" : ""}`);
    this.style.stroke = value == null ? "" : branchColor(value);
  });
  layers.halo.selectAll("line").attr("display", (b) => (b.id === state.selected ? null : "none"));
  layers.buses.selectAll("path").style("fill", (n) => busColor(view.bus.value(sol.buses[n.id])));
}

// ---------- legend ----------

function rampBlock(spec, ramp) {
  return `<div>
    <div class="title">${esc(t(spec.title))}</div>
    <div class="ramp" style="background: linear-gradient(to right, ${ramp.join(", ")})"></div>
    <div class="ticks">${spec.ticks.map((tick) => `<span>${esc(tick)}</span>`).join("")}</div>
  </div>`;
}

function lineKey(cls, label) {
  return `<span class="key"><svg width="26" height="10"><line class="branch ${cls}" x1="2" y1="5" x2="24" y2="5"/></svg>${label}</span>`;
}

function shapeKey(type) {
  const shape = BUS_SHAPES[type];
  const path = d3.symbol(shape.symbol, shape.size * 0.8)();
  return `<span class="key"><svg width="16" height="16"><path transform="translate(8 8)" d="${path}" fill="var(--muted)"/></svg>${t(shape.label)}</span>`;
}

function drawLegend() {
  const view = VIEWS[state.view];
  const parts = [];
  if (view.branch) parts.push(rampBlock(view.branch, ramps().branch));
  parts.push(rampBlock(view.bus, ramps().bus));
  const band = state.grid ? state.grid.advisor.band.map(pct) : ["", ""];
  const labels = {
    over: ["over", t("overloaded")],
    uncertain: ["uncertain", t("uncertainKey", ...band)],
    possible: ["uncertain", t("possibleKey")],
    overNewton: ["over", t("overNewtonKey")],
    cleared: ["cleared", t("clearedKey")],
  };
  const lines = (view.keys ? view.keys(state.sol) : []).map((key) => lineKey(...labels[key]));
  lines.push(lineKey("out", t("outOfService")));
  parts.push(`<div class="keys">${lines.join("")}</div>`);
  parts.push(`<div class="keys">${["PQ", "PV", "REF"].map(shapeKey).join("")}</div>`);
  $("legend").innerHTML = parts.join("");
}

// ---------- tooltips ----------

function tableRow(label, genco, ac, digits) {
  const delta = genco == null || ac == null ? null : genco - ac;
  return `<tr><td>${label}</td><td>${fmt(genco, digits)}</td><td>${fmt(ac, digits)}</td><td>${fmt(delta, digits)}</td></tr>`;
}
const tableHead = () => `<tr><th></th><th>GENCO</th><th>${t("acCol")}</th><th>Δ</th></tr>`;

function branchTooltip(b) {
  const row = state.sol ? state.sol.branches[b.id] : null;
  const kind = t(b.transformer ? "kindTransformer" : "kindLine");
  const rating = b.rate_a == null ? t("noRating") : t("rated", fmt(b.rate_a, 0));
  let html = `<div class="head">${lineName(b.id)}</div>
    <div class="sub">${t("branchSub", kind, busSpan(b), rating)}</div>`;
  if (!row) return html;
  if (!row.in_service) return `${html}<div class="flag">${t("clickToRestore")}</div>`;
  const pc = (v) => (v == null ? null : v * 100);
  html += `<table>${tableHead()}
    ${tableRow(t("activeFlow"), row.p_genco, row.p_ac, 1)}
    ${tableRow(t("reactiveFlow"), row.q_genco, row.q_ac, 1)}
    ${tableRow(t("loadingPct"), pc(row.loading_genco), pc(row.loading_ac), 1)}
  </table>
  <div class="sub" style="margin: 6px 0 0">${t("dcApprox", fmt(row.p_dc, 1))}</div>`;
  if (state.sol.staged && row.stage1) {
    html += `<div class="flag">${t("newtonPct")}: ${row.loading_newton == null ? "–" : `${fmt(row.loading_newton * 100, 1)}%`}<br>
      ${t("tipStage1", `${fmt(row.loading_anchored * 100, 1)}%`, `${fmt(row.band_limit * 100, 1)}%`)}: ${t(`tip_${row.stage1}`)}<br>
      ${t("acSays")}: ${t(state.sol.summary.overloads_ac.includes(b.id) ? "acOver" : "acOk")}</div>`;
  } else if (row.verdict) {
    const s = state.sol.summary;
    const acOver = s.overloads_ac.includes(b.id);
    const icon = s.advisor.missed.includes(b.id) ? "⚠ " : s.advisor.false_alarms.includes(b.id) ? "▲ " : "";
    html += `<div class="flag">${t("verdictLabel")}: ${t(`verdict_${row.verdict}`)}<br>${icon}${t("acSays")}: ${t(acOver ? "acOver" : "acOk")}</div>`;
  }
  if (b.bridge) html += `<div class="flag">${t("bridge")}</div>`;
  return html;
}

function busTooltip(n) {
  const row = state.sol ? state.sol.buses[n.id] : null;
  let html = `<div class="head">${busName(n.id)}</div><div class="sub">${t("busSub", t(BUS_SHAPES[n.type].label), fmt(n.vn_kv, 0))}`;
  if (!row) return `${html}</div>`;
  html += `${t("busLoad", fmt(row.pd, 1), fmt(row.qd, 1))}</div>
  <table>${tableHead()}
    ${tableRow(t("voltage"), row.vm_genco, row.vm_ac, 4)}
    ${tableRow(t("angle"), row.va_genco, row.va_ac, 2)}
    ${n.type === "PQ" ? "" : tableRow(t("genP"), row.pg_genco, row.pg_ac, 1)}
    ${n.type === "PQ" ? "" : tableRow(t("genQ"), row.qg_genco, row.qg_ac, 1)}
  </table>
  <div class="sub" style="margin: 6px 0 0">${t("residualLine", fmt(row.res_p, 2), fmt(row.res_q, 2))}</div>`;
  return html;
}

function showTooltip(event, html) {
  const tip = $("tooltip");
  const box = tip.parentElement.getBoundingClientRect();
  tip.innerHTML = html;
  tip.hidden = false;
  const x = event.clientX - box.left, y = event.clientY - box.top;
  const left = x + 16 + tip.offsetWidth > box.width ? x - 16 - tip.offsetWidth : x + 16;
  const top = Math.min(Math.max(8, y + 16), box.height - tip.offsetHeight - 8);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${top}px`;
}

function hideTooltip() {
  $("tooltip").hidden = true;
}

// ---------- side panel ----------

function statRows(rows) {
  return rows.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
}

function verdictRow(cls, icon, text, detail) {
  return `<div class="row ${cls}"><span class="icon" aria-hidden="true">${icon}</span><span>${text}${detail ? ` <span class="detail">${detail}</span>` : ""}</span></div>`;
}

function drawPanel() {
  const { sol, grid, scenario } = state;
  if (!sol) return;
  const s = sol.summary;

  $("s-load").textContent = `${pct(scenario.load_scale)} · ${d3.format(",.0f")(s.total_load_mw)} MW`;
  $("s-out").textContent = scenario.outages.length;
  $("outages").innerHTML = scenario.outages
    .map((id) => `<li>${lineName(id)}<button type="button" data-restore="${id}" aria-label="${esc(t("restore", lineName(id)))}">×</button></li>`)
    .join("");
  $("reset").disabled = scenario.outages.length === 0 && scenario.load_scale === grid.load_scale.default;

  const staged = sol.staged;
  $("staged-section").hidden = !staged;
  $("session-section").hidden = !staged;
  $("advisor-section").hidden = !!staged;
  if (staged) {
    drawStaged(sol);
    drawSession();
  }

  // What GENCO says on its own: a confidence level, and each line in one of three tiers.
  const a = s.advisor;
  const conf = { high: ["good", "✓"], medium: ["warn", "▲"], low: ["crit", "⚠"] }[a.confidence];
  const says = [
    verdictRow(conf[0], conf[1], `${t("confidence")}: ${t(`conf_${a.confidence}`)}`,
      t("confDetail", fmt(s.max_res_p_mw, 2)) + (a.confidence === "low" ? ` · ${t("confLow")}` : "")),
  ];
  if (a.overload.length) says.push(verdictRow("crit", "⚠", t("callsOverload", a.overload.length), lineList(a.overload)));
  if (a.uncertain.length) says.push(verdictRow("unsure", "?", t("callsUncertain", a.uncertain.length), lineList(a.uncertain)));
  if (!a.overload.length && !a.uncertain.length) says.push(verdictRow("good", "✓", t("callsNone")));
  $("advisor").innerHTML = says.join("");

  // How those calls hold up against the AC solver.
  const rows = [];
  if (staged) {
    const anchor = staged.decided_by === "anchor";
    const cls = { correct: ["good", "✓"], missed: ["crit", "⚠"], false_alarm: ["warn", "▲"] }[staged.check];
    rows.push(verdictRow(cls[0], cls[1], `${t("checkLabel")}:`,
      anchor ? t("check_anchor") : t(`check_${staged.check}`, s.overloads_ac.length)));
    if (s.overloads_ac.length) rows.push(verdictRow("crit", "⚠", t("acFinds", s.overloads_ac.length), lineList(s.overloads_ac)));
  } else if (a.missed.length) rows.push(verdictRow("crit", "⚠", t("missed", a.missed.length), lineList(a.missed)));
  if (!staged && a.false_alarms.length) rows.push(verdictRow("warn", "▲", t("falseAlarms", a.false_alarms.length), lineList(a.false_alarms)));
  if (!staged && a.confirmed.length) rows.push(verdictRow("good", "✓", t("confirmed", a.confirmed.length), lineList(a.confirmed)));
  if (!staged && a.uncertain.length) rows.push(verdictRow("unsure", "?", t("uncertainResolved", a.uncertain.length, a.uncertain_overloaded.length), a.uncertain_overloaded.length ? lineList(a.uncertain_overloaded) : ""));
  if (!s.overloads_ac.length) rows.push(verdictRow("good", "✓", t("noOverloads")));
  $("verdict").innerHTML = rows.join("");

  $("errors").innerHTML = statRows([
    [t("maxVoltageErr"), `${fmt(s.max_vm_err, 4)} p.u.`],
    [t("maxAngleErr"), `${fmt(s.max_va_err_deg, 2)}°`],
    [t("maxFlowErr"), `${fmt(s.max_p_flow_err_mw, 1)} MW`],
  ]);

  const ranked = sol.branches
    .filter((b) => b.in_service)
    .map((b) => ({ ...b, err: Math.abs(b.p_genco - b.p_ac) }))
    .sort((a, b) => b.err - a.err)
    .slice(0, 8);
  $("ranked").innerHTML = ranked
    .map((b) => `<li tabindex="0" role="button" data-branch="${b.id}" aria-pressed="${b.id === state.selected}">
        <span>${lineName(b.id)} <span class="muted">${busSpan(grid.branches[b.id])}</span></span>
        <span class="delta">${fmt(b.err, 1)} MW</span>
        <span class="sub">GENCO ${fmt(b.p_genco, 1)} · ${t("acCol")} ${fmt(b.p_ac, 1)} MW</span>
      </li>`)
    .join("");

  const ms = sol.timing_ms;
  $("timing").innerHTML = statRows([
    [t("timeAc"), `${fmt(ms.ac, 1)} ms`],
    [t("timeGenco"), `${fmt(ms.genco, 1)} ms`],
    [t("timeDc"), `${fmt(ms.dc, 1)} ms`],
    [t("timeOpf"), ms.opf == null ? t("cached") : `${fmt(ms.opf, 0)} ms`],
  ]);
}

// ---------- staged check ----------

const pct1 = (v) => (v == null ? "–" : `${fmt(v * 100, 1)}%`);
const mw = (v) => (v == null ? "–" : fmt(v, v < 0.1 ? 3 : 2));

function drawStaged(sol) {
  const g = sol.staged, n = g.newton;
  const anchor = g.decided_by === "anchor";
  const nOver = sol.summary.overloads_ac.length;

  // Which stage decided: earlier stages passed the state on, later ones were not needed.
  $("stages").innerHTML = [1, 2, 3].map((i) => {
    const kind = i === g.stage ? `decided ${g.verdict}` : i < g.stage && !anchor ? "passed" : "idle";
    const label = i === g.stage ? "stageDecided" : anchor ? "stageUnused" : i < g.stage ? "stagePassed" : "stageIdle";
    return `<li class="${kind}"><span class="num">${i}</span><span class="name">${t(`stageName${i}`)}</span><span class="state">${t(label)}</span></li>`;
  }).join("");

  const head = $("stage-head");
  head.className = `stage-head ${g.verdict}`;
  head.textContent = g.stage < 3 ? t(`head_${g.decided_by}`) : t(`head_${g.decided_by}_${g.verdict}`, nOver);

  const worstLine = n.worst_line == null ? "–" : lineName(n.worst_line);
  const why = {
    band_clear: () => t("why_band_clear"),
    newton_clear: () => t("why_newton_clear", g.flagged.length, pct1(n.worst_loading), worstLine),
    newton_overload: () => t("why_newton_overload", worstLine, pct1(n.worst_loading), n.overloads.length - 1),
    selfcheck: () => t("why_selfcheck", mw(n.mismatch_mw), n.threshold_mw),
    anchor: () => t("why_anchor"),
  }[g.reason];
  $("stage-why").textContent = why();

  // One line per stage: what it found, whether or not the flow needed it.
  const reference = (used) => (used ? "" : ` · ${t("forReference")}`);
  const rows = [];
  rows.push(g.flagged.length
    ? verdictRow("unsure", "?", `${t("row1")}: ${t("s1Flagged", g.flagged.length)}`, lineList(g.flagged))
    : verdictRow("good", "✓", `${t("row1")}: ${t("s1None")}`));
  const stepUsed = !anchor && g.stage >= 2;
  const stepBad = n.overloads.length > 0 || !n.selfcheck_ok;
  rows.push(verdictRow(stepBad ? "crit" : "good", stepBad ? "⚠" : "✓",
    `${t("row2")}: ${n.overloads.length ? t("s2Over", n.overloads.length) : t("s2Worst", pct1(n.worst_loading), worstLine)}`,
    `${n.overloads.length ? `${lineList(n.overloads)} · ` : ""}${t(n.selfcheck_ok ? "s2Check" : "s2CheckFail", mw(n.mismatch_mw), n.threshold_mw)}${reference(stepUsed)}`));
  rows.push(verdictRow(nOver ? "crit" : "good", nOver ? "⚠" : "✓",
    `${t("row3")}: ${t(g.ac.needed ? "acNeeded" : "acNotNeeded")}`,
    nOver ? `${t("acFinds", nOver)}, ${lineList(sol.summary.overloads_ac)}` : t("acFindsNone")));
  $("stage-rows").innerHTML = rows.join("");
}

const stateKey = (sol) => `${sol.load_scale}|${sol.outages.join(",")}`;

function drawSession() {
  const seen = [...state.session.values()];
  const count = (test) => seen.filter(test).length;
  const anchors = count((g) => g.decided_by === "anchor");
  const ac = count((g) => g.stage === 3);
  const counted = seen.filter((g) => g.iterations.saved != null);
  const flat = d3.sum(counted, (g) => g.iterations.flat);
  const saved = d3.sum(counted, (g) => g.iterations.saved);
  const judged = seen.filter((g) => g.decided_by !== "anchor");
  const tally = (kind) => judged.filter((g) => g.check === kind).length;
  $("session").innerHTML = statRows([
    [t("sessStates"), seen.length],
    [t("sessStage1"), count((g) => g.stage === 1)],
    [t("sessStage2"), count((g) => g.stage === 2)],
    [t("sessNoAc"), seen.length - ac],
    [t("sessAc"), `${ac}${anchors ? ` <span class="muted">(${t("sessAcDetail", anchors)})</span>` : ""}`],
    [t("sessIter"), t("sessIterValue", saved, flat)],
    [t("sessCheck"), t("sessCheckValue", tally("correct"), tally("missed"), tally("false_alarm"))],
  ]);
  const dt = $("session").querySelectorAll("dt")[5];
  dt.title = t("sessIterTip");
  dt.tabIndex = 0;
  $("session").querySelectorAll("dt")[6].title = t("sessCheckTip");
}

function drawAlpha() {
  const staged = state.grid && state.grid.staged;
  $("alpha-box").hidden = !staged;
  document.querySelector('#views [data-view="newton"]').hidden = !staged;
  if (!staged) return;
  $("alpha").innerHTML = staged.alphas
    .map((a) => `<button type="button" role="radio" data-alpha="${a}" aria-checked="${a === state.alpha}">${Math.round(a * 100)}%</button>`)
    .join("");
}

// The state on screen as a link: ?load=1.1&out=32,104&alpha=0.01&view=newton&lang=en (line numbers as displayed).
function syncUrl() {
  const { scenario, grid } = state;
  const query = new URLSearchParams();
  if (scenario.load_scale !== grid.load_scale.default) query.set("load", scenario.load_scale);
  if (scenario.outages.length) query.set("out", scenario.outages.map((id) => id + 1).join(","));
  if (grid.staged && state.alpha !== grid.staged.default_alpha) query.set("alpha", state.alpha);
  if (state.view !== "genco") query.set("view", state.view);
  if (QUERY.get("lang")) query.set("lang", lang);
  const text = query.toString().replace(/%2C/g, ",");
  try {
    history.replaceState(null, "", text ? `?${text}` : location.pathname);
  } catch {
    // Not every embedding allows it; the link just stays as it was.
  }
}

function scenarioFromUrl() {
  const { grid } = state;
  const scale = grid.load_scale;
  let load = Number(QUERY.get("load"));
  if (!QUERY.get("load") || !Number.isFinite(load)) load = scale.default;
  load = Math.min(scale.max, Math.max(scale.min, Math.round(load / scale.step) * scale.step));
  const outages = (QUERY.get("out") || "").split(",").map((v) => Number(v) - 1)
    .filter((id, i, all) => Number.isInteger(id) && id >= 0 && id < grid.branches.length && all.indexOf(id) === i);
  if (grid.staged) {
    let alpha = Number(QUERY.get("alpha"));
    if (alpha >= 1) alpha /= 100; // alpha=2 means 2%
    state.alpha = grid.staged.alphas.find((a) => Math.abs(a - alpha) < 1e-9) ?? grid.staged.default_alpha;
  }
  const view = QUERY.get("view");
  if (VIEWS[view] && (view !== "newton" || grid.staged)) state.view = view;
  return { load_scale: Number(load.toFixed(4)), outages };
}

function render() {
  paint();
  drawLegend();
  drawPanel();
}

// Fixed text in index.html is keyed by data-i18n / data-i18n-aria attributes.
function applyLanguage() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = t("brand");
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  document.querySelectorAll("[data-i18n-title]").forEach((el) => (el.title = t(el.dataset.i18nTitle)));
  document.querySelectorAll("#lang button").forEach((b) => b.setAttribute("aria-checked", b.dataset.lang === lang));
  hideTooltip();
  clearNotice();
  drawLegend();
  drawPanel();
}

// ---------- solving ----------

let requestId = 0;
let busyTimer = null;
let noticeTimer = null;

function setBusy(on) {
  clearTimeout(busyTimer);
  if (on) busyTimer = setTimeout(() => ($("busy").hidden = false), 150);
  else $("busy").hidden = true;
}

function notify(message, retry) {
  const box = $("notice");
  clearTimeout(noticeTimer);
  box.innerHTML = `<span>${esc(message)}</span>`;
  if (retry) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost";
    button.style.marginTop = "0";
    button.textContent = t("retry");
    button.onclick = retry;
    box.append(button);
  } else {
    noticeTimer = setTimeout(clearNotice, 7000);
  }
  box.hidden = false;
}

function clearNotice() {
  clearTimeout(noticeTimer);
  $("notice").hidden = true;
}

// Put the controls back to the last scenario the solver accepted.
function syncControls() {
  if (!state.scenario) return;
  $("load").value = state.scenario.load_scale;
  $("load-out").textContent = pct(state.scenario.load_scale);
}

async function solve(next) {
  const id = ++requestId;
  setBusy(true);
  let data;
  try {
    const res = await fetch("api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.alpha == null ? next : { ...next, alpha: state.alpha }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    if (id !== requestId) return;
    setBusy(false);
    syncControls();
    notify(t("unreachable"), () => solve(next));
    return;
  }
  if (id !== requestId) return; // a newer request superseded this one
  setBusy(false);
  if (!data.ok) {
    syncControls();
    notify(t(data.reason) || data.message);
    // A state from the link that cannot be solved: fall back to the base case.
    if (!state.sol && (next.outages.length || next.load_scale !== state.grid.load_scale.default)) {
      solve({ load_scale: state.grid.load_scale.default, outages: [] });
    }
    return;
  }
  clearNotice();
  state.sol = data;
  state.scenario = { load_scale: data.load_scale, outages: data.outages };
  if (data.staged) state.session.set(stateKey(data), data.staged);
  syncUrl();
  if (state.selected != null && !data.branches[state.selected].in_service) state.selected = null;
  syncControls();
  render();
}

function toggleBranch(b) {
  if (!state.scenario) return;
  const out = state.scenario.outages;
  if (!out.includes(b.id) && b.bridge) {
    notify(t("bridgeNotice", lineName(b.id)));
    return;
  }
  hideTooltip();
  solve({
    load_scale: state.scenario.load_scale,
    outages: out.includes(b.id) ? out.filter((id) => id !== b.id) : [...out, b.id],
  });
}

function select(id) {
  state.selected = state.selected === id ? null : id;
  paint();
  drawPanel();
}

// ---------- wiring ----------

function wire() {
  let debounce = null;
  $("load").addEventListener("input", (event) => {
    const value = Number(event.target.value);
    $("load-out").textContent = pct(value);
    clearTimeout(debounce);
    debounce = setTimeout(() => solve({ load_scale: value, outages: state.scenario.outages }), 150);
  });

  document.querySelectorAll("#views button").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      document.querySelectorAll("#views button").forEach((b) => b.setAttribute("aria-checked", b === button));
      paint();
      drawLegend();
      if (state.scenario && state.grid) syncUrl();
    });
  });

  document.querySelectorAll("#lang button").forEach((button) => {
    button.addEventListener("click", () => {
      lang = button.dataset.lang;
      try {
        localStorage.setItem("lang", lang);
      } catch {
        // The choice just won't be remembered.
      }
      applyLanguage();
    });
  });

  $("outages").addEventListener("click", (event) => {
    const id = event.target.dataset.restore;
    if (id != null) toggleBranch(state.grid.branches[Number(id)]);
  });

  $("alpha").addEventListener("click", (event) => {
    const value = event.target.dataset.alpha;
    if (value == null || !state.scenario) return;
    state.alpha = Number(value);
    drawAlpha();
    solve(state.scenario);
  });

  $("session-reset").addEventListener("click", () => {
    state.session.clear();
    if (state.sol && state.sol.staged) state.session.set(stateKey(state.sol), state.sol.staged);
    drawSession();
  });

  $("reset").addEventListener("click", () => solve({ load_scale: state.grid.load_scale.default, outages: [] }));

  const pick = (event) => {
    const item = event.target.closest("[data-branch]");
    if (item) select(Number(item.dataset.branch));
  };
  $("ranked").addEventListener("click", pick);
  $("ranked").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      pick(event);
    }
  });

  dark.addEventListener("change", () => {
    paint();
    drawLegend();
  });
}

async function init() {
  try {
    const res = await fetch("api/grid");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.grid = await res.json();
  } catch (err) {
    notify(t("unreachable"), init);
    return;
  }
  clearNotice();
  const scale = state.grid.load_scale;
  const slider = $("load");
  slider.min = scale.min;
  slider.max = scale.max;
  slider.step = scale.step;
  slider.disabled = false;
  $("grid-label").textContent = state.grid.label;
  if (state.grid.staged) state.alpha = state.grid.staged.default_alpha;
  state.scenario = scenarioFromUrl();
  slider.value = state.scenario.load_scale;
  $("load-out").textContent = pct(state.scenario.load_scale);
  document.querySelectorAll("#views button").forEach((b) => b.setAttribute("aria-checked", b.dataset.view === state.view));
  drawAlpha();
  drawGrid();
  drawLegend();
  solve(state.scenario);
}

wire();
applyLanguage();
init();
