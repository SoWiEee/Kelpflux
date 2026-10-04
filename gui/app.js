"use strict";

const POLL_INTERVAL_MS = 4000;
const API_STORAGE_KEY = "kelpflux.monitor.apiBase";
const DEFAULT_API_BASE = "/api";

const dom = {
  apiForm: document.querySelector("#api-form"),
  apiBase: document.querySelector("#api-base"),
  apiLabel: document.querySelector("#api-label"),
  refreshButton: document.querySelector("#refresh-button"),
  connectionLabel: document.querySelector("#connection-label"),
  statusDot: document.querySelector("#status-dot"),
  lastUpdated: document.querySelector("#last-updated"),
  errorNotice: document.querySelector("#error-notice"),
  errorMessage: document.querySelector("#error-message"),
  staleNotice: document.querySelector("#stale-notice"),
  jobsBody: document.querySelector("#jobs-body"),
  pendingStream: document.querySelector("#pending-stream"),
  pendingStageCount: document.querySelector("#pending-stage-count"),
  nodeBoard: document.querySelector("#node-board"),
  jobDetailCount: document.querySelector("#job-detail-count"),
  historyBody: document.querySelector("#history-body"),
  historyCount: document.querySelector("#history-count"),
  jobTooltip: document.querySelector("#job-tooltip"),
  totalJobs: document.querySelector("#total-jobs"),
  totalMeta: document.querySelector("#total-meta"),
  runningJobs: document.querySelector("#running-jobs"),
  pendingJobs: document.querySelector("#pending-jobs"),
  requestedMps: document.querySelector("#requested-mps"),
  mpsMeta: document.querySelector("#mps-meta"),
  nodeCount: document.querySelector("#node-count"),
  gpuCount: document.querySelector("#gpu-count"),
  allocatedMps: document.querySelector("#allocated-mps"),
  usageSignal: document.querySelector("#usage-signal"),
  schedulerReady: document.querySelector("#scheduler-ready"),
  schedulerAge: document.querySelector("#scheduler-age"),
  schedulerFreeMps: document.querySelector("#scheduler-free-mps"),
  schedulerAction: document.querySelector("#scheduler-action"),
  queueOldestWait: document.querySelector("#queue-oldest-wait"),
  queueAverageWait: document.querySelector("#queue-average-wait"),
  queueCycle: document.querySelector("#queue-cycle"),
  queueBackfill: document.querySelector("#queue-backfill"),
  gpuMetrics: document.querySelector("#gpu-metrics"),
};

const monitor = {
  apiBase: readApiBase(),
  jobs: [],
  history: [],
  loaded: false,
  inFlight: false,
  lastUpdated: null,
  placements: new Map(),
  knownJobs: new Set(),
  boardRendered: false,
};

dom.apiBase.value = monitor.apiBase;
dom.apiLabel.textContent = `API base: ${monitor.apiBase}`;

dom.apiForm.addEventListener("submit", (event) => {
  event.preventDefault();
  monitor.apiBase = normaliseApiBase(dom.apiBase.value);
  dom.apiBase.value = monitor.apiBase;
  localStorage.setItem(API_STORAGE_KEY, monitor.apiBase);
  dom.apiLabel.textContent = `API base: ${monitor.apiBase}`;
  monitor.loaded = false;
  monitor.jobs = [];
  renderLoading();
  poll();
});

dom.refreshButton.addEventListener("click", () => poll());

poll();
window.setInterval(poll, POLL_INTERVAL_MS);

function readApiBase() {
  const queryValue = new URLSearchParams(window.location.search).get("api");
  return normaliseApiBase(queryValue || localStorage.getItem(API_STORAGE_KEY) || DEFAULT_API_BASE);
}

function normaliseApiBase(value) {
  const trimmed = String(value || "").trim();
  return (trimmed || DEFAULT_API_BASE).replace(/\/+$/, "");
}

function jobsUrl() {
  return monitor.apiBase.toLowerCase().endsWith("/jobs")
    ? monitor.apiBase
    : `${monitor.apiBase}/jobs`;
}

async function poll() {
  if (monitor.inFlight) return;
  monitor.inFlight = true;
  setConnection("connecting", "連線中");

  try {
    const response = await fetch(jobsUrl(), {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`API returned HTTP ${response.status}`);

    const payload = await response.json();
    const rawJobs = extractJobs(payload);
    monitor.jobs = rawJobs.map(normaliseJob).filter(Boolean);
    monitor.history = extractHistory(payload).map(normaliseJob).filter(Boolean);
    monitor.loaded = true;
    monitor.lastUpdated = new Date();
    render(payload);
    setConnection("online", "已連線");
    dom.errorNotice.hidden = true;
    dom.staleNotice.hidden = true;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown API error";
    setConnection("error", "離線");
    dom.errorMessage.textContent = message;
    dom.errorNotice.hidden = false;
    dom.staleNotice.hidden = !monitor.loaded;
    if (!monitor.loaded) renderError();
  } finally {
    monitor.inFlight = false;
  }
}

function setConnection(kind, label) {
  dom.connectionLabel.textContent = label;
  dom.statusDot.className = `status-dot is-${kind}`;
  dom.refreshButton.disabled = kind === "connecting";
  if (monitor.lastUpdated) {
    dom.lastUpdated.textContent = `最後更新 ${formatClock(monitor.lastUpdated)}`;
  } else {
    dom.lastUpdated.textContent = kind === "error" ? "尚無成功回應" : "等待資料";
  }
}

function extractJobs(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload && Array.isArray(payload.jobs)) return payload.jobs;
  throw new Error("Response must contain a jobs array");
}

function extractHistory(payload) {
  return payload && Array.isArray(payload.history) ? payload.history : [];
}

function normaliseJob(raw) {
  if (!raw || typeof raw !== "object") return null;
  const id = raw.job_id ?? raw.id ?? raw.name;
  if (id === undefined || id === null || String(id).trim() === "") return null;

  const state = normaliseState(raw.job_state ?? raw.state);
  const gpu = gpuInfo(raw);
  return {
    id: String(id),
    state: state || "UNKNOWN",
    node: textValue(firstDefined(raw, ["node", "node_name", "nodes", "node_list", "allocated_node", "allocated_nodes"])),
    gpu,
    mpsRequested: mpsValue(raw, "requested"),
    mpsAllocated: mpsValue(raw, "allocated"),
    usage: usageInfo(raw),
    submitted: submittedValue(raw),
    started: dateValue(raw, ["start_time", "started_at", "start_ts"]),
    ended: dateValue(raw, ["end_time", "completed_at", "end_ts"]),
    name: textValue(firstDefined(raw, ["job_name", "name", "type"])) || "未命名工作",
    command: textValue(firstDefined(raw, ["command", "command_line", "cmd"])),
    partition: textValue(raw.partition),
    jctSeconds: numberFrom(raw, ["jct_seconds", "jct"]),
    waitSeconds: numberFrom(raw, ["wait_seconds", "queue_seconds"]),
    runtimeSeconds: numberFrom(raw, ["runtime_seconds", "elapsed_seconds", "elapsed"]),
  };
}

function firstDefined(object, keys) {
  for (const key of keys) {
    if (object[key] !== undefined && object[key] !== null && object[key] !== "") return object[key];
  }
  return null;
}

function normaliseState(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    value = value.current ?? value.name ?? value.state ?? "";
  }
  const tokens = Array.isArray(value)
    ? value
    : String(value || "").replace(/[|,]/g, " ").split(/\s+/);
  const known = tokens.map((token) => String(token).trim().toUpperCase()).filter(Boolean);
  return known.find((token) => !["JOB", "STATE"].includes(token)) || "";
}

function unwrap(value) {
  if (value && typeof value === "object") {
    if (value.number !== undefined) return value.number;
    if (value.value !== undefined) return value.value;
  }
  return value;
}

function numberOrNull(value) {
  const unwrapped = unwrap(value);
  if (unwrapped === undefined || unwrapped === null || unwrapped === "") return null;
  const number = Number(unwrapped);
  return Number.isFinite(number) ? number : null;
}

function numberFrom(object, keys) {
  for (const key of keys) {
    const number = numberOrNull(object[key]);
    if (number !== null) return number;
  }
  return null;
}

function textValue(value) {
  if (value && typeof value === "object") {
    if (value.name !== undefined) return textValue(value.name);
    if (value.hostname !== undefined) return textValue(value.hostname);
  }
  const text = String(unwrap(value) ?? "").trim();
  return ["", "(null)", "N/A", "none", "None"].includes(text) ? "" : text;
}

function tresText(object, keys) {
  return keys
    .map((key) => object[key])
    .filter((value) => value !== undefined && value !== null && value !== "")
    .map((value) => {
      if (typeof value === "string") return value;
      if (Array.isArray(value)) return value.join(",");
      return Object.entries(value).map(([key, item]) => `${key}=${unwrap(item)}`).join(",");
    })
    .join(",");
}

function parseTres(text, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const equal = new RegExp(`(?:^|[, ]|:)(?:gres/)?${escaped}(?:/[^=,: ]+)?=(\\d+(?:\\.\\d+)?)`, "i").exec(text);
  if (equal) return Number(equal[1]);
  const colon = new RegExp(`(?:^|[, ])(?:gres/)?${escaped}:(?:[^,: ]+:)?(\\d+(?:\\.\\d+)?)`, "i").exec(text);
  return colon ? Number(colon[1]) : null;
}

function mpsValue(raw, kind) {
  const mps = raw.mps && typeof raw.mps === "object" ? raw.mps : {};
  const direct = kind === "requested"
    ? numberFrom(raw, ["mps_requested", "mps_req", "requested_mps"])
    : numberFrom(raw, ["mps_allocated", "mps_alloc", "allocated_mps"]);
  const nested = kind === "requested"
    ? numberFrom(mps, ["requested", "request", "requested_slots"])
    : numberFrom(mps, ["allocated", "alloc", "allocated_slots"]);
  if (direct !== null) return direct;
  if (nested !== null) return nested;

  const source = kind === "requested"
    ? tresText(raw, ["tres_req_str", "tres_per_node", "gres"])
    : tresText(raw, ["tres_alloc_str", "tres_alloc", "allocated_tres", "alloc_tres", "gres_detail"]);
  return parseTres(source, "mps");
}

function gpuInfo(raw) {
  const explicit = firstDefined(raw, ["gpu", "gpus", "gpu_id", "gpu_ids", "allocated_gpu"]);
  if (Array.isArray(explicit)) {
    const items = explicit.map((item) => gpuItem(item)).filter(Boolean);
    if (items.length) return { label: items.map((item) => item.label).join(", "), count: items.length };
  }
  if (explicit && typeof explicit === "object") {
    const item = gpuItem(explicit);
    if (item) return { label: item.label, count: item.count };
    return { label: "", count: 0 };
  }
  if (explicit !== null) return { label: textValue(explicit), count: 1 };

  const detail = tresText(raw, ["gres_detail", "tres_alloc_str", "tres_alloc", "allocated_tres", "alloc_tres"]);
  const detailed = /(?:^|[, ])(?:gres\/)?gpu:([^:(),= ]+):(\d+)(?:\(IDX:([^)]*)\))?/i.exec(detail);
  if (detailed) {
    const type = detailed[1];
    const count = Number(detailed[2]);
    const indexes = detailed[3] ? `GPU ${detailed[3]}` : `GPU x${count}`;
    return { label: `${type} | ${indexes}`, count };
  }
  const count = parseTres(detail, "gpu");
  return count === null ? { label: "", count: 0 } : { label: `GPU x${count}`, count };
}

function gpuItem(value) {
  if (value === null || value === undefined) return null;
  if (typeof value !== "object") return { label: textValue(value), count: 1 };
  const type = textValue(value.type ?? value.gpu_type ?? value.name);
  const index = value.index ?? value.gpu_index ?? value.id;
  const count = numberOrNull(value.count ?? value.quantity) ?? 1;
  if (!type && index === undefined) return null;
  return {
    label: `${type || "GPU"}${index === undefined ? ` x${count}` : ` ${index}`}`,
    count,
  };
}

function usageInfo(raw) {
  const usage = raw.resource_usage ?? raw.resourceUsage ?? raw.usage;
  const source = usage && typeof usage === "object" ? usage : raw;
  const sm = numberFrom(source, ["sm_percent", "sm_utilization", "gpu_utilization", "gpu_util"]);
  const sharedSm = numberFrom(source, ["shared_gpu_sm_percent"]);
  const vram = numberFrom(source, ["vram_used_mib", "gpu_memory_used_mib"]);
  const vramPercent = numberFrom(source, ["vram_percent", "gpu_memory_percent"]);
  const parts = [];
  if (sm !== null) parts.push(`SM ${formatNumber(sm)}%`);
  else if (sharedSm !== null) parts.push(`共享 SM ${formatNumber(sharedSm)}%`);
  if (vram !== null) parts.push(`VRAM ${formatNumber(vram)} MiB`);
  else if (vramPercent !== null) parts.push(`VRAM ${formatNumber(vramPercent)}%`);
  if (parts.length) return { value: parts.join(" · "), reported: true };
  const mps = numberFrom(source, ["mps_used", "mps_usage", "used_mps"]);
  if (mps !== null) return { value: `GPU 分享量 ${formatNumber(mps)}%`, reported: true };
  return { value: "尚未提供", reported: false };
}

function mpsSummary(jobs, key) {
  const values = jobs.map((job) => job[key]).filter((value) => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
}

function submittedValue(raw) {
  return dateValue(raw, ["submit_time", "submitted_at", "submitted", "submit_ts"]);
}

function dateValue(raw, keys) {
  const value = firstDefined(raw, keys);
  const number = numberOrNull(value);
  if (number !== null && number > 0) {
    const timestamp = number < 1e12 ? number * 1000 : number;
    const date = new Date(timestamp);
    if (!Number.isNaN(date.valueOf())) return date;
  }
  if (typeof value === "string") {
    const date = new Date(value);
    if (!Number.isNaN(date.valueOf())) return date;
  }
  return null;
}

function render(payload) {
  renderSummary(monitor.jobs);
  renderFlow(monitor.jobs, payload?.nodes);
  renderRows(monitor.jobs);
  renderHistory(monitor.history);
  renderScheduler(payload?.scheduler);
  renderQueueMetrics(payload?.queue_metrics);
  renderGpuMetrics(payload?.gpu_metrics);

  const jobNodes = new Set(monitor.jobs.map((job) => job.node).filter(Boolean));
  const jobGpus = monitor.jobs.reduce((total, job) => total + job.gpu.count, 0);
  dom.nodeCount.textContent = `${jobNodes.size} 個節點`;
  dom.gpuCount.textContent = `${jobGpus} 張`;
  const usageJobs = monitor.jobs.filter((job) => job.usage.reported).length;
  dom.usageSignal.textContent = usageJobs ? `${usageJobs} 個工作有回報` : "尚未提供";
}

function renderFlow(jobs, rawNodes) {
  const pending = jobs.filter((job) => job.state === "PENDING");
  const nodes = normaliseGpuNodes(rawNodes, jobs);
  renderPendingJobs(pending);
  renderNodeBoard(nodes, jobs);
  monitor.placements = new Map(jobs.map((job) => [job.id, job.node]));
  monitor.knownJobs = new Set(jobs.map((job) => job.id));
  monitor.boardRendered = true;
}

function renderPendingJobs(jobs) {
  dom.pendingStream.replaceChildren();
  dom.pendingStageCount.textContent = `${jobs.length} 個工作`;
  if (!jobs.length) {
    dom.pendingStream.append(stageEmpty("目前沒有等待中的工作"));
    return;
  }
  jobs.forEach((job, index) => {
    const tile = document.createElement("article");
    const isNew = monitor.boardRendered && !monitor.knownJobs.has(job.id);
    tile.className = `pending-job${isNew ? " is-new" : ""}`;
    tile.style.animationDelay = `${Math.min(index, 7) * 45}ms`;
    tile.setAttribute("aria-label", `佇列第 ${index + 1} 位，工作 ${job.id}，要求 ${formatMaybe(job.mpsRequested, "% MPS")}`);
    attachJobTooltip(tile, job);
    const order = document.createElement("span");
    order.className = "pending-order";
    order.textContent = `Q${index + 1}`;
    const id = document.createElement("strong");
    id.textContent = `#${job.id}`;
    const mps = document.createElement("span");
    mps.className = "pending-mps";
    mps.textContent = formatMaybe(job.mpsRequested, "% MPS");
    tile.append(order, id, mps);
    dom.pendingStream.append(tile);
  });
}

function normaliseGpuNodes(rawNodes, jobs) {
  const source = Array.isArray(rawNodes) ? rawNodes : [];
  const nodes = source
    .map((node) => ({
      name: textValue(node.name ?? node.node_name ?? node.NodeName),
      state: textValue(node.state ?? node.State) || "UNKNOWN",
      capacity: numberOrNull(node.mps_capacity ?? node.capacity) ?? 0,
      allocated: numberOrNull(node.mps_allocated ?? node.allocated) ?? 0,
      gpuMetrics: Array.isArray(node.gpu_metrics) ? node.gpu_metrics : [],
    }))
    .filter((node) => node.name && (node.capacity > 0 || node.gpuMetrics.length || node.name.includes("-gpu-")));
  if (nodes.length) return nodes;
  return [...new Set(jobs.map((job) => job.node).filter(Boolean))].map((name) => ({
    name,
    state: "ALLOCATED",
    capacity: 100,
    allocated: mpsSummary(jobs.filter((job) => job.node === name), "mpsAllocated") ?? 0,
    gpuMetrics: [],
  }));
}

function renderNodeBoard(nodes, jobs) {
  dom.nodeBoard.replaceChildren();
  if (!nodes.length) {
    dom.nodeBoard.append(stageEmpty("尚未取得 GPU node 狀態"));
    return;
  }
  nodes.forEach((node) => {
    const assigned = jobs.filter((job) => job.node === node.name && job.state !== "PENDING");
    const lane = document.createElement("article");
    lane.className = "node-lane";

    const header = document.createElement("header");
    const identity = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = gpuDisplayName(node.name);
    const hostname = document.createElement("span");
    hostname.textContent = node.name;
    identity.append(name, hostname);
    const capacity = document.createElement("div");
    capacity.className = "node-capacity";
    const state = document.createElement("span");
    state.textContent = nodeStateLabel(node.state);
    const usage = document.createElement("strong");
    usage.textContent = `${formatNumber(node.allocated)} / ${formatNumber(node.capacity || 100)}%`;
    capacity.append(state, usage);
    header.append(identity, capacity);

    const track = document.createElement("div");
    track.className = "mps-track";
    const slots = document.createElement("div");
    slots.className = "capacity-slots";
    slots.setAttribute("aria-hidden", "true");
    for (let index = 0; index < 4; index += 1) {
      const slot = document.createElement("span");
      slot.textContent = "25%";
      slots.append(slot);
    }
    const allocations = document.createElement("div");
    allocations.className = "allocation-grid";
    assigned.forEach((job) => allocations.append(allocationBlock(job, node.name)));
    track.append(slots, allocations);
    lane.append(header, track);
    if (!assigned.length) lane.append(stageEmpty("100% MPS 可用", "node-empty"));
    dom.nodeBoard.append(lane);
  });
}

function allocationBlock(job, nodeName) {
  const allocated = job.mpsAllocated ?? job.mpsRequested ?? 25;
  const slots = Math.max(1, Math.min(4, Math.ceil(allocated / 25)));
  const previousNode = monitor.placements.get(job.id);
  const isDispatched = monitor.boardRendered && previousNode !== nodeName;
  const block = document.createElement("article");
  block.className = `allocation-job${isDispatched ? " is-dispatched" : ""}`;
  block.style.gridColumn = `span ${slots}`;
  block.setAttribute("aria-label", `工作 ${job.id} 已配置至 ${nodeName}，使用 ${allocated}% MPS`);
  attachJobTooltip(block, job);
  const id = document.createElement("strong");
  id.textContent = `#${job.id}`;
  const mps = document.createElement("span");
  mps.textContent = `${formatNumber(allocated)}% MPS`;
  block.append(id, mps);
  return block;
}

function stageEmpty(message, className = "stage-empty") {
  const empty = document.createElement("div");
  empty.className = className;
  empty.textContent = message;
  return empty;
}

function gpuDisplayName(nodeName) {
  const match = /gpu-([^-]+)-/i.exec(nodeName);
  if (!match) return "GPU Node";
  return match[1].replace(/^rtx/i, "RTX ").toUpperCase();
}

function nodeStateLabel(value) {
  const state = String(value || "").toUpperCase().split(/[+\/]/)[0];
  return {
    IDLE: "可用",
    MIXED: "部分使用",
    ALLOCATED: "已滿載",
    COMPLETING: "工作結束中",
    DOWN: "離線",
    FUTURE: "未啟用",
  }[state] || state;
}

function renderScheduler(scheduler) {
  if (!scheduler || typeof scheduler !== "object") {
    dom.schedulerReady.textContent = "尚未提供";
    dom.schedulerAge.textContent = "尚未提供";
    dom.schedulerFreeMps.textContent = "尚未提供";
    dom.schedulerAction.textContent = "尚未提供";
    return;
  }
  const ready = numberOrNull(scheduler.ready);
  const shadow = numberOrNull(scheduler.shadow_mode);
  const age = numberOrNull(scheduler.snapshot_age_s);
  const freeMps = numberOrNull(scheduler.free_mps);
  const jobIndex = numberOrNull(scheduler.last_job_index);
  const nodeIndex = numberOrNull(scheduler.last_node_index);
  const gpuIndex = numberOrNull(scheduler.last_gpu_index);
  dom.schedulerReady.textContent = ready === null ? "尚未提供" : ready > 0 ? "已就緒" : "未就緒";
  if (shadow !== null && shadow > 0) dom.schedulerReady.textContent += "（觀察模式）";
  dom.schedulerAge.textContent = age === null ? "尚未提供" : `${formatNumber(age)} 秒`;
  dom.schedulerFreeMps.textContent = freeMps === null ? "尚未提供" : `${formatNumber(freeMps)}%`;
  dom.schedulerAction.textContent = jobIndex === null
    ? "尚未提供"
    : `槽${formatNumber(jobIndex)} · 節點${formatMaybe(nodeIndex)} · GPU${formatMaybe(gpuIndex)}`;
}

function renderQueueMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") {
    dom.queueOldestWait.textContent = "尚未提供";
    dom.queueAverageWait.textContent = "尚未提供";
    dom.queueCycle.textContent = "尚未提供";
    dom.queueBackfill.textContent = "尚未提供";
    return;
  }
  const seconds = (value) => {
    const number = numberOrNull(value);
    return number === null ? "尚未提供" : `${formatNumber(number)} 秒`;
  };
  dom.queueOldestWait.textContent = seconds(metrics.oldest_wait_s);
  dom.queueAverageWait.textContent = seconds(metrics.average_wait_s);
  dom.queueCycle.textContent = seconds(metrics.scheduler_cycle_s);
  const backfill = numberOrNull(metrics.backfill_queue);
  dom.queueBackfill.textContent = backfill === null ? "尚未提供" : formatNumber(backfill);
}

function renderGpuMetrics(metrics) {
  dom.gpuMetrics.replaceChildren();
  if (!Array.isArray(metrics) || !metrics.length) {
    const empty = document.createElement("div");
    empty.className = "gpu-empty";
    empty.textContent = "尚未提供 GPU 即時資料";
    dom.gpuMetrics.append(empty);
    return;
  }
  metrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "gpu-card";
    const heading = document.createElement("div");
    heading.className = "gpu-card-heading";
    const name = document.createElement("strong");
    name.textContent = `${metric.gpu_type || "GPU"} · GPU ${formatMaybe(numberOrNull(metric.gpu_index))}`;
    const state = document.createElement("span");
    state.className = `gpu-state${metric.available === false ? " gpu-state--offline" : ""}`;
    state.textContent = metric.available === false ? "離線" : "即時";
    heading.append(name, state);
    const node = document.createElement("span");
    node.className = "gpu-node";
    node.textContent = metric.node || "尚未提供節點";
    const stats = document.createElement("dl");
    stats.className = "gpu-stats";
    addGpuStat(stats, "SM", metric.gpu_util_percent, "%");
    addGpuStat(stats, "VRAM", metric.vram_percent, "%");
    addGpuStat(stats, "功耗", metric.power_w, " W");
    addGpuStat(stats, "溫度", metric.temperature_c, " °C");
    card.append(heading, node, stats);
    dom.gpuMetrics.append(card);
  });
}

function addGpuStat(list, label, value, suffix) {
  const item = document.createElement("div");
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  const number = numberOrNull(value);
  detail.textContent = number === null ? "--" : `${formatNumber(number)}${suffix}`;
  item.append(term, detail);
  list.append(item);
}

function renderSummary(jobs) {
  const pending = jobs.filter((job) => job.state === "PENDING").length;
  const running = jobs.filter((job) => ["RUNNING", "COMPLETING"].includes(job.state)).length;
  const requested = mpsSummary(jobs, "mpsRequested");
  const allocated = mpsSummary(jobs, "mpsAllocated");

  dom.totalJobs.textContent = String(jobs.length);
  dom.totalMeta.textContent = `${pending} 個等待 / ${running} 個執行中`;
  dom.pendingJobs.textContent = String(pending);
  dom.runningJobs.textContent = String(running);
  dom.pendingJobs.closest(".summary-card").classList.toggle("is-active", pending > 0);
  dom.runningJobs.closest(".summary-card").classList.toggle("is-active", running > 0);
  dom.requestedMps.textContent = requested === null ? "--" : `${formatNumber(requested)}%`;
  dom.mpsMeta.textContent = requested === null ? "尚未回報需求量" : "所有工作合計（MPS 百分比總和）";
  dom.allocatedMps.textContent = allocated === null ? "--" : `${formatNumber(allocated)}%`;
}

function renderRows(jobs) {
  dom.jobsBody.replaceChildren();
  dom.jobDetailCount.textContent = String(jobs.length);
  if (!jobs.length) {
    const row = document.createElement("tr");
    row.className = "placeholder-row";
    const tableCell = document.createElement("td");
    tableCell.colSpan = 6;
    tableCell.textContent = "目前沒有活動中的工作。";
    row.append(tableCell);
    dom.jobsBody.append(row);
    return;
  }

  jobs.forEach((job, index) => {
    const row = document.createElement("tr");
    row.style.animationDelay = `${Math.min(index, 8) * 25}ms`;
    row.append(
      cell("job-id", job.id),
      stateCell(job.state),
      placementCell(job),
      mpsCell(job),
      usageCell(job),
      submittedCell(job.submitted),
    );
    dom.jobsBody.append(row);
  });
}

function renderHistory(jobs) {
  dom.historyBody.replaceChildren();
  dom.historyCount.textContent = String(jobs.length);
  if (!jobs.length) {
    const row = document.createElement("tr");
    row.className = "placeholder-row";
    const tableCell = document.createElement("td");
    tableCell.colSpan = 6;
    tableCell.textContent = "最近 24 小時沒有完成紀錄。";
    row.append(tableCell);
    dom.historyBody.append(row);
    return;
  }
  jobs.forEach((job) => {
    const row = document.createElement("tr");
    row.setAttribute("aria-label", `已完成工作 ${job.id}，${job.name}`);
    attachJobTooltip(row, job);
    row.append(
      historyJobCell(job),
      stateCell(job.state),
      cell("duration-cell", formatDuration(job.jctSeconds)),
      cell("duration-cell", formatDuration(job.waitSeconds)),
      cell("duration-cell", formatDuration(job.runtimeSeconds)),
      submittedCell(job.ended),
    );
    dom.historyBody.append(row);
  });
}

function historyJobCell(job) {
  const tableCell = document.createElement("td");
  const wrapper = document.createElement("div");
  wrapper.className = "history-job";
  const id = document.createElement("strong");
  id.textContent = `#${job.id}`;
  const name = document.createElement("span");
  name.textContent = job.name;
  wrapper.append(id, name);
  tableCell.append(wrapper);
  return tableCell;
}

function attachJobTooltip(element, job) {
  element.tabIndex = 0;
  element.addEventListener("mouseenter", () => showJobTooltip(element, job));
  element.addEventListener("mouseleave", hideJobTooltip);
  element.addEventListener("focus", () => showJobTooltip(element, job));
  element.addEventListener("blur", hideJobTooltip);
}

function showJobTooltip(anchor, job) {
  dom.jobTooltip.replaceChildren(
    tooltipRow("工作類型", job.name),
    tooltipRow("命令", job.command || "Slurm 未提供"),
    tooltipRow("Partition", job.partition || "--"),
    tooltipRow("提交時間", job.submitted ? formatDate(job.submitted) : "--"),
    tooltipRow("即時用量", job.usage.value),
  );
  dom.jobTooltip.hidden = false;
  const anchorRect = anchor.getBoundingClientRect();
  const tooltipRect = dom.jobTooltip.getBoundingClientRect();
  const margin = 10;
  const left = Math.min(
    window.innerWidth - tooltipRect.width - margin,
    Math.max(margin, anchorRect.left + (anchorRect.width - tooltipRect.width) / 2),
  );
  const below = anchorRect.bottom + margin;
  const top = below + tooltipRect.height <= window.innerHeight
    ? below
    : Math.max(margin, anchorRect.top - tooltipRect.height - margin);
  dom.jobTooltip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}

function hideJobTooltip() {
  dom.jobTooltip.hidden = true;
}

function tooltipRow(label, value) {
  const row = document.createElement("div");
  const term = document.createElement("span");
  term.textContent = label;
  const detail = document.createElement("strong");
  detail.textContent = value;
  row.append(term, detail);
  return row;
}

function renderLoading() {
  dom.pendingStageCount.textContent = "--";
  dom.pendingStream.replaceChildren(stageEmpty("正在取得工作佇列"));
  dom.nodeBoard.replaceChildren(stageEmpty("正在取得 GPU node 狀態"));
  dom.jobDetailCount.textContent = "--";
  dom.historyCount.textContent = "--";
  dom.historyBody.replaceChildren();
  const historyRow = document.createElement("tr");
  const historyCell = document.createElement("td");
  historyCell.colSpan = 6;
  historyCell.textContent = "正在取得完成紀錄。";
  historyRow.append(historyCell);
  dom.historyBody.append(historyRow);
  dom.jobsBody.replaceChildren();
  const row = document.createElement("tr");
  row.className = "placeholder-row";
  const tableCell = document.createElement("td");
  tableCell.colSpan = 6;
  tableCell.textContent = "正在連線至工作佇列。";
  row.append(tableCell);
  dom.jobsBody.append(row);
}

function renderError() {
  dom.pendingStageCount.textContent = "--";
  dom.pendingStream.replaceChildren(stageEmpty("工作佇列目前無法使用"));
  dom.nodeBoard.replaceChildren(stageEmpty("GPU node 狀態目前無法使用"));
  dom.jobDetailCount.textContent = "--";
  dom.historyCount.textContent = "--";
  dom.historyBody.replaceChildren();
  const historyRow = document.createElement("tr");
  const historyCell = document.createElement("td");
  historyCell.colSpan = 6;
  historyCell.textContent = "完成紀錄目前無法使用。";
  historyRow.append(historyCell);
  dom.historyBody.append(historyRow);
  dom.jobsBody.replaceChildren();
  const row = document.createElement("tr");
  row.className = "placeholder-row";
  const tableCell = document.createElement("td");
  tableCell.colSpan = 6;
  tableCell.textContent = "請檢查 API 位址或重新連線後端。";
  row.append(tableCell);
  dom.jobsBody.append(row);
}

function cell(className, value) {
  const tableCell = document.createElement("td");
  tableCell.className = className;
  tableCell.textContent = value || "--";
  return tableCell;
}

function stateCell(state) {
  const tableCell = document.createElement("td");
  const badge = document.createElement("span");
  const stateClass = state.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  badge.className = `state-badge state-badge--${stateClass}`;
  badge.textContent = state;
  tableCell.append(badge);
  return tableCell;
}

function placementCell(job) {
  const tableCell = document.createElement("td");
  const wrapper = document.createElement("div");
  wrapper.className = "placement";
  const node = document.createElement("strong");
  node.textContent = job.node || "尚未分配節點";
  const gpu = document.createElement("span");
  gpu.textContent = job.gpu.label || "尚未分配 GPU";
  wrapper.append(node, gpu);
  tableCell.append(wrapper);
  return tableCell;
}

function mpsCell(job) {
  const tableCell = document.createElement("td");
  const wrapper = document.createElement("div");
  wrapper.className = "mps-cell";
  const value = document.createElement("strong");
  value.textContent = `${formatMaybe(job.mpsRequested, "%")} / ${formatMaybe(job.mpsAllocated, "%")}`;
  const detail = document.createElement("span");
  detail.textContent = "要求 / 已配置（MPS 百分比）";
  wrapper.append(value, detail);
  tableCell.append(wrapper);
  return tableCell;
}

function usageCell(job) {
  const tableCell = document.createElement("td");
  const wrapper = document.createElement("div");
  wrapper.className = "usage-cell";
  const value = document.createElement("span");
  value.className = `usage-value${job.usage.reported ? "" : " usage-value--missing"}`;
  value.textContent = job.usage.value;
  const detail = document.createElement("span");
  detail.textContent = job.usage.reported ? "API 即時回報" : "尚無工作層級資料";
  wrapper.append(value, detail);
  tableCell.append(wrapper);
  return tableCell;
}

function submittedCell(date) {
  const tableCell = document.createElement("td");
  tableCell.className = "submitted";
  tableCell.textContent = date ? formatDate(date) : "--";
  if (date) tableCell.title = date.toISOString();
  return tableCell;
}

function formatMaybe(value, suffix = "") {
  return value === null ? "--" : `${formatNumber(value)}${suffix}`;
}

function formatNumber(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatDate(date) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatDuration(value) {
  const seconds = numberOrNull(value);
  if (seconds === null) return "--";
  if (seconds < 60) return `${formatNumber(seconds)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes} 分 ${remainder} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 時 ${minutes % 60} 分`;
}

function formatClock(date) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}
