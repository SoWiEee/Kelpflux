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
  tableWrap: document.querySelector(".table-wrap"),
  emptyState: document.querySelector("#empty-state"),
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
  loaded: false,
  inFlight: false,
  lastUpdated: null,
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
  if (sm !== null) return { value: `SM 使用率 ${formatNumber(sm)}%`, reported: true };
  const mps = numberFrom(source, ["mps_used", "mps_usage", "used_mps"]);
  if (mps !== null) return { value: `GPU 分享量 ${formatNumber(mps)}%`, reported: true };
  return { value: "尚未提供", reported: false };
}

function mpsSummary(jobs, key) {
  const values = jobs.map((job) => job[key]).filter((value) => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null;
}

function submittedValue(raw) {
  const value = firstDefined(raw, ["submit_time", "submitted_at", "submitted", "submit_ts"]);
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
  renderRows(monitor.jobs);
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
  dom.requestedMps.textContent = requested === null ? "--" : `${formatNumber(requested)}%`;
  dom.mpsMeta.textContent = requested === null ? "尚未回報需求量" : "所有工作合計（MPS 百分比總和）";
  dom.allocatedMps.textContent = allocated === null ? "--" : `${formatNumber(allocated)}%`;
}

function renderRows(jobs) {
  dom.jobsBody.replaceChildren();
  dom.emptyState.hidden = jobs.length !== 0;
  dom.tableWrap.hidden = jobs.length === 0;
  if (!jobs.length) return;

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

function renderLoading() {
  dom.jobsBody.replaceChildren();
  dom.tableWrap.hidden = false;
  dom.emptyState.hidden = true;
  const row = document.createElement("tr");
  row.className = "placeholder-row";
  const tableCell = document.createElement("td");
  tableCell.colSpan = 6;
  tableCell.innerHTML = '<div class="empty-state empty-state--loading"><span class="loading-line" aria-hidden="true"></span><strong>正在連線至工作佇列</strong><span>等待第一次回應。</span></div>';
  row.append(tableCell);
  dom.jobsBody.append(row);
}

function renderError() {
  dom.jobsBody.replaceChildren();
  dom.tableWrap.hidden = false;
  dom.emptyState.hidden = true;
  const row = document.createElement("tr");
  row.className = "placeholder-row";
  const tableCell = document.createElement("td");
  tableCell.colSpan = 6;
  tableCell.innerHTML = '<div class="empty-state"><div class="empty-icon" aria-hidden="true">!</div><strong>工作佇列無法使用</strong><span>請檢查 API 位址或重新連線後端。</span></div>';
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

function formatClock(date) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}
