# DRA 遷移方案：以 NVIDIA DRA driver 取代 device-plugin + MPS stack

> 研究性規劃文件（2026-07-08）。目標：用 Kubernetes DRA（`resource.k8s.io/v1`，v1.34 GA）+
> `kubernetes-sigs/dra-driver-nvidia-gpu` 取代目前脆弱的 gpu-operator device-plugin ＋
> mps-control-daemon ＋ CDI `/mps` 注入路徑（該路徑的失效模式見
> `~/.claude` memory `project-mps-recovery-after-gaming`）。**尚未執行**；先在單節點 pilot。

## 0. Readiness（Kelpflux 已滿足全部前置，2026-07-08 實測）

| 前置 | 需求 | Kelpflux | 狀態 |
|---|---|---|---|
| Kubernetes | ≥ v1.34.2（DRA GA） | v1.34.6+k3s1 | ✅ |
| DRA API | `resource.k8s.io/v1` | deviceclasses/resourceclaims/resourceslices 已在線 | ✅ |
| feature gate | `DynamicResourceAllocation` | `kubernetes_feature_enabled{...}=1` | ✅ |
| CDI | containerd 2.0+（預設開） | containerd v2.2.2；`/var/run/cdi/` 已在用 | ✅ |
| NVIDIA driver | ≥ v565（GPU 分配） | acane 580.167.08 / node-2 580.159.03 | ✅ |
| NFD | GPU 節點標籤 | gpu-operator-node-feature-discovery 已部署 | ✅ |
| Helm | ≥ v3.8 | （確認本機 helm 版本） | ⏳ |

**結論：不需升級 k3s 或任何元件，DRA 現在就能裝。**

### 0.1 版本澄清：核心 DRA 是 v1.34 GA（不是 1.35）

常見混淆：DRA 不是單一開關，而是一「家子」feature gate。**核心 `DynamicResourceAllocation`
在 v1.34 就 GA + 預設啟用**；1.35/1.36 是**進階子功能**畢業的時間。Kelpflux 1.34.6 實測（逐 gate）：

| stage | gate | 預設 |
|---|---|---|
| **GA** | `DynamicResourceAllocation`（核心，MPS 用例只需這個） | ✅ 開 |
| BETA（預設開） | `DRAAdminAccess`、`DRAPrioritizedList`、`DRAResourceClaimDeviceStatus`、`DRASchedulerFilterTimeout`、`KubeletPodResourcesDynamicResources` | ✅ 開 |
| ALPHA（預設關） | `DRAPartitionableDevices`、`DRAConsumableCapacity`、`DRADeviceTaints`、`DRADeviceBindingConditions`、`DRAExtendedResource` | ❌ 關 |

API 服務的是 `resource.k8s.io/**v1**`（GA），非 v1beta1。MPS 整卡分享靠 NVIDIA driver 的 `GpuConfig`
opaque config，**不依賴任何 alpha gate**（那些是 MIG 動態切分用）。故 1.34.6 即就緒，無須等 1.35。

## 1. 架構：什麼取代什麼

```
現況（device-plugin 路線）                     DRA 路線
────────────────────────────                 ──────────────────────────────
kube-system nvidia-device-plugin  ─┐          （刪除）
gpu-operator nvidia-device-plugin ─┼─ 搶 socket  → dra-driver-...-kubelet-plugin（每 GPU 節點，gpus 容器）
gpu-operator mps-control-daemon    │             → driver 依 ResourceClaim 宣告式託管 MPS daemon
  + config-manager（panic）        │             （templates/mps-control-daemon.tmpl.yaml）
CDI /mps 注入（race）           ───┘             → CDI 由 driver 依 claim 注入 /mps + env
worker pod: resources.limits         →          worker pod: resourceClaims → ResourceClaimTemplate
  nvidia.com/gpu: 1                              （gpu.nvidia.com, GpuConfig strategy=MPS）
```

**核心不變**：worker pod 仍拿到「整張 GPU + `/mps` pipe」，pod 內 **Slurm 照舊用 `gres/mps:N` 細分**。
DRA 只改「GPU/MPS 怎麼曝露給那個長壽 worker pod」。

## 2. ⚠️ 唯一的 go/no-go 設計問題（pilot 必驗）

Slurm 的 `gres/mps:N` 是 **Slurm 自己的排程帳目**；每個 job 的實際 MPS 算力切分是 Slurm 透過
**`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`（每 job env）**設定的。因此遷移要成立的條件是：

> **DRA 的 MPS claim 必須給 worker pod「不設上限（或 100%）」的 MPS server 存取，**
> **讓 Slurm 的每-job thread% 仍能生效——而不是被 claim 級 `defaultActiveThreadPercentage` 綁死。**

Quickstart（`demo/specs/quickstart/v1/gpu-test-mps.yaml`）的 `GpuConfig` 支援
`mpsConfig.defaultActiveThreadPercentage` 與 `defaultPinnedDeviceMemoryLimit`；
demo（`sharing: {strategy: MPS}`，無 mpsConfig）則不設限。**pilot 要驗證的就是「不設限 → Slurm 內部細分照常」。**

> **✅ 已於 2026-07-08 在 node-1 pilot 驗證通過（見 §6）。** `defaultActiveThreadPercentage: 100`
> 下，兩個容器的 CUDA workload 同時共卡多工成功。DRA 注入的 pipe 是
> **`CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`**（非 device-plugin 的 `/mps/nvidia.com/gpu/pipe`）。

## 3. 步驟（單節點 pilot：先只切 node-1 / 4070）

### 3.1 Pre-flight（可回滾）
- 記錄現況：`kubectl get ds -A | grep -i nvidia`、`helm list -A`、worker StatefulSet 的 GPU 請求區塊。
- 確認 `helm version` ≥ 3.8。
- 保留現有 gpu-operator 設定的備份（回滾用）。

### 3.2 停用 device-plugin（兩個都要，否則仍搶 `nvidia.com/gpu`）
- gpu-operator 那個：`--set devicePlugin.enabled=false`（helm upgrade gpu-operator）。
- kube-system 那個（`nvidia-device-plugin-daemonset`，71 天的遺留）：找出其 DaemonSet 來源並停用/刪除。
  > 這兩個正是 memory `project-mps-recovery-after-gaming` 裡「雙 plugin 搶 socket」的元兇；DRA 下 workload 改用 ResourceClaim，兩者變多餘。

### 3.3 安裝 DRA driver（GPU 分配開、ComputeDomain 關）
```bash
helm install dra-driver-nvidia-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
  --version <latest> \
  --create-namespace --namespace dra-driver-nvidia-gpu \
  --set gpuResourcesEnabledOverride=true \
  --set resources.computeDomains.enabled=false \
  --set nvidiaDriverRoot=/ \          # Kelpflux 用 host driver（driver.enabled=false），故為 /
  --set featureGates.MPSSupport=true  # ★ 必須：MPS 被 driver 級 MPSSupport gate 擋著，預設關（pilot 實測）
```
> pilot 教訓：**少了 `featureGates.MPSSupport=true`，claim 分配會 `FailedPrepareDynamicResources`：**
> `"MPS" is selected... but the "MPSSupport" feature gate is not enabled`。安裝文件沒明講此 gate。
> 若走 gpu-operator 託管路徑（推薦，讓它管 driver/CDI/NFD 並停用自家 device-plugin），
> 依 NVIDIA「GPU Operator DRA install guide」：
> https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html

### 3.4 驗證 driver 上線
```bash
kubectl get pod -n dra-driver-nvidia-gpu           # controller(關CD則無) + kubelet-plugin 1/1
kubectl get deviceclass                            # 應見 gpu.nvidia.com
kubectl get resourceslice -o wide                  # 每 GPU 節點一個 gpu.nvidia.com slice
```

### 3.5 建立 MPS DeviceClass / ResourceClaimTemplate（不設限，給 Slurm 全權）
```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata: { name: slurm-gpu-mps, namespace: slurm }
spec:
  spec:
    devices:
      requests:
      - name: gpu
        exactly: { deviceClassName: gpu.nvidia.com }
      config:
      - requests: ["gpu"]
        opaque:
          driver: gpu.nvidia.com
          parameters:
            apiVersion: resource.nvidia.com/v1beta1
            kind: GpuConfig
            sharing:
              strategy: MPS
              # 關鍵：不設 defaultActiveThreadPercentage（或設 100）→ 讓 Slurm 的
              # 每-job CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 生效
```

### 3.6 遷移 Slurm worker StatefulSet（先只 4070）
把 container 的
```yaml
resources: { limits: { nvidia.com/gpu: 1 } }
```
換成
```yaml
resources:
  claims: [ { name: gpu } ]
# 於 pod spec.resourceClaims:
resourceClaims:
- name: gpu
  resourceClaimTemplateName: slurm-gpu-mps
```

### 3.7 驗收（決定 go/no-go）
1. worker pod 內 `cat /proc/1/environ | tr '\0' '\n' | grep CUDA_MPS` → `/mps` pipe 有進來。
2. Slurm `sinfo` 該節點 `mps:rtx4070:100` 仍在、State=IDLE。
3. **多工測試**：`4×--gres=mps:25` 長 job → `sacct -X` 四個同時 RUNNING/COMPLETED（非 1/4 FAILED）。
4. Slurm 有正確為各 job 設 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`（驗證 DRA 沒把 pod 綁在低上限）。

**全過 → 再切 node-2、拔掉舊 stack。任一不過 → 回滾（3.8）。**

### 3.8 回滾
- 還原 worker StatefulSet 的 `nvidia.com/gpu: 1` 請求。
- `helm upgrade gpu-operator --set devicePlugin.enabled=true`；還原 kube-system device-plugin。
- `helm uninstall dra-driver-nvidia-gpu -n dra-driver-nvidia-gpu`。

## 4. 風險與現況標記
- **DRA driver 的 GPU plugin「非官方支援、預設停用（opt-in `gpuResourcesEnabledOverride=true`）」**——
  功能可用（quickstart/bats 有 MPS 測試、blog 於 1.35 實測 time-slicing），但無官方 SLA。屬風險偏好，非技術阻礙。
- **步驟 2 的 §2 設計問題是真正的 pilot gate**：DRA MPS 能否「不設限」讓 Slurm 內部細分。
- **不要同時切兩節點**：先 4070 pilot，node-2（3080，見 memory `project-node2-hw-constraints`）狀況更脆弱，最後再切。

## 5. 收益（若 pilot 通過）
一次拔掉今天咬我們一整個下午的脆弱來源：雙 plugin 搶 socket、config-manager panic、
CDI `/mps` 注入 race、worker pod 掉 mount。MPS 設定變成一份宣告式 ResourceClaimTemplate。

## 6. Pilot 結果（2026-07-08，node-1 / 4070）

**加法式安裝 + MPS 共卡測試，go/no-go 全數通過。全程未動 device-plugin / Slurm worker / node-2。**

安裝（driver v0.4.1，限 node-1，GPU 開、CD 關、`featureGates.MPSSupport=true`）→
- ✅ DeviceClass `gpu.nvidia.com` 註冊、node-1 advertise `ResourceSlice`（device=gpu-0, arch=Ada Lovelace）
- ✅ kubelet-plugin `1/1 Running`；kyverno 未擋

MPS 共卡測試（`ResourceClaimTemplate` strategy=MPS/thread%=100 + 一個 pod 兩容器，各跑真實
`/shared/bin/gpu_workload`）→ host `nvidia-smi` 實證：
```
compute mode: Exclusive_Process              ← DRA 自動設定
nvidia-cuda-mps-server (30 MiB)              ← DRA 自動啟動 MPS server
gpu_workload (868 MiB) ×2 併發               ← 兩 CUDA 程序同時共卡 = MPS 多工成立
GPU util 100%
```
pod 內 `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`（含完整 control/pid/log 檔）。

**關鍵發現（納入上方步驟）**：
1. **`featureGates.MPSSupport=true` 必開**（否則 claim `FailedPrepareDynamicResources`）。
2. **MPS pipe 路徑 = `/tmp/nvidia-mps`**（非 `/mps/...`）。對 Slurm 遷移的意涵：worker pod 的
   slurmd（PID1）environ 會拿到此值，`chart/lua`／`10-mps-env.sh` prolog 照樣傳遞給 job；
   Slurm 的每-job `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 疊加其上 → **Slurm 內部 `mps:N` 細分應照常**
   （thread%=100 不設限已驗證兩程序各取所需）。這是 §3.6/§3.7 遷移的最後待驗點。
3. **生命週期**：claim 分配時 DRA 設 Exclusive_Process＋起 MPS server；釋放時拆除 MPS server，
   但 compute mode 留在 Exclusive_Process（要玩遊戲需手動 `nvidia-smi -c 0`，DRA 下次分配會再設回）。

**結論：DRA-on-Kelpflux 的核心可行性已證實。** 目前 DRA driver 仍裝在 node-1。

## 6.1 Slurm worker 正式遷移驗證（2026-07-08，node-1，✅ 通過）

實際把 `slurm-worker-gpu-rtx4070` StatefulSet 從 device-plugin 遷到 DRA ResourceClaim：
1. 建永久 `ResourceClaimTemplate slurm-gpu-mps`（slurm ns，MPS，thread%=100，mem 11Gi）。
2. JSON patch StatefulSet（備份於 `/tmp/sts-rtx4070-backup.yaml`）：移除 `runtimeClassName: nvidia`、
   移除 `resources.{limits,requests}.nvidia.com/gpu`、加 `resources.claims:[{name:gpu}]` +
   `spec.template.spec.resourceClaims:[{name:gpu, resourceClaimTemplateName:slurm-gpu-mps}]`。
3. 刪 pod 觸發用新 template 重建。

**驗證結果**：
- ✅ worker pod `1/1 Running`（0 restart）；DRA 注入 `/dev/nvidia0`、`nvidia-smi` 見 4070、
  `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`（bind-mount 自 DRA 容器內的 MPS control daemon，
  ppid=containerd-shim → 確認是 DRA 託管，非殘留）。
- ✅ Slurm 節點 `idle gpu:rtx4070:1,mps:rtx4070:100`（gres.conf 的 `File=/dev/nvidia0` 被 CDI 滿足）。
- ✅ **Slurm 內部細分成立**：job 環境 `PIPE=/tmp/nvidia-mps`、Slurm 依 `mps:25` 自動設
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25`。
- ✅ **4×`mps:25` 併發多工**：host `nvidia-smi` 實測 4 個 `gpu_workload` 同時在 4070 上跑。
- ✅ node-2（3080）完全未受影響（仍走 device-plugin）。

**⚠️ 唯一 caveat — MPS server 冷啟動競爭**：DRA 的 MPS *server*（非 control daemon）在無 client
時關閉；一次丟 4 個 job 同時連、server 還沒起就搶 Exclusive context → 前幾個 `CUDA-capable device
busy`。**暖機一個 job 讓 server 起來後，後續併發完全正常**（device-plugin 路徑因每節點常駐 MPS
daemon 而 server 常熱，故無此問題）。生產緩解：(a) Slurm prolog 先 ping/warm MPS server；
(b) 讓 MPS server 常駐；(c) 連續負載下 server 自然保持熱。此為運維細節，非阻礙。

**踩到的雷（已解）**：(1) Slurm `--wrap` 用 `/bin/sh`，`$RANDOM` 是空的 → 測試 job 少參數假性
FAILED（與 MPS 無關）。(2) pod 重啟後 slurmd 暫態 `NOT_RESPONDING` → `scontrol reconfigure`
+ `state=resume` 沉澱即可。

**遷移可行性 100% 證實。永久化**：上述 patch 是直接改 StatefulSet（Helm-managed，`helm upgrade`
slurm-platform 會還原）——正式落地需把 DRA claim 寫進 slurm-platform chart 的 worker template。
**回滾**：`kubectl apply -f /tmp/sts-rtx4070-backup.yaml` + 刪 pod。

## 6.2 Stage B：node-2 遷移 + 全叢集純 DRA + 移除 device-plugin（2026-07-08，✅ 完成）

**兩節點皆已遷至 DRA，device-plugin 全數移除。**

1. **DRA driver 擴到 node-2**：`helm upgrade dra-driver-nvidia-gpu`（移除 node-1-only
   `kubeletPlugin.nodeSelector`）→ kubelet-plugin 在兩節點皆 Running；node-2 advertise
   ResourceSlice（gpu-0, Ampere = 3080）。node-2 前置同樣滿足（containerd 2.2.2、k3s 1.34.6、driver 580）。
2. **chart-ify rtx3080**：`values-2x1.yaml` rtx3080 pool `useDra: true` + `draMpsMemLimit: 9Gi`
   （3080=10GB VRAM）。keepalive 保留——實測 host RSS 極小（worker pod 總記憶體 265Mi），node-2 的
   7.5GB 扛得住。`helm upgrade slurm-platform`（REV 64）。
3. **關 node-2 device-plugin MPS**（避免與 DRA 搶 Exclusive context）：`kubectl label node
   nutnadmin... nvidia.com/gpu.deploy.operands=false` → 逐出 gpu-operator device-plugin/mps；
   清 node-2 host 殘留 MPS + `nvidia-smi -c 0`。
4. **重建 rtx3080 pod**：經 DRA 取得 GPU + `/dev/nvidia0` + `/tmp/nvidia-mps`；Slurm 節點
   `idle gpu:rtx3080:1,mps:rtx3080:100`；**4×mps:25 併發多工**（node-2 host nvidia-smi 4 個 gpu_workload）。
5. **移除不必要元件**：
   - 刪除孤兒 `kube-system/nvidia-device-plugin-daemonset`（72 天、無 owner、無 Helm 管理，
     正是雙-plugin 搶 socket 的元兇；無來源 manifest，不會被 deploy 重建）。
   - gpu-operator device-plugin/mps-control-daemon：兩節點 `operands=false` → 不再部署（NFD/dcgm 保留給 DRA 用）。
   - 結果：兩節點 `nvidia.com/gpu` allocatable=0，全部 GPU 存取走 `resource.k8s.io` DRA。

**⚠️ 運維變更 — 遊戲流程**：`scripts/gpu-toggle.sh` 在 DRA 下**已失效**（它切 gpu-operator operands，
但現在 GPU 由 worker pod 的 DRA claim + keepalive 持有）。DRA 下要玩遊戲需：`kubectl scale sts
slurm-worker-gpu-rtx4070 --replicas=0`（釋放 claim → DRA 拆 MPS）→ `nvidia-smi -c 0` → 玩 →
玩畢 `--replicas=1`。（gpu-toggle.sh 待更新以配合 DRA。）

**永久化狀態**：worker DRA 已進 chart（`values-2x1.yaml` + `workers.yaml` + `dra-resourceclaim.yaml`），
跨 `helm upgrade` 存活。待補：(a) gpu-operator release 設 `devicePlugin.enabled=false` 使 operands 停用
永久化（目前靠手動 label）；(b) 更新 `gpu-toggle.sh` 支援 DRA 遊戲流程；(c) `deploy-2.sh` 加裝 DRA driver。

## 參考
- Repo：https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu （v0.4.x，NVIDIA 已捐給 k8s SIG）
- 安裝文件：https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/install/
- GPU Operator DRA 指南：https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html
- MPS spec 範例：`demo/specs/quickstart/v1/gpu-test-mps.yaml`、`demo/specs/mig+mps/`
