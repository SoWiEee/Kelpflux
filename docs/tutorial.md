# Kelpflux User Tutorial

本文件是給「已經有一套 Kelpflux live cluster」的使用者看的操作教學。部署、安裝、Helm upgrade、GPU Operator 等管理工作請看 README 的 Getting Started；這裡只說明使用者如何從 login node 提交 Slurm jobs、使用 Lmod toolchain、查看狀態與讀取結果。

## 1. 登入與基本觀念

Kelpflux 的使用方式和傳統 HPC 很接近：使用者進入 login node，透過 `sbatch` 提交工作，實際計算會被 Slurm 排到 CPU 或 GPU worker 上。

管理者可用 Kubernetes 進入 login pod：

```bash
kubectl -n slurm exec -it deploy/slurm-login -- bash
```

若已開啟 SSH login，使用者也可以用 SSH 連入：

```bash
ssh -p 30022 root@<cluster-ip>
```

常用指令：

```bash
sinfo -Nel       # 查看 Slurm nodes / partition 狀態
squeue           # 查看 queue
sacct            # 查看歷史 job accounting
scancel <jobid>  # 取消 job
```

目前主要 partition：

| Partition | 用途 | 常見 request |
|-----------|------|--------------|
| `cpu` | CPU / OpenMP / MPI CPU job | `#SBATCH -p cpu` |
| `gpu-rtx4070` | RTX 4070 GPU job | `#SBATCH -p gpu-rtx4070` + `--gres=gpu:1` 或 `--gres=mps:25` |

CPU/GPU worker 由 Kelpflux operator 依 pending jobs 擴縮。也就是說，使用者 submit job 後，對應 worker pod 可能才會被啟動；但 worker image 應在部署階段完成 build/import/pre-pull，不會在 submit 時才 build。

## 2. 共享工作目錄

建議把 job script、source code、輸出檔放在 `/shared`，因為 login node 與 worker pods 都會掛載這個 RWX shared storage。

```bash
mkdir -p /shared/tutorial
cd /shared/tutorial
```

## 3. Lmod Toolchain

Lmod 是切換編譯器、MPI、CUDA 等工具鏈的入口。因為 `sbatch` 是 non-login shell，job script 內需要明確載入 Lmod：

```bash
source /etc/profile.d/lmod.sh
module avail
```

目前內建模組：

| Module | 用途 |
|--------|------|
| `gcc/11` | GNU C/C++/Fortran compiler；OpenMP 使用 `-fopenmp` |
| `openmpi/4.1` | OpenMPI 4.1，設定 `MPI_HOME`、MPI wrapper、`SLURM_MPI_TYPE=pmi2` |
| `python3/3.10` | Ubuntu 22.04 system Python |
| `cuda/12.4` | CUDA toolkit / `nvcc` |

互動式檢查：

```bash
source /etc/profile.d/lmod.sh
module load gcc/11 openmpi/4.1 cuda/12.4
module list
which gcc
which mpicc
which nvcc
module purge
```

## 4. 提交 OpenMP CPU Job

```bash
cat > /shared/tutorial/omp-test.sh << 'EOF'
#!/bin/bash
#SBATCH -J omp-test
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -o /shared/tutorial/omp-%j.out

source /etc/profile.d/lmod.sh
module load gcc/11

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cat > omp_hello.c <<'SRC'
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
  printf("thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
  return 0;
}
SRC

gcc -fopenmp omp_hello.c -o omp_hello
srun ./omp_hello
EOF

sbatch /shared/tutorial/omp-test.sh
```

查看結果：

```bash
squeue
cat /shared/tutorial/omp-<jobid>.out
```

## 5. 提交 MPI Job

單節點兩個 rank 範例：

```bash
cat > /shared/tutorial/mpi-test.sh << 'EOF'
#!/bin/bash
#SBATCH -J mpi-test
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -o /shared/tutorial/mpi-%j.out

source /etc/profile.d/lmod.sh
module load gcc/11 openmpi/4.1

cat > mpi_hello.c <<'SRC'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("rank %d / %d\n", rank, size);
  MPI_Finalize();
  return 0;
}
SRC

mpicc mpi_hello.c -o mpi_hello
srun --mpi=pmi2 ./mpi_hello
EOF

sbatch /shared/tutorial/mpi-test.sh
```

注意：目前 OpenMPI module 會設定 `SLURM_MPI_TYPE=pmi2`，所以 job 內使用 `srun --mpi=pmi2` 是預期路徑。

## 6. 提交 CUDA / nvcc GPU Job

完整使用一張 GPU：

```bash
cat > /shared/tutorial/cuda-test.sh << 'EOF'
#!/bin/bash
#SBATCH -J cuda-test
#SBATCH -p gpu-rtx4070
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o /shared/tutorial/cuda-%j.out

source /etc/profile.d/lmod.sh
module load gcc/11 cuda/12.4

cat > hello_cuda.cu <<'SRC'
#include <cstdio>
__global__ void hello() {
  printf("hello from cuda block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}
int main() {
  hello<<<1, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}
SRC

nvcc hello_cuda.cu -o hello_cuda
srun ./hello_cuda
EOF

sbatch /shared/tutorial/cuda-test.sh
```

如果只需要共享 GPU 的一部分 MPS slot，可以改成：

```bash
#SBATCH -p gpu-rtx4070
#SBATCH --gres=mps:25
```

MPS request 會讓 Slurm 以 MPS GRES 控制排程容量；實際是否立刻執行取決於 GPU worker 是否已啟動、目前 GPU/MPS slot 是否足夠，以及 queue priority。

## 7. Python Job

```bash
cat > /shared/tutorial/python-test.sh << 'EOF'
#!/bin/bash
#SBATCH -J python-test
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -o /shared/tutorial/python-%j.out

source /etc/profile.d/lmod.sh
module load python3/3.10

python3 --version
python3 - <<'SRC'
print('hello from python job')
SRC
EOF

sbatch /shared/tutorial/python-test.sh
```

若需要額外套件，建議把 virtualenv 建在 `/shared`：

```bash
python3 -m venv /shared/venvs/myenv
source /shared/venvs/myenv/bin/activate
pip install <package>
```

job script 內再 `source /shared/venvs/myenv/bin/activate`。

## 8. 查看、取消與除錯

查看 queue：

```bash
squeue
squeue -j <jobid>
```

查看 node / partition：

```bash
sinfo -Nel
scontrol show job <jobid>
```

取消 job：

```bash
scancel <jobid>
```

常見 pending reason：

| Reason | 意義 |
|--------|------|
| `Resources` | 需要的 CPU/GPU/MPS 資源暫時不足，或 worker 正在啟動 |
| `Priority` | 資源可能可用，但目前 queue priority 還沒輪到 |
| `Nodes required for job are DOWN...` | 對應 worker 在 Slurm 裡不是 idle/mix，需要管理者檢查 worker pod/slurmd |

## 9. Grafana 觀測

管理者可以開 Grafana port-forward：

```bash
kubectl -n monitoring port-forward svc/grafana 3000:3000
```

常用 dashboard：

| Dashboard | 用途 |
|-----------|------|
| Bridge Overview | 查看 Slurm queue 如何驅動 K8s worker pool |
| Scheduler Live Resource View | 查看 DSAC decision、snapshot freshness、boost/abstain |
| GPU Utilisation | 查看 DCGM GPU utilization、VRAM、power、temperature |
| Per-Job GPU Profile | 以 job / GPU 維度觀察使用狀態 |

## 10. 查看 OpenTelemetry Trace

一般使用者平常看 `squeue`、job output 和 Grafana dashboard 就夠了；當 job 等待時間異常、DSAC placement 看起來不符合預期、或需要向管理者回報一次完整決策流程時，再查 OTel trace。

在 Grafana Explore 選擇 Tempo datasource，使用 TraceQL 以 job id 查詢：

```text
{ job_id = "<jobid>" }
```

也可以從 **Per-Job GPU Profile** dashboard 的 Tempo 連結進入。若是剛提交的 job，請把查詢時間範圍設成最近 15 分鐘到 1 小時。

一筆完整 trace 通常會呈現：

| Span | 意義 |
|------|------|
| `job_submit` | job 送進系統時，RL scheduler 建立決策 trace |
| `queue_wait` | operator 觀察到 pending job 後接續 trace，代表排隊等待 |
| `job_running` | job 進入 running，表示 Slurm 已完成 placement / allocation |
| `scale_up_decision` / `k8s_provisioning` | 若 worker 需要啟動，會看到 operator 擴容與 pod ready 的耗時 |

實測範例：OpenMP job `135` 在 `slurm-worker-cpu-0` 上執行，Tempo trace id 是 `462ec09dba445df68352c97812568f5a`。這筆 trace 同時包含 `slurm-rl-scheduler/job_submit` 與 `slurm-operator/job_running`，而且 `job_running` 的 parent span 會接到 `job_submit`，代表 submit hook、RL scheduler、operator lifecycle tracing 已串成同一條 trace。

OpenMP 輸出範例：

```text
joined trace openmp thread 0 / 4
joined trace openmp thread 1 / 4
joined trace openmp thread 2 / 4
joined trace openmp thread 3 / 4
```

這類 trace 可以用來回答「job 是否真的進入 Slurm」、「跑在哪個 node」、「operator 什麼時候看到 job running」這類問題。

如果 Tempo 查不到 trace，先確認 job 是透過 Kelpflux login node 的 `sbatch` 提交，並把 `job_id`、提交時間、`scontrol show job <jobid>` 結果提供給管理者。

## 11. 最小檢查清單

使用者回報 job 跑不起來時，先收集：

```bash
squeue
sinfo -Nel
scontrol show job <jobid>
cat /shared/tutorial/*<jobid>*.out
```

如果是 module/toolchain 問題，在 login node 跑：

```bash
source /etc/profile.d/lmod.sh
module avail
module load gcc/11 openmpi/4.1 cuda/12.4
which gcc
which mpicc
which nvcc
```
