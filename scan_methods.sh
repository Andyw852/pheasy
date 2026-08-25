#!/bin/bash
# =============================================================================
#  scan_methods.sh —— NDATA × 方法 扫描驱动（并行版）
# =============================================================================
#  为什么需要它：
#    串行跑 50 次 pheasy_fit.sh 时，每次都要重付
#      - RFE 的 SM_prime·NS 稀疏乘 15-19 s（这条路径没有缓存）
#      - fc2/fc3 的 hdf5 写出 10-11 s（189^3×27 ≈ 1.8e8 个数，约 1.5 GB）
#    而 8 个核大部分时间闲着。这里把每个 (方法, NDATA) 组合放进独立子目录，
#    共享只读的中间文件（软链接），然后 J 个任务并行、每个用 T 线程。
#
#  用法：
#    bash scan_methods.sh                       # 默认全扫
#    bash scan_methods.sh JOBS=4 THREADS=2
#    bash scan_methods.sh NDATA_LIST="4 8 20 45" METHODS="OLS LASSO"
#
#  结果：scan/results/phi_<方法>_n<N>.npz  +  scan/results/summary.tsv
#        用 analyze_scan.py 出表和图。
#
#  注意：比较用的是 phi.npz（3132 维的完整 IFC 向量 = NS_full·coef），
#        不是 fc3.hdf5。前者约 30 KB，后者约 1.5 GB，而且 fc3.hdf5 只是
#        phi 按轨道展开的结果，信息完全等价。
# =============================================================================
set -uo pipefail

NDATA_LIST="4 5 6 8 10 12 16 20 30 45"
METHODS="OLS LASSO ALASSO RFE RFE-OLS-TSQR"
JOBS=4                  # 并行任务数
THREADS=2               # 每个任务的 BLAS 线程数（JOBS × THREADS ≈ 物理核数）
SM_DTYPE=float32
C3_CUTOFF=5.2
FIT_ORDER=3
NULL_SPACE_EPS=0.001
SCANDIR=scan

_ALLOWED="NDATA_LIST METHODS JOBS THREADS SM_DTYPE C3_CUTOFF FIT_ORDER NULL_SPACE_EPS SCANDIR"
for kv in "$@"; do
  case "$kv" in
    *=*) k="${kv%%=*}"; v="${kv#*=}"
         case " ${_ALLOWED} " in
           *" ${k} "*) printf -v "$k" '%s' "$v" ;;
           *) echo "未知参数 ${k}；可用: ${_ALLOWED}" >&2; exit 2 ;;
         esac ;;
    *) echo "参数必须是 KEY=VAL 形式: ${kv}" >&2; exit 2 ;;
  esac
done

for f in POSCAR SPOSCAR disp_matrix.pkl force_matrix.pkl pheasy_fit.sh; do
  [ -f "$f" ] || { echo "缺少 ${f}（请在算例目录下运行）" >&2; exit 2; }
done

export PHEASY_SM_DTYPE="${SM_DTYPE}"

# ---- 1. 先把共享的中间文件建好（cluster space / null space / sensing matrix）----
echo "=== 预热：建立共享中间文件 ==="
bash pheasy_fit.sh FIT_METHOD=OLS C3_CUTOFF="${C3_CUTOFF}" FIT_ORDER="${FIT_ORDER}" \
     NULL_SPACE_EPS="${NULL_SPACE_EPS}" SM_DTYPE="${SM_DTYPE}" \
     NCPU="$(( JOBS * THREADS ))" > "${SCANDIR}_warmup.log" 2>&1 || {
  echo "预热失败，见 ${SCANDIR}_warmup.log" >&2; exit 1; }
echo "预热完成。"

SHARED="cs.pkl neighbor_list.pkl ns_harm.npz ns_anharm3.npz ns_anharm4.npz \
sm_prime.npz disp_matrix.pkl force_matrix.pkl dim_detected.txt \
pcell.symops scell.symops .pheasy_stamp_struct .pheasy_stamp_data"

mkdir -p "${SCANDIR}/results"
ROOT=$(pwd)

run_one () {
  local m="$1" n="$2"
  local d="${ROOT}/${SCANDIR}/${m}_n${n}"
  rm -rf "$d"; mkdir -p "$d"
  # 只读的大文件用软链接；POSCAR/SPOSCAR 必须复制 —— pheasy 每次运行都会重写 SPOSCAR
  local f
  for f in ${SHARED}; do
    [ -e "${ROOT}/$f" ] && ln -sf "${ROOT}/$f" "$d/$f"
  done
  cp "${ROOT}/POSCAR" "${ROOT}/SPOSCAR" "${ROOT}/pheasy_fit.sh" "$d/"
  cd "$d" || return 1
  # RFE 的稀疏路径每次都要重算 SM_prime·NS（无缓存）；这里的 SM 100% 稠密，
  # 走 dense 缓存反而更快也更省内存。
  export PHEASY_RFE_SPARSE=0
  local t0=$(date +%s)
  bash pheasy_fit.sh FIT_METHOD="$m" NDATA="$n" NCPU="${THREADS}" \
       C3_CUTOFF="${C3_CUTOFF}" FIT_ORDER="${FIT_ORDER}" \
       NULL_SPACE_EPS="${NULL_SPACE_EPS}" SM_DTYPE="${SM_DTYPE}" \
       > fit.log 2>&1
  local rc=$? t1=$(date +%s)
  if [ $rc -eq 0 ] && [ -f phi.npz ]; then
    cp phi.npz "${ROOT}/${SCANDIR}/results/phi_${m}_n${n}.npz"
    cp fit.log "${ROOT}/${SCANDIR}/results/fit_${m}_n${n}.log"
    # 大文件用完即删，避免 50 份 1.5 GB 的 fc3.hdf5 撑爆磁盘
    rm -f fc3.hdf5 fc2.hdf5 fc4.hdf5 sm_dense.npy sm_dense.npy.meta.json \
          FORCE_CONSTANTS FORCE_CONSTANTS_3RD
  fi
  printf '%-14s n=%-3s rc=%s %ss\n' "$m" "$n" "$rc" "$((t1-t0))"
}
export -f run_one
export ROOT SCANDIR SHARED THREADS C3_CUTOFF FIT_ORDER NULL_SPACE_EPS SM_DTYPE

echo "=== 扫描：$(echo ${METHODS} | wc -w) 方法 × $(echo ${NDATA_LIST} | wc -w) 个 NDATA，${JOBS} 路并行 × ${THREADS} 线程 ==="
# 大 NDATA 排前面：先启动最慢的任务，尾部收敛更快
for n in $(echo ${NDATA_LIST} | tr ' ' '\n' | sort -rn); do
  for m in ${METHODS}; do
    echo "$m $n"
  done
done | xargs -P "${JOBS}" -n 2 bash -c 'run_one "$0" "$1"'

echo
echo "=== 完成，结果在 ${SCANDIR}/results/ ==="
ls "${SCANDIR}/results/" | grep -c '^phi_' | xargs echo "phi 文件数:"
echo "下一步： python3 analyze_scan.py ${SCANDIR}/results"
