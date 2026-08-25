#!/bin/bash
# =============================================================================
#  pheasy_fit.sh —— 精简版 pheasy 力常数拟合脚本 (v2, 两级指纹)
# =============================================================================
#  只保留必要参数：截断半径 / 拟合方法 / 阶次 / 核数。
#  已被程序接管、无需再写的东西：
#    - alpha 网格自动推导 + 撞界重试    -> 程序内 --alpha_auto (默认开)
#    - 方法重定向 PHEASY_USE_*         -> -l 直接支持 OLS/LASSO/ALASSO/RFE/RFE-OLS-TSQR/RIDGE
#    - CV 构型分组 PHEASY_CV_GROUP_SIZE -> 程序内自动 = 3 × 超胞原子数
#    - LASSO 去偏                       -> 程序内默认开 (PHEASY_LASSO_DEBIAS=0 关闭)
#
#  依赖：POSCAR SPOSCAR disp_matrix.pkl force_matrix.pkl
#        后两个由 tools/prepare_dataset.py 从 npy 生成，例如：
#            python3 tools/prepare_dataset.py SPOSCAR \
#                    dataset_disps.npy dataset_forces.npy --frac
#
#  用法： bash pheasy_fit.sh FIT_METHOD=LASSO C3_CUTOFF=5.2 NCPU=8
#         bash pheasy_fit.sh FIT_METHOD=LASSO NDATA=8      # 只用前 8 个构型
#
#  两级指纹（v2 新增）
#  -------------------
#    .pheasy_stamp_struct  DIM / 阶次 / cutoff / eps / POSCAR+SPOSCAR 内容哈希
#         -> 守 cs.pkl, neighbor_list.pkl, ns_*.npz
#    .pheasy_stamp_data    struct + dtype + disp/force 内容哈希 + 建表构型数
#         -> 守 sm_prime.npz
#  指纹一律用内容哈希，不用 mtime —— pheasy 每次运行都会重写 SPOSCAR，用 mtime
#  会让每一次运行都自我失效、无谓重建。
#  sensing matrix 永远按**可用的最大构型数**建一次；拟合时用 --ndata N 取前 N 个
#  构型（run_pheasy 内部按 3*natoms*NDATA 切片 SM_prime，力矩阵同样只取前 N 个）。
#  所以扫描 NDATA=4,6,8,...,45 只需要建一次 sensing matrix。
# =============================================================================
set -uo pipefail

# ===== 必要参数（命令行 KEY=VAL 可覆盖）=====
FIT_METHOD="LASSO"
FIT_ORDER=3
C2_CUTOFF=None
C3_CUTOFF=5.2
C4_CUTOFF=None
NULL_SPACE_EPS=0.001
STANDARDIZE=true
LASSO_TOL=1e-6          # 见下方 FIT_FLAGS 处的说明，不要回到 1e-3
LASSO_MAX_ITER=20000
NMU=20                  # alpha / ridge 网格点数
MU_MIN=-6               # 仅 RIDGE 用（LASSO/ALASSO 走 --alpha_auto 自动推导）
MU_MAX=-2
SM_DTYPE=float32        # float32 省一半内存; float64 精度更高。必须全局一致。
NDATA=""                # 拟合用的构型数（留空 = 全部）
NCPU=8
FORCE_REBUILD=false     # true 时忽略所有中间文件重算

_ALLOWED="FIT_METHOD FIT_ORDER C2_CUTOFF C3_CUTOFF C4_CUTOFF NULL_SPACE_EPS \
STANDARDIZE LASSO_TOL LASSO_MAX_ITER NMU MU_MIN MU_MAX SM_DTYPE NDATA NCPU FORCE_REBUILD"
for kv in "$@"; do
  case "$kv" in
    *=*)
      k="${kv%%=*}"; v="${kv#*=}"
      case " ${_ALLOWED} " in
        *" ${k} "*) printf -v "$k" '%s' "$v" ;;
        *) echo "未知参数 ${k}；可用: ${_ALLOWED}" >&2; exit 2 ;;
      esac ;;
    *) echo "参数必须是 KEY=VAL 形式: ${kv}" >&2; exit 2 ;;
  esac
done

case "${FIT_METHOD}" in
  OLS|LASSO|ALASSO|RFE|RFE-OLS-TSQR|RIDGE) ;;
  *) echo "FIT_METHOD=${FIT_METHOD} 不是合法方法; 可选: OLS LASSO ALASSO RFE RFE-OLS-TSQR RIDGE" >&2; exit 2 ;;
esac
if [ "${FIT_ORDER}" -ge 4 ] && { [ "${C4_CUTOFF}" = "None" ] || [ "${C4_CUTOFF}" = "none" ]; }; then
  echo "FIT_ORDER=4 但没有设 C4_CUTOFF；四阶不截断会让轨道数爆炸。" >&2
  echo "请显式给一个值，例如 C4_CUTOFF=4.0" >&2
  exit 2
fi

for f in POSCAR SPOSCAR disp_matrix.pkl force_matrix.pkl; do
  [ -f "$f" ] || { echo "缺少 ${f}。disp_matrix.pkl / force_matrix.pkl 用 tools/prepare_dataset.py 生成。" >&2; exit 2; }
done

export OPENBLAS_NUM_THREADS="${NCPU}" OMP_NUM_THREADS="${NCPU}" MKL_NUM_THREADS="${NCPU}"
export PHEASY_N_JOBS="${NCPU}"
export PHEASY_SM_DTYPE="${SM_DTYPE}"

_STRUCT_FILES="cs.pkl neighbor_list.pkl ns_harm.npz ns_anharm3.npz ns_anharm4.npz"
_DATA_FILES="sm_prime.npz fm1d.npz sm_dense.npy sm_dense.npy.meta.json"

if [ "${FORCE_REBUILD}" = "true" ]; then
  rm -f dim_detected.txt ${_STRUCT_FILES} ${_DATA_FILES} \
        .pheasy_stamp_struct .pheasy_stamp_data .pheasy_stamp
fi

if [ ! -f dim_detected.txt ]; then
  python3 - <<'PY' || exit 1
import numpy as np
from phonopy.interface.vasp import read_vasp
uc = read_vasp('POSCAR'); sc = read_vasp('SPOSCAR')
R = sc.cell @ np.linalg.inv(uc.cell)
M = np.round(R).astype(int)
if not np.allclose(R, M, atol=1e-4):
    raise SystemExit("SPOSCAR 晶格不是 POSCAR 的整数倍组合")
if not np.allclose(M, np.diag(np.diag(M))):
    raise SystemExit("生成矩阵非对角；pheasy --dim 只接受对角胞")
dim = np.diag(M)
open('dim_detected.txt', 'w').write(' '.join(map(str, dim)))
print("DIM =", ' '.join(map(str, dim)))
PY
fi
DIM=$(cat dim_detected.txt)

# 可用构型数 = disp_matrix.pkl 里的帧数（sensing matrix 按这个建满）
SM_NDATA=$(python3 -c "
import pickle
print(pickle.load(open('disp_matrix.pkl','rb')).shape[0])") || exit 1
[ -z "${NDATA}" ] && NDATA="${SM_NDATA}"
if [ "${NDATA}" -gt "${SM_NDATA}" ]; then
  echo "NDATA=${NDATA} 超过 disp_matrix.pkl 里的 ${SM_NDATA} 个构型。" >&2; exit 2
fi

C_FLAG=""
[ "${C2_CUTOFF}" != "None" ] && [ "${C2_CUTOFF}" != "none" ] && C_FLAG="${C_FLAG} --c2 ${C2_CUTOFF}"
[ "${FIT_ORDER}" -ge 3 ] && [ "${C3_CUTOFF}" != "None" ] && [ "${C3_CUTOFF}" != "none" ] && C_FLAG="${C_FLAG} --c3 ${C3_CUTOFF}"
[ "${FIT_ORDER}" -ge 4 ] && [ "${C4_CUTOFF}" != "None" ] && [ "${C4_CUTOFF}" != "none" ] && C_FLAG="${C_FLAG} --c4 ${C4_CUTOFF}"
W_FLAG="-w ${FIT_ORDER}"

# pheasy 每次运行都会重写 SPOSCAR（内容不变但 mtime 刷新），所以结构指纹必须
# 用内容哈希而不是 mtime，否则每跑一次都判定"结构变了"并重建 cluster space。
_fp_hash() {
  if   command -v md5sum  >/dev/null 2>&1; then md5sum  "$1" | cut -d' ' -f1
  elif command -v sha1sum >/dev/null 2>&1; then sha1sum "$1" | cut -d' ' -f1
  else cksum "$1" | cut -d' ' -f1,2 | tr ' ' '_'; fi
}

# ---------------- 一级指纹：结构 / cutoff（守 cluster space 与 null space）------
_stamp_struct=$(printf 'dim=%s order=%s c2=%s c3=%s c4=%s eps=%s poscar=%s sposcar=%s' \
  "${DIM}" "${FIT_ORDER}" "${C2_CUTOFF}" "${C3_CUTOFF}" "${C4_CUTOFF}" "${NULL_SPACE_EPS}" \
  "$(_fp_hash POSCAR)" "$(_fp_hash SPOSCAR)")
if [ -f .pheasy_stamp_struct ] && [ "$(cat .pheasy_stamp_struct)" != "${_stamp_struct}" ]; then
  echo "结构/截断参数已变化，丢弃 cluster space、null space 与 sensing matrix："
  echo "  旧: $(cat .pheasy_stamp_struct)"
  echo "  新: ${_stamp_struct}"
  rm -f ${_STRUCT_FILES} ${_DATA_FILES} .pheasy_stamp_data
fi

# ---------------- 二级指纹：数据（守 sensing matrix）---------------------------
# 注意不含 NDATA：sensing matrix 按 SM_NDATA 建满，拟合时切片即可复用。
_stamp_data=$(printf '%s | dtype=%s sm_ndata=%s disp=%s,%s force=%s,%s' \
  "${_stamp_struct}" "${SM_DTYPE}" "${SM_NDATA}" \
  "$(stat -Lc %s disp_matrix.pkl)"  "$(_fp_hash disp_matrix.pkl)" \
  "$(stat -Lc %s force_matrix.pkl)" "$(_fp_hash force_matrix.pkl)")
if [ -f .pheasy_stamp_data ] && [ "$(cat .pheasy_stamp_data)" != "${_stamp_data}" ]; then
  echo "位移/力数据或精度已变化，丢弃 sensing matrix："
  echo "  旧: $(cat .pheasy_stamp_data)"
  echo "  新: ${_stamp_data}"
  rm -f ${_DATA_FILES}
fi

echo "拟合: ${FIT_METHOD} | 阶次 ${FIT_ORDER} | c2=${C2_CUTOFF} c3=${C3_CUTOFF} c4=${C4_CUTOFF} | DIM=${DIM} | ndata=${NDATA}/${SM_NDATA} | dtype=${SM_DTYPE}"

if [ ! -f cs.pkl ]; then
  echo "[1/4] cluster space"
  pheasy --dim ${DIM} ${W_FLAG} -s ${C_FLAG} --eps ${NULL_SPACE_EPS} || exit 1
else
  echo "[1/4] cluster space 跳过 (cs.pkl)"
fi

if [ ! -f ns_harm.npz ]; then
  echo "[2/4] null space"
  pheasy --dim ${DIM} ${W_FLAG} -c ${C_FLAG} --eps ${NULL_SPACE_EPS} || exit 1
else
  echo "[2/4] null space 跳过 (ns_harm.npz)"
fi
printf '%s' "${_stamp_struct}" > .pheasy_stamp_struct

if [ ! -f sm_prime.npz ]; then
  echo "[3/4] sensing matrix (按全部 ${SM_NDATA} 个构型建表, 供任意 NDATA<=${SM_NDATA} 复用)"
  pheasy --dim ${DIM} ${W_FLAG} -d ${C_FLAG} --ndata ${SM_NDATA} --disp_file --eps ${NULL_SPACE_EPS} || exit 1
else
  echo "[3/4] sensing matrix 跳过 (sm_prime.npz, 建表构型数 ${SM_NDATA})"
fi
printf '%s' "${_stamp_data}" > .pheasy_stamp_data

echo "[4/4] fit (${FIT_METHOD}, ndata=${NDATA})"
FIT_FLAGS="--full_ifc -l ${FIT_METHOD} --hdf5"
# --std 对 LASSO / ALASSO / RIDGE 都生效 (optimizer.py: method in LASSO/ALASSO/RIDGE)。
# RIDGE 尤其需要：列范数跨度可达 1e2，不标准化等于对不同项施加差百倍的 L2 惩罚。
if [ "${STANDARDIZE}" = "true" ] && [[ "${FIT_METHOD}" =~ ^(LASSO|ALASSO|RIDGE)$ ]]; then
  FIT_FLAGS="${FIT_FLAGS} --std"
fi
if [[ "${FIT_METHOD}" =~ ^(LASSO|ALASSO)$ ]]; then
  # --tol 0.001 不是"够紧"：sklearn 把 tol 按 ||y||^2 缩放, 小 alpha 端坐标下降
  # 远未收敛就停, CV 曲线在末端压平, argmin 在并列值里挑到最强正则的那个 ->
  # 力常数偏低约 10%。1e-6 只多花几十秒。
  FIT_FLAGS="${FIT_FLAGS} --alpha_auto --max_iter ${LASSO_MAX_ITER} --cv 5 --nmu ${NMU} --tol ${LASSO_TOL}"
elif [ "${FIT_METHOD}" = "RIDGE" ]; then
  FIT_FLAGS="${FIT_FLAGS} --mu_min ${MU_MIN} --mu_max ${MU_MAX} --nmu ${NMU}"
fi
pheasy --dim ${DIM} ${W_FLAG} -f ${C_FLAG} --ndata ${NDATA} --eps ${NULL_SPACE_EPS} ${FIT_FLAGS} || exit 1

echo "完成"
python3 -c "
import h5py, numpy as np
for fn, keys in (('fc2.hdf5', ('fc2', 'force_constants')),
                 ('fc3.hdf5', ('fc3', 'force_constants_third')),
                 ('fc4.hdf5', ('fc4', 'force_constants_fourth'))):
    try:
        with h5py.File(fn, 'r') as f:
            k = next(k for k in keys if k in f)
            print('%-9s max = %.4f' % (k, float(np.max(np.abs(np.asarray(f[k]))))))
    except (OSError, StopIteration):
        pass
" || exit 1