#!/bin/bash
# =============================================================================
#  pheasy_fit.sh —— 精简版 pheasy 力常数拟合脚本
# =============================================================================
#  只保留必要参数：截断半径 / 拟合方法 / 阶次 / 核数。
#  已被程序接管、无需再写的东西：
#    - alpha 网格自动推导 + 撞界重试    -> 程序内 --alpha_auto (默认开)
#    - 方法重定向 PHEASY_USE_*         -> -l 直接支持 OLS/LASSO/ALASSO/RFE/RFE-OLS-TSQR/RIDGE
#    - CV 构型分组 PHEASY_CV_GROUP_SIZE -> 程序内自动 = 3 × 超胞原子数
#  依赖：POSCAR SPOSCAR dataset_forces.npy dataset_disps.npy
#        force_matrix.pkl disp_matrix.pkl（数据处理步生成）
#  用法： bash pheasy_fit.sh FIT_METHOD=LASSO C3_CUTOFF=5.2 NCPU=8
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
NDATA=""
NCPU=8

for kv in "$@"; do
  case "$kv" in
    *=*) k="${kv%%=*}"; v="${kv#*=}"; eval "$k=\"\$v\"" ;;
  esac
done

export OPENBLAS_NUM_THREADS="${NCPU}" OMP_NUM_THREADS="${NCPU}" MKL_NUM_THREADS="${NCPU}"
export PHEASY_N_JOBS="${NCPU}"

if [ ! -f dim_detected.txt ]; then
  python3 - <<'PY'
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

[ -z "${NDATA}" ] && [ -f ndata_total.txt ] && NDATA=$(cat ndata_total.txt)
[ -z "${NDATA}" ] && NDATA=20

C_FLAG=""
[ "${C2_CUTOFF}" != "None" ] && [ "${C2_CUTOFF}" != "none" ] && C_FLAG="${C_FLAG} --c2 ${C2_CUTOFF}"
[ "${FIT_ORDER}" -ge 3 ] && [ "${C3_CUTOFF}" != "None" ] && [ "${C3_CUTOFF}" != "none" ] && C_FLAG="${C_FLAG} --c3 ${C3_CUTOFF}"
[ "${FIT_ORDER}" -ge 4 ] && [ "${C4_CUTOFF}" != "None" ] && [ "${C4_CUTOFF}" != "none" ] && C_FLAG="${C_FLAG} --c4 ${C4_CUTOFF}"
W_FLAG="-w ${FIT_ORDER}"

echo "拟合: ${FIT_METHOD} | 阶次 ${FIT_ORDER} | c2=${C2_CUTOFF} c3=${C3_CUTOFF} c4=${C4_CUTOFF} | DIM=${DIM} | ndata=${NDATA}"

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

if [ ! -f sm_prime.npz ]; then
  echo "[3/4] sensing matrix"
  pheasy --dim ${DIM} ${W_FLAG} -d ${C_FLAG} --ndata ${NDATA} --disp_file --eps ${NULL_SPACE_EPS} || exit 1
else
  echo "[3/4] sensing matrix 跳过 (sm_prime.npz)"
fi

echo "[4/4] fit (${FIT_METHOD})"
FIT_FLAGS="--full_ifc -l ${FIT_METHOD} --hdf5"
if [ "${STANDARDIZE}" = "true" ] && [[ "${FIT_METHOD}" =~ ^(LASSO|ALASSO)$ ]]; then
  FIT_FLAGS="${FIT_FLAGS} --std"
fi
if [[ "${FIT_METHOD}" =~ ^(LASSO|ALASSO)$ ]]; then
  FIT_FLAGS="${FIT_FLAGS} --alpha_auto --max_iter 3000 --cv 5 --nmu 20 --tol 0.001"
fi
pheasy --dim ${DIM} ${W_FLAG} -f ${C_FLAG} --ndata ${NDATA} --eps ${NULL_SPACE_EPS} ${FIT_FLAGS} || exit 1

echo "完成"
if [ -f fc3.hdf5 ]; then
  python3 -c "
import h5py, numpy as np
with h5py.File('fc3.hdf5','r') as f:
    k = 'fc3' if 'fc3' in f else 'force_constants_third'
    print('fc3 max = %.4f eV/A^3' % float(np.max(np.abs(np.asarray(f[k])))))
"
fi
