#!/bin/bash
# =============================================================================
#  pheasy_fit.sh —— 通用力常数拟合脚本 (v3)
# =============================================================================
#  在含 POSCAR / SPOSCAR / disp_matrix.pkl / force_matrix.pkl 的目录里运行：
#      bash pheasy_fit.sh [KEY=VAL ...]
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │ 拟合方法 (FIT_METHOD = -l/--model)                                   │
#  ├─────────────────────────────────────────────────────────────────────┤
#  │  OLS            普通最小二乘。稠密→SVD(gelsd)，稀疏→LSQR，两级→LSMR │
#  │                 (按内存自动分发)。无正则化，基准方法。              │
#  │  LASSO          L1 正则 + 交叉验证选 alpha，默认去偏 (relaxed)。    │
#  │  ALASSO         自适应 LASSO (Zou 2006)：初始岭估计 → 自适应权重 →  │
#  │                 LASSO。比 LASSO 更少收缩偏差。                     │
#  │  RFE            递归特征消除：scale-invariant 重要性 + 分组 CV。    │
#  │  RFE-OLS-TSQR   RFE + 高瘦 QR (Q-less tree TSQR)，超大规模专用。   │
#  │  RIDGE          L2 岭回归 (CV 选 alpha)。                          │
#  └─────────────────────────────────────────────────────────────────────┘
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │ 参数分类 (KEY=VAL)                                                   │
#  ├─────────────────────────────────────────────────────────────────────┤
#  │ 【结构 / 截断】                                                      │
#  │   FIT_ORDER      阶次 2/3/4                       (默认 3)          │
#  │   C2_CUTOFF      二阶截断半径 Å 或负整数最近邻   (默认 None)        │
#  │   C3_CUTOFF      三阶截断半径 Å                  (默认 5.2)         │
#  │   C4_CUTOFF      四阶截断 (FIT_ORDER=4 时必填)  (默认 None)         │
#  │   NULL_SPACE_EPS 零空间构造数值容差              (默认 0.001)       │
#  │ 【数据】                                                             │
#  │   NDATA          拟合用构型数 (空 = 全部)        (默认 空)          │
#  │   FORCE_REBUILD  true 时忽略中间缓存重算         (默认 false)       │
#  │ 【拟合求解】                                                         │
#  │   STANDARDIZE    列标准化 LASSO/ALASSO/RIDGE    (默认 true)         │
#  │   LASSO_TOL      坐标下降容差 (勿回退 1e-3)     (默认 1e-6)         │
#  │   LASSO_MAX_ITER 最大迭代次数                    (默认 20000)       │
#  │ 【正则化 / CV】                                                      │
#  │   CV             CV 折数 (按构型分组)            (默认 5)           │
#  │   NMU            alpha/ridge 网格点数           (默认 20)           │
#  │                  (也决定 ALASSO 默认密度 (nmu-1)/4/decade)          │
#  │   ALPHA_DECADES  alpha_auto 网格跨度(数量级)    (默认 4.0)         │
#  │                  (ALASSO 超定分支有 max(decades,6) 下限)             │
#  │   MU_MIN/MU_MAX  RIDGE 手动网格 10^min..10^max  (默认 -6/-2)       │
#  │ 【内存 / 并行】                                                      │
#  │   SM_DTYPE       sensing matrix 精度 float32/64 (默认 float32)     │
#  │   NCPU           线程数 / joblib worker 数      (默认 8)           │
#  │   LASSO_SPARSE   PHEASY_LASSO_SPARSE=1 走稀疏 SM(默认 0=稠密)      │
#  │   LASSO_TWOLEVEL PHEASY_LASSO_TWOLEVEL=1 两级matvec(连稀疏乘积都不  │
#  │                  物化，仅当稀疏乘积也放不下内存时才划算)            │
#  └─────────────────────────────────────────────────────────────────────┘
#
#  其它常用环境变量：
#    PHEASY_LASSO_DEBIAS=0   关闭 LASSO/ALASSO 去偏
#    PHEASY_RFE_STEP=0.05    RFE 每轮删除比例
#    PHEASY_RFE_MIN_FEATURES RFE 最小保留特征数
#    PHEASY_OLS_TWOLEVEL=0   关闭 OLS 两级 matvec
#    PHEASY_LASSO_1SE=1      LASSO CV 用 1-SE 规则
#    PHEASY_TSQR_CRITERION=  TSQR 判停: cv|bic|aic (默认 cv)
#    PHEASY_BIC_N_EFF=      BIC/AIC 的有效观测: groups|samples (默认 groups=构型数)
#    PHEASY_ALASSO_WEIGHTED_GRID= ALASSO 加权 alpha 网格 (默认 1)。
#                           1=加权 KKT 阈值网格 (推荐, P37/P38/P40):
#                           超定时自动下探到未加权阈值尺度, 密度每 decade
#                           固定 (PHEASY_ALPHA_PER_DECADE); 0=回退 mu_shift
#                           经验网格 (历史可用, 不推荐)。注意: ALASSO 惩罚
#                           在加权空间, 手动网格
#                           (--no-alpha_auto --mu_min/--mu_max) 是已知劣化
#                           路径 -- 未加权尺度的网格会严重过正则化, 例如
#                           MnIn2Se4 c3=5.2 n45 [1e-6,1e-2]: re 0.0063->0.077,
#                           nnz 1223->375 (P38 起手动网格会被原样尊重)
#    PHEASY_ALPHA_PER_DECADE ALASSO 加权网格每 decade 格点数 (默认 (nmu-1)/4)
#    PHEASY_ALPHA_NMAX=       ALASSO 加权网格格点数硬上限 (默认 200)
#    PHEASY_ALASSO_GRID_DIAG= 手动网格尺度假错配诊断 rmatvec (默认 1, 仅
#                              --no-alpha_auto 路径生效)
#
#  示例：
#    bash pheasy_fit.sh FIT_METHOD=OLS    C3_CUTOFF=5.2
#    bash pheasy_fit.sh FIT_METHOD=LASSO  C3_CUTOFF=7.0 NDATA=45 NCPU=8
#    bash pheasy_fit.sh FIT_METHOD=ALASSO C3_CUTOFF=5.2
#    bash pheasy_fit.sh FIT_METHOD=RFE    C3_CUTOFF=5.2
#    bash pheasy_fit.sh FIT_METHOD=RIDGE  C3_CUTOFF=5.2 MU_MIN=-6 MU_MAX=-2
# =============================================================================
set -uo pipefail

# ===== 默认参数 (命令行 KEY=VAL 可覆盖) =====
FIT_METHOD="OLS"
FIT_ORDER=3
C2_CUTOFF=None
C3_CUTOFF=5.2
C4_CUTOFF=None
NULL_SPACE_EPS=0.001
NDATA=""                # 拟合构型数 (空 = 全部)
FORCE_REBUILD=false
STANDARDIZE=true
LASSO_TOL=1e-6
LASSO_MAX_ITER=20000
CV=5
NMU=20
ALPHA_DECADES=4.0
MU_MIN=-6
MU_MAX=-2
SM_DTYPE=float32
NCPU=8
LASSO_SPARSE=0          # PHEASY_LASSO_SPARSE
LASSO_TWOLEVEL=0        # PHEASY_LASSO_TWOLEVEL

_ALLOWED="FIT_METHOD FIT_ORDER C2_CUTOFF C3_CUTOFF C4_CUTOFF NULL_SPACE_EPS \
NDATA FORCE_REBUILD STANDARDIZE LASSO_TOL LASSO_MAX_ITER CV NMU ALPHA_DECADES \
MU_MIN MU_MAX SM_DTYPE NCPU LASSO_SPARSE LASSO_TWOLEVEL"
for kv in "$@"; do
  case "$kv" in
    *=*)
      k="${kv%%=*}"; v="${kv#*=}"
      case " $_ALLOWED " in
        *" $k "*) printf -v "$k" '%s' "$v" ;;
        *) echo "未知参数 $k；可用: $_ALLOWED" >&2; exit 2 ;;
      esac ;;
    *) echo "参数必须是 KEY=VAL 形式: $kv" >&2; exit 2 ;;
  esac
done

case "$FIT_METHOD" in
  OLS|LASSO|ALASSO|RFE|RFE-OLS-TSQR|RIDGE) ;;
  *) echo "FIT_METHOD=$FIT_METHOD 不是合法方法; 可选: OLS LASSO ALASSO RFE RFE-OLS-TSQR RIDGE" >&2; exit 2 ;;
esac
if [ "$FIT_ORDER" -ge 4 ] && { [ "$C4_CUTOFF" = "None" ] || [ "$C4_CUTOFF" = "none" ]; }; then
  echo "FIT_ORDER=4 但没有设 C4_CUTOFF；四阶不截断会让轨道数爆炸。" >&2
  echo "请显式给一个值，例如 C4_CUTOFF=4.0" >&2
  exit 2
fi

for f in POSCAR SPOSCAR disp_matrix.pkl force_matrix.pkl; do
  [ -f "$f" ] || { echo "缺少 $f。disp_matrix.pkl / force_matrix.pkl 用 tools/prepare_dataset.py 生成。" >&2; exit 2; }
done

export OPENBLAS_NUM_THREADS="$NCPU" OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export PHEASY_N_JOBS="$NCPU"
export PHEASY_SM_DTYPE="$SM_DTYPE"
if [ "$LASSO_SPARSE" = "1" ] || [ "$LASSO_SPARSE" = "true" ]; then
  export PHEASY_LASSO_SPARSE=1
fi
if [ "$LASSO_TWOLEVEL" = "1" ] || [ "$LASSO_TWOLEVEL" = "true" ]; then
  export PHEASY_LASSO_TWOLEVEL=1
fi

_STRUCT_FILES="cs.pkl neighbor_list.pkl ns_harm.npz ns_anharm3.npz ns_anharm4.npz"
_DATA_FILES="sm_prime.npz fm1d.npz sm_dense.npy sm_dense.npy.meta.json"

if [ "$FORCE_REBUILD" = "true" ]; then
  rm -f dim_detected.txt $_STRUCT_FILES $_DATA_FILES \
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
[ -z "$NDATA" ] && NDATA="$SM_NDATA"
if [ "$NDATA" -gt "$SM_NDATA" ]; then
  echo "NDATA=$NDATA 超过 disp_matrix.pkl 里的 $SM_NDATA 个构型。" >&2; exit 2
fi

# [FIX P31] CV 折数不能超过构型数（GroupKFold 按构型分组，NDATA=2/3/4 时 5 折
# 会每折只有 0-1 个构型做验证）。程序内虽也有 clamp，这里提前夹住并提示。
if [ "$CV" -gt "$NDATA" ]; then
  CV=$(( NDATA < 2 ? 2 : NDATA ))
  echo "CV 折数超过构型数，已夹到 $CV (NDATA=$NDATA)"
fi

C_FLAG=""
[ "$C2_CUTOFF" != "None" ] && [ "$C2_CUTOFF" != "none" ] && C_FLAG="$C_FLAG --c2 $C2_CUTOFF"
[ "$FIT_ORDER" -ge 3 ] && [ "$C3_CUTOFF" != "None" ] && [ "$C3_CUTOFF" != "none" ] && C_FLAG="$C_FLAG --c3 $C3_CUTOFF"
[ "$FIT_ORDER" -ge 4 ] && [ "$C4_CUTOFF" != "None" ] && [ "$C4_CUTOFF" != "none" ] && C_FLAG="$C_FLAG --c4 $C4_CUTOFF"
W_FLAG="-w $FIT_ORDER"

# pheasy 每次运行都会重写 SPOSCAR（内容不变但 mtime 刷新），所以结构指纹必须
# 用内容哈希而不是 mtime，否则每跑一次都判定"结构变了"并重建 cluster space。
_fp_hash() {
  if   command -v md5sum  >/dev/null 2>&1; then md5sum  "$1" | cut -d' ' -f1
  elif command -v sha1sum >/dev/null 2>&1; then sha1sum "$1" | cut -d' ' -f1
  else cksum "$1" | cut -d' ' -f1,2 | tr ' ' '_'; fi
}

# ---------------- 一级指纹：结构 / cutoff（守 cluster space 与 null space）------
_stamp_struct=$(printf 'dim=%s order=%s c2=%s c3=%s c4=%s eps=%s poscar=%s sposcar=%s' \
  "$DIM" "$FIT_ORDER" "$C2_CUTOFF" "$C3_CUTOFF" "$C4_CUTOFF" "$NULL_SPACE_EPS" \
  "$(_fp_hash POSCAR)" "$(_fp_hash SPOSCAR)")
if [ -f .pheasy_stamp_struct ] && [ "$(cat .pheasy_stamp_struct)" != "$_stamp_struct" ]; then
  echo "结构/截断参数已变化，丢弃 cluster space、null space 与 sensing matrix："
  rm -f $_STRUCT_FILES $_DATA_FILES .pheasy_stamp_data
fi

# ---------------- 二级指纹：数据（守 sensing matrix）---------------------------
_stamp_data=$(printf '%s | dtype=%s sm_ndata=%s disp=%s,%s force=%s,%s' \
  "$_stamp_struct" "$SM_DTYPE" "$SM_NDATA" \
  "$(stat -Lc %s disp_matrix.pkl)"  "$(_fp_hash disp_matrix.pkl)" \
  "$(stat -Lc %s force_matrix.pkl)" "$(_fp_hash force_matrix.pkl)")
if [ -f .pheasy_stamp_data ] && [ "$(cat .pheasy_stamp_data)" != "$_stamp_data" ]; then
  echo "位移/力数据或精度已变化，丢弃 sensing matrix："
  rm -f $_DATA_FILES
fi

echo "拟合: $FIT_METHOD | 阶次 $FIT_ORDER | c2=$C2_CUTOFF c3=$C3_CUTOFF c4=$C4_CUTOFF | DIM=$DIM | ndata=$NDATA/$SM_NDATA | dtype=$SM_DTYPE"

if [ ! -f cs.pkl ]; then
  echo "[1/4] cluster space"
  pheasy --dim $DIM $W_FLAG -s $C_FLAG --eps $NULL_SPACE_EPS || exit 1
else
  echo "[1/4] cluster space 跳过 (cs.pkl)"
fi

if [ ! -f ns_harm.npz ]; then
  echo "[2/4] null space"
  pheasy --dim $DIM $W_FLAG -c $C_FLAG --eps $NULL_SPACE_EPS || exit 1
else
  echo "[2/4] null space 跳过 (ns_harm.npz)"
fi
printf '%s' "$_stamp_struct" > .pheasy_stamp_struct

if [ ! -f sm_prime.npz ]; then
  echo "[3/4] sensing matrix (按全部 $SM_NDATA 个构型建表, 供任意 NDATA<=$SM_NDATA 复用)"
  pheasy --dim $DIM $W_FLAG -d $C_FLAG --ndata $SM_NDATA --disp_file --eps $NULL_SPACE_EPS || exit 1
else
  echo "[3/4] sensing matrix 跳过 (sm_prime.npz, 建表构型数 $SM_NDATA)"
fi
printf '%s' "$_stamp_data" > .pheasy_stamp_data

echo "[4/4] fit ($FIT_METHOD, ndata=$NDATA)"
FIT_FLAGS="--full_ifc -l $FIT_METHOD --hdf5"
# --std 对 LASSO / ALASSO / RIDGE 都生效。RIDGE 尤其需要：列范数跨度可达 1e2，
# 不标准化等于对不同项施加差百倍的 L2 惩罚。
if [ "$STANDARDIZE" = "true" ] && [[ "$FIT_METHOD" =~ ^(LASSO|ALASSO|RIDGE)$ ]]; then
  FIT_FLAGS="$FIT_FLAGS --std"
fi
if [[ "$FIT_METHOD" =~ ^(LASSO|ALASSO)$ ]]; then
  # --tol 0.001 不是"够紧"：sklearn 把 tol 按 ||y||^2 缩放, 小 alpha 端坐标下降
  # 远未收敛就停, CV 曲线在末端压平, argmin 在并列值里挑到最强正则的那个。
  FIT_FLAGS="$FIT_FLAGS --alpha_auto --alpha_decades $ALPHA_DECADES --max_iter $LASSO_MAX_ITER --cv $CV --nmu $NMU --tol $LASSO_TOL"
elif [ "$FIT_METHOD" = "RIDGE" ]; then
  FIT_FLAGS="$FIT_FLAGS --mu_min $MU_MIN --mu_max $MU_MAX --nmu $NMU"
fi
pheasy --dim $DIM $W_FLAG -f $C_FLAG --ndata $NDATA --eps $NULL_SPACE_EPS $FIT_FLAGS || exit 1

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
