# pheasy 测试材料

本目录存放力常数拟合的测试材料，按材料分门别类。

## 目录结构

    test/
    ├── MnIn2Se4/            # 测试材料：MnIn2Se4（原胞 + 3×3×3 超胞）
    │   ├── POSCAR           # 原胞结构
    │   ├── SPOSCAR          # 超胞结构 (3×3×3, 189 原子)
    │   ├── pcell.symops     # 原胞对称操作
    │   ├── scell.symops     # 超胞对称操作
    │   ├── disp_matrix.pkl  # 位移矩阵（45 个构型）
    │   ├── force_matrix.pkl # 力矩阵
    │   └── dataset_*.npy    # 原始位移/力数据（用 tools/prepare_dataset.py 可从 npy 重新生成 pkl）
    └── results/             # c3=7.0 扫描的收敛结果（convergence.csv/png）

## 怎么跑

脚本 pheasy_fit.sh 在项目根目录，直接 cd 到材料目录运行：

    cd test/MnIn2Se4
    bash ../../pheasy_fit.sh FIT_METHOD=OLS    C3_CUTOFF=5.2
    bash ../../pheasy_fit.sh FIT_METHOD=LASSO  C3_CUTOFF=5.2 NDATA=45 NCPU=8
    bash ../../pheasy_fit.sh FIT_METHOD=ALASSO C3_CUTOFF=5.2
    bash ../../pheasy_fit.sh FIT_METHOD=RFE    C3_CUTOFF=5.2
    bash ../../pheasy_fit.sh FIT_METHOD=RIDGE  C3_CUTOFF=5.2

## 材料说明

- MnIn2Se4：原胞，超胞 3×3×3（189 原子），45 个随机位移构型。
  - 二阶截断 C3_CUTOFF=5.2 Å（原来的默认，对应 MnIn2Se4_lasso）。
  - 三阶截断 C3_CUTOFF=7.0 Å 时自由 IFC 数从 1303 涨到 3678（对应 MnIn2Se4_c7 扫描）。
  - 同一份位移/力数据，只改 cutoff 即可复现两种扫描。
