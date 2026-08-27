#!/bin/bash
# launch c3=7.0 scan with TSQR on 3090 (scan8, keeps scan7 as reference)
cd /home/user_3090/software/pheasy/MnIn2Se4_c7
source ~/miniconda3/etc/profile.d/conda.sh && conda activate wc || exit 1
tmux kill-session -t scan 2>/dev/null
tmux new-session -d -s scan "bash scan_methods.sh JOBS=3 THREADS=2 C3_CUTOFF=7.0 NDATA_LIST='2 3 4 5 6 8 12 20 45' METHODS='OLS LASSO ALASSO RFE RFE-OLS-TSQR' SCANDIR=scan8 > scan8.out 2>&1"
echo "scan launched in tmux session: scan"
