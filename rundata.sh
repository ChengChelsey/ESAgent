#!/bin/bash
# ==========================================
# 在 ESAGENT 根目录执行，调度 EasyTSAD/run.py
# 动态队列方式跑 4 数据集 × 6 方法（24 个任务）
# 每次最多 4 个并发，每个 GPU 0/1/2/3 轮流分配
# ==========================================

# 确保日志目录存在
mkdir -p logs

# 1) 生成任务列表
python3 - <<'PY' > jobs.txt
datasets = ["AIOPS","TODS","UCR","WSD"]
methods  = ["AE","Donut","EncDecAD","LSTMADalpha","LSTMADbeta","FCVAE"]
for d in datasets:
    for m in methods:
        print(d, m)
PY

# 2) 分配 GPU 号 (0,1,2,3 循环)
awk '{print NR-1}' jobs.txt | awk '{print $1%4}' > gpus.txt
paste -d' ' gpus.txt jobs.txt > jobs_gpu.txt

# 3) 执行 (注意 run.py 的路径)
cat jobs_gpu.txt | parallel -j4 --colsep ' ' '
CUDA_VISIBLE_DEVICES={1} OMP_NUM_THREADS=4 \
python3 EasyTSAD/Examples/run_cli.py --dataset {2} --method {3} --dirname ../datasets --schema naive \
> logs/{2}_{3}.log 2>&1
'
