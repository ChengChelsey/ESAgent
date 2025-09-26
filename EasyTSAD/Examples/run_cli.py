# run.py
import argparse
from EasyTSAD.Controller import TSADController
import importlib

METHOD_ALIASES = ["AE", "Donut", "EncDecAD", "LSTMADalpha", "LSTMADbeta", "FCVAE"]
for name in METHOD_ALIASES:
    importlib.import_module(f"EasyTSAD.Methods.{name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="单个数据集名，如 AIOPS/TODS/UCR/WSD")
    parser.add_argument("--method",  type=str, help="单个方法名，如 AE/Donut/EncDecAD/LSTMADalpha/LSTMADbeta/FCVAE")
    parser.add_argument("--dirname", type=str, default="../datasets", help="并列结构用 ../datasets")
    parser.add_argument("--schema",  type=str, default="naive")
    parser.add_argument("--no-eval", action="store_true", help="只产出 scores，不做评估")
    parser.add_argument("--plots",   action="store_true", help="是否绘图")
    args = parser.parse_args()

    # 默认列表（用于本地单机调试或不带参数时）
    datasets = [args.dataset] if args.dataset else ["AIOPS", "TODS", "UCR", "WSD"]
    methods  = [args.method ] if args.method  else ["AE", "Donut", "EncDecAD", "LSTMADalpha", "LSTMADbeta", "FCVAE"]

    gctrl = TSADController()
    gctrl.set_dataset(dataset_type="UTS", dirname=args.dirname, datasets=datasets)

    # 训练/推理（保存 scores 到 Results/Scores/<Method>/one_by_one/<DATASET>/<Series>.npy）
    for method in methods:
        print(f"[RUN] dataset={datasets} method={method} schema={args.schema}")
        gctrl.run_exps(method=method, training_schema=args.schema)

    # 评估（可关）
    if not args.no_eval:
        from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
        gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
        for method in methods:
            gctrl.do_evals(method=method, training_schema=args.schema)

    # 绘图（可选）
    if args.plots:
        for method in methods:
            gctrl.plots(method=method, training_schema=args.schema)

if __name__ == "__main__":
    main()
