

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances_argmin_min
# 生成示例数据




datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
#datasets = ["Yahoo"]
methods = ['AE', 'Donut','EncDecAD', 'LSTMADalpha', 'FCVAE','AR', 'LSTMADbeta', 'Diff']

base_folder_test = '/home/mjp/tsad/122/Results/Scores_test'

base_folder_final = '/home/mjp/tsad/122/Results/Scores'

specific_path = 'one_by_one'

for dataset in datasets:

    directory_names2 = []
    path1 = os.path.join(base_folder_test, methods[0], specific_path, dataset)

    if os.path.exists(path1):
        for file_name in os.listdir(path1):
            if file_name.endswith('.npy'):
                directory_names2.append(file_name)
        
    for method in methods:
        for directory_name2 in directory_names2:
            dataall = np.array([])
            pathfinal1 = os.path.join(base_folder_test, method, specific_path, dataset, directory_name2)
            pathfinal2 = os.path.join(base_folder_final, method, specific_path, dataset, directory_name2)
            
            arr = np.load(pathfinal1)   
            arr = np.nan_to_num(arr, nan=0)
            arr_min = min(arr)
            arr_max = max(arr)
            arr_avg = (arr_min+arr_max)/2

            arr_down = (arr_min+arr_avg)/2
            arr_up = (arr_max+arr_avg)/2
            # 计算相邻点的差分
            diffs = np.diff(arr)

            a = diffs/arr[:-1]

            a = np.insert(a, 0, 0)

            ratio_min = 0
            ratio_max = max(a)
            ratio_avg = (ratio_min+ratio_max)/2

            ratio_down = (ratio_min+ratio_avg)/2
            ratio_up = (ratio_max+ratio_avg)/2

            dataall = np.zeros_like(arr)

            for i in range(len(arr)):
                score = 0
                if arr[i]>arr_up:
                    score = score + 4
                elif arr[i]>arr_avg:
                    score = score + 3
                elif arr[i]>arr_down:
                    score = score + 2
                else:
                    score = score + 1
                
                if a[i]>ratio_up:
                    score = score + 4
                elif a[i]>ratio_avg:
                    score = score + 3
                elif a[i]>ratio_down:
                    score = score + 2
                else:
                    score = score + 1

                if score >= 5:
                    dataall[i] = 1
                

            print(pathfinal2)
            os.makedirs(os.path.dirname(pathfinal2), exist_ok=True)
            np.save(pathfinal2, dataall)
        