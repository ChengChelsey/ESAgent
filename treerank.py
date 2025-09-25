

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import rankdata
#from isotree import IsolationForest
# 生成示例数据

def minmax(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if(min_val == max_val):
        return arr
    arr_minmax = (arr - min_val) / (max_val - min_val)
    return arr_minmax
def z_score(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def diff(arr):
    diffs = np.diff(arr)

    a = np.insert(diffs, 0, 0)
    a = np.abs(a)
    return a

def find10(arr):
    # arr1 = sorted(arr, reverse=True)
    # a = int(len(arr) / 5)
    # x = arr1[a]

    average = np.mean(arr)
    std_dev = np.std(arr)

    #b = min(arr)

    for i in range(len(arr)):
        if arr[i] <= average + 3*std_dev:
            arr[i] = 0

    return arr
def find100(arr):
    # arr1 = sorted(arr, reverse=True)
    # a = int(len(arr) / 5)
    # x = arr1[a]

    average = np.mean(arr)
    std_dev = np.std(arr)

    #b = min(arr)

    for i in range(len(arr)):
        if arr[i] <= average + 5*std_dev:
            arr[i] = 0
    return arr

def softmax(x):
    # 防止数值溢出，减去最大值
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


datasets = ["TODS", "UCR", "AIOPS", "WSD"]
datasets = ["WSD"]
method = "zhijiejia"

base_folder = '/home/mjp/tsad/122/Results/Scores_train'
base_folder_test = '/home/mjp/tsad/122/Results/Scores'
base_folder_final = '/home/mjp/tsad/122/Results/Scores'
directory_names1 = ['AE', 'Donut', 'EncDecAD', 'LSTMADalpha', 'LSTMADbeta']
directory_names1 = ['AE', 'Donut','EncDecAD', 'LSTMADalpha', 'FCVAE','AR', 'DAMP', 'Diff']
directory_names1 = ['AE', 'Donut','EncDecAD', 'LSTMADalpha', 'FCVAE', 'LSTMADbeta','Diff']
directory_names1 = ['AE', 'Donut', 'LSTMADalpha', 'FCVAE', 'LSTMADbeta','Diff', 'EncDecAD', 'SubLOF']
directory_names1 = ['AE', 'Donut', 'LSTMADalpha', 'FCVAE', 'LSTMADbeta','Diff', 'EncDecAD', 'SubLOF','TimesNet']
directory_names1 = ['AE', 'Donut', 'LSTMADalpha', 'FCVAE', 'LSTMADbeta','Diff']
directory_names1 = [ 'AE', 'Donut', 'LSTMADalpha', 'FCVAE', 'LSTMADbeta', 'EncDecAD']

xishu  = [1,1.2,1.1,1.3,1.5,1.4,1.25]

acscsc = [0.6206,0.6722,0.5708,0.6651,0.7670,0.6530]
#directory_names1 = ['LSTMADalpha']

specific_path = 'one_by_one'

for dataset in datasets:

    directory_names2 = []


    path1 = os.path.join(base_folder, directory_names1[0], specific_path, dataset)

    if os.path.exists(path1):
        for file_name in os.listdir(path1):
            if file_name.endswith('.npy'):
                directory_names2.append(file_name)


    #print(directory_names2)
                

    for directory_name2 in directory_names2:
        key = 0
        dataall = np.array([])
        pathfinal2 = os.path.join(base_folder_final, method, specific_path, dataset, directory_name2)
        for directory_name1 in directory_names1:
            pathfinal1 = os.path.join(base_folder, directory_name1, specific_path, dataset, directory_name2)
            pathfinal1_test = os.path.join(base_folder_test, directory_name1, specific_path, dataset, directory_name2)
            if directory_name1 == directory_names1[0]:
                # arr1 = np.load(pathfinal1)
                # arr1 = np.nan_to_num(arr1, nan=0)

                arr1_test = np.load(pathfinal1_test)
                arr1_test = np.nan_to_num(arr1_test, nan=0)
                # arr1 = z_score(arr1)
                # arr1_test = z_score(arr1_test)
                # arr1 = diff(arr1)
                # arr1_test = diff(arr1_test)
            if directory_name1 == directory_names1[1]:
                # arr2 = np.load(pathfinal1)
                # arr2 = np.nan_to_num(arr2, nan=0)
                                
                arr2_test = np.load(pathfinal1_test)
                arr2_test = np.nan_to_num(arr2_test, nan=0)
                # arr2 = z_score(arr2)
                # arr2_test = z_score(arr2_test)
                # arr2 = diff(arr2)
                # arr2_test = diff(arr2_test)
            if directory_name1 == directory_names1[2]:
                # arr3 = np.load(pathfinal1)  
                # arr3 = np.nan_to_num(arr3, nan=0)
                                
                arr3_test = np.load(pathfinal1_test)
                arr3_test = np.nan_to_num(arr3_test, nan=0)
                # arr3 = z_score(arr3)
                # arr3_test = z_score(arr3_test)
                # arr3 = diff(arr3)
                # arr3_test = diff(arr3_test)

 
            if directory_name1 == directory_names1[3]:
                # arr4 = np.load(pathfinal1)
                # arr4 = np.nan_to_num(arr4, nan=0)
                                
                arr4_test = np.load(pathfinal1_test)
                arr4_test = np.nan_to_num(arr4_test, nan=0)
                # arr4 = z_score(arr4)
                # arr4_test = z_score(arr4_test)
                # arr4 = diff(arr4)
                # arr4_test = diff(arr4_test)
            if directory_name1 == directory_names1[4]:
                # arr5 = np.load(pathfinal1)   
                # arr5 = np.nan_to_num(arr5, nan=0)
                                
                arr5_test = np.load(pathfinal1_test)
                arr5_test = np.nan_to_num(arr5_test, nan=0)
                # arr5 = z_score(arr5)
                # arr5_test = z_score(arr5_test)
                # arr5 = diff(arr5)
                # arr5_test = diff(arr5_test)
            if directory_name1 == directory_names1[5]:
                # arr6 = np.load(pathfinal1)
                # arr6 = np.nan_to_num(arr6, nan=0)
                                            
                arr6_test = np.load(pathfinal1_test)
                arr6_test = np.nan_to_num(arr6_test, nan=0)
                # arr6 = z_score(arr6)
                # arr6_test = z_score(arr6_test)
                # arr6 = diff(arr6)
                # arr6_test = diff(arr6_test)




        # min_len = min(len(arr1), len(arr2), len(arr3), len(arr4), len(arr5), len(arr6), len(arr7))

        # arrays = [arr1, arr2, arr3, arr4, arr5, arr6, arr7]

        # for i in range(len(arrays)):
        #     arrays[i] = arrays[i][len(arrays[i]) - min_len:]
        #     #arrays[i] = z_score(arrays[i])
            
        #     #arrays[i] = diff(arrays[i])
        #     #arrays[i] = minmax(arrays[i])
        #     #arrays[i] = find10(arrays[i])
        
        # arr1, arr2, arr3, arr4, arr5, arr6, arr7 = arrays
        arrays_test = [arr1_test, arr2_test, arr3_test, arr4_test, arr5_test, arr6_test]

        min_len_test = min(len(array) for array in arrays_test)

        flag = 1
        for i in range(len(arrays_test)):
            arrays_test[i] = arrays_test[i][len(arrays_test[i]) - min_len_test:]
            #arrays_test[i] = minmax(arrays_test[i])
            if flag == 1:
                dataall = np.zeros_like(arrays_test[i])
                flag = 0
            
            #score1 = minmax(arrays_test[i])
            # #score1 = z_score(arrays_test[i])
            #score1 = minmax(arrays_test[i])
            #score1 = np.sqrt(score1)
            #score1 = np.array(score1) ** 2
            # #score1 = z_score(arrays_test[i])
            # 对数组应用 softmax
            #softmax_arr = softmax(arrays_test[i])
            
            rank1 = rankdata(-arrays_test[i])
            #rank1 = np.sqrt(rank1)
            #rank1 = np.cbrt(rank1)
            # #population_std_dev = np.std(score1)
            #exp_arr = np.exp(arrays_test[i])
           # score1 = 1/exp_arr
           # rank1 = np.power(rank1, 3)
            score1 = 1 / rank1
            # # score1 = 1 / np.array(rank1) ** 2
            epsilon = 1e-8  # 极小值，防止除以零
            A = arrays_test[i]
            A_safe = np.where(A == 0, epsilon, A)

            B = score1 /  A_safe


            sub_arr = B[132:143]

            print("第33到第43个元素：", sub_arr)


        # 如果minmax是一个函数，则进行minmax处理
        
        # arr1_test, arr2_test, arr3_test, arr4_test, arr5_test, arr6_test = arrays_test
        # dataall = np.zeros_like(arr1_test)
        # X = np.stack((arr1_test, arr2_test, arr3_test, arr4_test, arr5_test, arr6_test)).T
        # #X = dataall.reshape(-1, 1)

        # # X = np.stack((arr1, arr2, arr3, arr4, arr5, arr6, arr7)).T

        # aaaa = len(dataall)//256
        # clf = IsolationForest(random_state=0, max_samples=256, n_estimators = max(100,2*(aaaa+1)))
        # clf.fit(X)
        # #aaa = clf.predict(X_test)

        # dataall = clf.decision_function(X) - 0.5
        # dataall = - dataall
        #dataall = find100(dataall)

        #aaa = np.where(aaa == 1, 0, aaa)  # 将1替换为0
        #aaa = np.where(aaa == -1, 1, aaa)  # 将-1替换为1

                
        #dataall = aaa

        # while(1):
        #     a = 1
        print(pathfinal2)
        os.makedirs(os.path.dirname(pathfinal2), exist_ok=True)
        np.save(pathfinal2, dataall)


