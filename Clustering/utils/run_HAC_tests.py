import os
import numpy as np
from glob import glob
from tqdm import tqdm

def test_clustering(labels, labels2):
    mapping_dict = {}
    for label, label2 in zip(labels, labels2):
        if label not in mapping_dict:
            mapping_dict[label] = label2
        else:
            if label2 != mapping_dict[label]:
                return False
    return True      

passed = 0
failed = 0
for yf in tqdm(glob("datasets/HAC/*.y.csv")):
    test_id = os.path.basename(yf).split(".")[0]
    Xf = "datasets/HAC/%s.X.csv"%test_id
    n_clusters, linkage = os.path.basename(yf).split(".")[1].split("_")
    os.system(f"python3 main_HAC.py --predictions-file pred.txt --test-data {Xf} --number-of-clusters {n_clusters} --linkage {linkage}")

    y_pred = np.loadtxt(yf, delimiter=",").astype("int")
    y_actual = np.loadtxt("pred.txt", delimiter=",").astype("int")

    if test_clustering(y_pred, y_actual):
        passed += 1
    else:
        failed += 1


print("%s out of %s test cases passed."%(passed, passed+failed))

