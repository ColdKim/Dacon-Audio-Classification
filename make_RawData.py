from multiprocessing import Process
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import librosa
import warnings

def get_id(data):
    return int(data.split("\\")[1].split(".")[0])

def load_data(paths, name):

    result = []
    if name == "test_npy_1":
        itr = tqdm(paths, desc=name)
    else:
        itr = paths
    for path in itr:
        # sr = 16000이 의미하는 것은 1초당 16000개의 데이터를 샘플링 한다는 것입니다.
        data, sr = librosa.load(path, sr = 16000)
        result.append(data)
    result = np.array(result)
    # 메모리가 부족할 때는 데이터 타입을 변경해 주세요 ex) np.array(data, dtype = np.float32)

    np.save(f"./npy_data/{name}", result)
    return result



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    sample_submission = pd.read_csv("./open/sample_submission.csv")

    africa_train_paths = glob("./open/train/africa/*.wav")
    australia_train_paths = glob("./open/train/australia/*.wav")
    canada_train_paths = glob("./open/train/canada/*.wav")
    england_train_paths = glob("./open/train/england/*.wav")
    hongkong_train_paths = glob("./open/train/hongkong/*.wav")
    us_train_paths = glob("./open/train/us/*.wav")

    path_list = [africa_train_paths, australia_train_paths, canada_train_paths,
                england_train_paths, hongkong_train_paths, us_train_paths]

    test_ = pd.DataFrame(index = range(0, 6100), columns = ["path", "id"])
    test_["path"] = glob("./open/test/*.wav")
    test_["id"] = test_["path"].apply(lambda x : get_id(x))

    # os.mkdir("./npy_data")

    th1 = Process(target = load_data, args=(test_["path"][:2000], "test_npy_1"))
    th2 = Process(target = load_data, args=(test_["path"][2000:4000], "test_npy_2"))
    th3 = Process(target = load_data, args=(test_["path"][4000:], "test_npy_3"))


    th1.start(); th2.start(); th3.start();
    th1.join(); th2.join(); th3.join();
