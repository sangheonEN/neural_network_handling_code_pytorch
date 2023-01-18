from pathlib import Path
import os
import requests

def data_load():

    # 파일 경로 설정.
    DATA_PATH = Path("data")
    PATH = DATA_PATH / 'mnist'
    PATH.mkdir(parents=True, exist_ok=True)

    # mnist download url.
    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    # requests를 활용한 git url 접근하여 데이터셋 다운
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME, verify=False).content
        (PATH / FILENAME).open('wb').write(content)

    # data 직렬화를 위한 python pickle 사용
    import pickle
    import gzip

    with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f: # PurePath.as_posix() 슬래시(/)가 있는 경로의 문자열 표현을 반환합니다
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding='latin-1')

    # data visualization
    import cv2
    import numpy as np

    cv2.imshow('sample_vis', x_train[0].reshape((28, 28)))
    cv2.waitKey()
    cv2.destroyAllWindows()

    # torch Tensor transform, check data class
    import torch
    print(f"class_number : {np.unique(y_train)}") # check class
    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)) # numpy to torch tensor

    return x_train, y_train, x_valid, y_valid, x_test, y_test