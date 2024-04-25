import sys

sys_list = ['/home/kjw/seminar/CorePatchSampling_0409', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/kjw/seminar/PatchCoreTesting/lib/python3.8/site-packages']
for i in sys_list:
    sys.path.append(i)

sys.path.append('/usr/lib/python3/dist-packages/click')
# sys.path.append('/home/kjw/seminar/PatchCoreTesting/lib/python3.8/site-packages')
sys.path.append('/home/kjw/seminar/CorePatchSampling_0409/PatchCore')
# sys.path.append('/home/kjw/seminar/CorePatchSampling_0409/PatchCore/sampler.py')
# sys.path.append('/home/kjw/seminar/CorePatchSampling_0409/PatchCore/utils.py')
print(sys.path)

import logging
import os
import pickle

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import random

from PatchCore import patchcore
from PatchCore import backbones
from PatchCore import common
from PatchCore import sampler
from PatchCore import utils

# 기본 파이썬 라이브러리
from collections import defaultdict

# 데이터 처리 및 모델 학습을 위한 파이토치 관련 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 모델 직렬화를 위한 dill 라이브러리
import dill as pickle

# 이미지 처리를 위한 PIL과 torchvision
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms as vision_transforms

# PyTorch Lightning을 사용한 효율적인 모델 학습
import pytorch_lightning as pl

# 인터랙티브 인터페이스 제작을 위한 Gradio 라이브러리
import gradio as gr

from copy import deepcopy

from glob import glob
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser(description="patchcore")
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--backbone', default='wideresnet101', type=str)
parser.add_argument('--layers_to_extract_from', nargs='+', default=['layer2','layer3'], type=str)
parser.add_argument('--pretrain_embed_dimension', default=2304, type=int)
parser.add_argument('--target_embed_dimension', default=1024+512, type=int)
parser.add_argument('--patchsize', default=3, type=int)
parser.add_argument('--anomaly_scorer_num_nn', default=5, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--cv', default=5, type=int)
parser.add_argument('--seed', default=826, type=int)
parser.add_argument('--device', nargs='+', default=[0], type=int)
parser.add_argument('--num_workers', default=0, type=int)
args = parser.parse_args('')

image_size = args.image_size
BATCH_SIZE = args.batch_size
CV = args.cv
SEED = args.seed

def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(SEED)

set_seeds()
print("!")


# 탐색할 폴더 경로
directory_path = "/home/kjw/seminar/data"

# 해당 경로 내의 모든 폴더명을 리스트로 만듦
folder_names = [name for name in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, name))]

folder_names.sort()
folder_names = folder_names[0:1]
print(folder_names)




from collections import defaultdict
import random

# random 모듈의 시드 고정
random.seed(42)

# numpy 랜덤 샘플링의 시드 고정
np.random.seed(42)

# 각 원단에서 뽑을 샘플 데이터의 갯수
train_data_num = 20


X = defaultdict(list)
y = defaultdict(list)
X_test = defaultdict(list)

for fabric in folder_names:
    print(fabric)
    temp_X = []
    temp_y = []
    fabric_path = ("/home/kjw/seminar/data/" + fabric + "/")
    for image_path in tqdm(glob(fabric_path + "train/*")):
        temp_X.append(image_path)
        temp_y.append(0)

    # 전체 인덱스 집합 생성
    total_indices_set = set(range(len(temp_X)))

    # 랜덤으로 선택된 인덱스 집합 생성
    print(len(temp_X))
    random_indices = random.sample(range(len(temp_X)), train_data_num)
    random_indices_set = set(random_indices)

    # 랜덤 인덱스에 포함되지 않은 나머지 인덱스 집합
    rest_indices_set = total_indices_set - random_indices_set

    # 선택된 인덱스에 해당하는 데이터 추출
    print("학습 :", sorted(list(random_indices_set)))
    temp_X = np.array(temp_X)[sorted(list(random_indices_set))]
    temp_y = np.array(temp_y)[sorted(list(random_indices_set))]


    X[fabric].extend(temp_X)
    y[fabric].extend(temp_y)


    test_directory_path = fabric_path + "test"
    anomaly_types = [name for name in os.listdir(test_directory_path)
                if os.path.isdir(os.path.join(test_directory_path, name))]

    anomaly_types.sort()
    print(anomaly_types)

    for anomaly_type in anomaly_types:
      temp = []
      for image_path in tqdm(glob(test_directory_path+"/"+anomaly_type+"/*")):
          temp.append(image_path)
      temp.sort()
      if anomaly_type == "n 정상":
          print("테스트_정상 :", sorted(list(rest_indices_set)))
          temp = np.array(temp)[sorted(list(rest_indices_set))]
      else:
          temp = np.array(temp)
      X_test[(fabric, anomaly_type)].extend(temp)


    print()
    print()


# X_test = np.array(X_test)

print(len(X))
total_num = 0
num_dict = defaultdict(int)
for key in sorted(X_test.keys()):
    print(key, len(X_test[key]))
    total_num += len(X_test[key])
    num_dict[key[0]] +=  len(X_test[key])

print("전체 데이터 수 :", total_num)
for key in num_dict.keys():
    print(key, num_dict[key])

    # print(X)
    # print(X_test)
    
    
class TransistorDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X = Image.open(X)

        if self.transform is not None:
            X = self.transform(X)

        if self.y is None:
            return {"image": X}

        y = self.y[idx]
        return {"image": X, "label": y}
    
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(round(image_size*1.143), round(image_size*1.143))),
    v2.CenterCrop(size=(image_size, image_size)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(round(image_size*1.143), round(image_size*1.143))),
    v2.CenterCrop(size=(image_size, image_size)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(round(image_size*1.143), round(image_size*1.143))),
    v2.CenterCrop(size=(image_size, image_size)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


patchcore_pickle_list = []
print(len(patchcore_pickle_list))





import time
from copy import deepcopy
import dill as pickle
from sklearn.model_selection import KFold

time_dict = defaultdict(list)

threshold_dict = defaultdict(list)
scores_dict = defaultdict(list)

for fabric in folder_names:
    print(fabric)
    temp_threshold = []
    print(fabric, len(X[fabric]))

    kf = KFold(n_splits=CV, shuffle=True, random_state=SEED)

    for i, (train_index, val_index) in enumerate(kf.split(X[fabric])):
        X_train = np.array(X[fabric])[train_index]
        y_train = np.array(y[fabric])[train_index]
        X_val = np.array(X[fabric])[val_index]
        y_val = np.array(y[fabric])[val_index]

        train_dataset = TransistorDataset(X_train, y_train, train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)

        val_dataset = TransistorDataset(X_val, y_val, val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

        device = utils.set_torch_device(gpu_ids=args.device)

        patch_core = patchcore.PatchCore(device)

        patch_core.load(
            backbone                 = backbones.load(args.backbone),
            layers_to_extract_from   = args.layers_to_extract_from,
            device                   = device,
            input_shape              = (3, image_size, image_size),
            pretrain_embed_dimension = args.pretrain_embed_dimension,
            target_embed_dimension   = args.target_embed_dimension,
            patchsize                = args.patchsize,
            anomaly_scorer_num_nn    = args.anomaly_scorer_num_nn,
            featuresampler           = sampler.GreedyCoresetSampler(percentage=0.1, device=device),
            nn_method                = common.FaissNN(on_gpu=False, num_workers=args.num_workers) # True로 바꿔볼까
        )

        patch_core.fit(train_dataloader)

        patchcore_pickle_list.append(patch_core)
        with open('/home/kjw/seminar/result/'+str(fabric)+str((i+1))+'.pkl', 'wb') as file:
            pickle.dump(patch_core, file)

        scores, _ = patch_core.predict(
            val_dataloader
        )

        threshold = np.max(scores) # val_dataloader의 수 만큼 scores 가 나오는데 여기서의 최대값.
        print(f"threshold: {threshold}")
        threshold_dict[fabric].append(threshold)
        temp_threshold.append(threshold)

    print(fabric, temp_threshold)
    with open('/home/kjw/seminar/result/' + str(fabric) +'_layer2_threshold_list.pkl', 'wb') as file:
        pickle.dump(temp_threshold, file)


        # 불량 타입
        for anomaly in anomaly_types:
            if anomaly != 'g 오염및손상' and anomaly != 'n 정상':
                continue
            print(f"testing: {fabric}, {anomaly}")
            test_dataset = TransistorDataset(X_test[(fabric, anomaly)], None, test_transform)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
            start_time = time.time()
            scores, _ = patch_core.predict(
                test_dataloader
            )
            end_time = time.time()
            print(f"total time: {end_time - start_time}, 이미지 갯수 : {len(scores)}")
            if len(scores) != 0:
                print(f"time per image: {(end_time - start_time)/len(scores)}\n")
                time_dict[(fabric, anomaly)].append((end_time - start_time)/len(scores))
            else:
                print()

            scores_dict[(fabric, anomaly)].append(scores)
            
            
            
            
            
            
for fabric in folder_names:
    print(fabric, len(X[fabric]))
    print(f"threshold: {threshold_dict[fabric]}")
    threshold = np.mean(threshold_dict[fabric])
    print(f"threshold: {threshold}")

    for anomaly in  anomaly_types:
        if anomaly != 'g 오염및손상' and anomaly != 'n 정상':
            continue
        if anomaly == 'n 정상': # 정상 타입인 경우
            print(f"정상 타입 : {anomaly}")
            scores = np.max(scores_dict[(fabric, anomaly)], axis=0)
            print(f"scores {anomaly}: \n{scores}")
            prediction = np.where(scores<threshold, 0, 1)
            # 논문에서는 임계값 초과인 경우 이상으로 분류하였음,
            n_anomaly = np.sum(prediction == 1)
            print(f"n_anomaly: {n_anomaly}")
            print(f"Accuracy: {1 - (n_anomaly/len(prediction))}")
            print()
            print()

        else:
            print(f"불량 타입 {anomaly}")
            scores = np.max(scores_dict[(fabric, anomaly)], axis=0)
            print(f"scores {anomaly}: \n{scores}")
            prediction = np.where(scores<threshold, 0, 1)
            n_anomaly = np.sum(prediction == 1)
            print(f"n_anomaly: {n_anomaly}")
            print(f"Accuracy: {n_anomaly/len(prediction)}")
            print()
            print()

    print("-"* 30)
    print()
    print()