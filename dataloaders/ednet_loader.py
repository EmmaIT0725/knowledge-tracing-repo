import numpy as np
import pandas as pd
import os
import pickle
from torch.utils.data import Dataset

DATASET_DIR = "../datasets/ednet/preprocessed_df.csv"
PICKLE_DIR = "../datasets/ednet/"

'''
"Pickle"은 파이썬에서 객체를 직렬화(serialize)하거나 역직렬화(deserialize)하는 데 사용되는 내장 모듈. 
객체를 직렬화하는 것은 해당 객체의 메모리 상태를 보존하고 파일 또는 네트워크를 통해 전송 가능한 형태로 변환하는 과정. 
Pickle 모듈은 이러한 과정을 수행하는 데 사용된다.

직렬화된 객체는 나중에 다시 역직렬화하여 원래의 객체로 복원가능. 
이는 객체를 파일에 저장하고 나중에 읽어 들이거나, 
네트워크를 통해 객체를 전송하고 다른 프로세스나 컴퓨터에서 객체를 재사용할 때 유용.
직렬화된 데이터는 텍스트나 바이너리 형태로 저장. 이는 데이터를 읽기 어렵게 만들어 무단으로 접근하는 것을 방지.

다음은 Pickle 모듈의 주요 기능과 사용법에 대한 간단한 설명:

    직렬화(Serialization): Pickle 모듈을 사용하여 파이썬 객체를 직렬화할 수 있음. 
    이는 pickle.dump() 함수를 사용하여 객체를 직렬화하고 파일에 저장.

    역직렬화(Deserialization): 저장된 직렬화된 객체를 다시 복원하기 위해 
    pickle.load() 함수를 사용하여 파일에서 객체를 읽어 들일 수 있음.

    파일 I/O 지원: Pickle은 객체를 파일로 저장하고 파일에서 객체를 읽어들이는 데 사용됨. 
    따라서 객체를 보관하고 나중에 사용가능.

    다양한 데이터 타입 지원: Pickle은 파이썬에서 사용되는 대부분의 데이터 타입을 직렬화. 
    이는 리스트, 딕셔너리, 클래스 인스턴스 등 모든 종류의 객체를 직렬화하고 역직렬화할 수 있다는 의미.

    보안 주의: Pickle을 사용할 때 주의해야 할 점은, Pickle은 보안에 취약. 
    특히, 신뢰할 수 없는 소스에서 Pickle 파일을 열거나 직렬화된 데이터를 로드할 때 보안 문제 발생가능.

Pickle 모듈은 데이터를 효율적으로 저장하고 전송하는 데 유용한 도구로서 널리 사용됨. 
그러나 보안 및 호환성과 같은 고려 사항을 고려하여 사용해야함.
'''

# EDNET(Dataset) 함수 의미 이해하기
# dataset_dir = DATASET_DIR
# pickle_dir = PICKLE_DIR

# if os.path.exists(os.path.join(pickle_dir, "q_seqs.pkl")):
#     with open(os.path.join(pickle_dir, "q_seqs.pkl"), "rb") as f:
#        q_seqs = pickle.load(f)

class EDNET(Dataset):
    def __init__(self, max_seq_len, dataset_dir=DATASET_DIR, pickle_dir=PICKLE_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.pickle_dir = pickle_dir
        
        if os.path.exists(os.path.join(self.pickle_dir, "q_seqs.pkl")):
            with open(os.path.join(self.pickle_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)        # 역직렬화(저장된 직렬화된 객체 다시 복원)
            with open(os.path.join(self.pickle_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "r_list.pkl"), "rb") as f:
                self.r_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
                self.u2idx = self.preprocess() #가장 아래에서 각각의 요소를 가져옴

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]

        # match_seq_len은 경우에 따라 설정하기 -> 사용하려면 parameter에 seq_len을 추가해야 함
        # match_seq_len을 거치면, 모든 데이터는 101개로 통일되고, 빈칸인 부분은 -1로 전처리되어있음
        self.q_seqs, self.r_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, max_seq_len) #아래 method를 한번 거치도록 처리

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        #출력되는 벡터는 모두 101개로 전처리되어있고, 만약 빈칸이 있는 데이터의 경우에는 -1로 채워져있음
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir)
        df = df[(df["answered_correctly"] == 0) | (df["answered_correctly"] == 1)]

        u_list = np.unique(df["user_id"].values) #중복되지 않은 user의 목록
        q_list = np.unique(df["content_id"].values) #중복되지 않은 question의 목록
        r_list = np.unique(df["answered_correctly"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)} #중복되지 않은 user에게 idx를 붙여준 딕셔너리
        q2idx = {q: idx for idx, q in enumerate(q_list)} #중복되지 않은 question에 idx를 붙여준 딕셔너리

        q_seqs = [] #로그 기준으로 각 user별 질문 목록을 담은 리스트
        r_seqs = [] #로그 기준으로 각 user별 정답 목록을 담은 리스트

        for u in u_list:
            df_u = df[df["user_id"] == u].sort_values("timestamp")

            q_seq = np.array([q2idx[q] for q in df_u["content_id"].values]) # 판다스로 짜는게 좋음
            r_seq = df_u["answered_correctly"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        # pickle로 만들기 - 직렬화하기
        with open(os.path.join(self.pickle_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.pickle_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.pickle_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.pickle_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.pickle_dir, "r_list.pkl"), "wb") as f:
            pickle.dump(r_list, f)
        with open(os.path.join(self.pickle_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.pickle_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx

    #수정할 것
    def match_seq_len(self, q_seqs, r_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []

        for q_seq, r_seq in zip(q_seqs, r_seqs):

            #max_seq_len(100)보다 작거나 같은 데이터는 넘기고, 100보다 큰 데이터는 while문을 통과하게 됨
            #while을 통과할 경우, 100개씩 데이터를 떼서 proc에 넣음
            i = 0 #i는 while을 통과할 경우 추가되고, 아니면 추가되지 않음
            while i + max_seq_len < len(q_seq): # 첫반복: 100보다 큰 경우, 두번째 반복: 200보다 큰 경우
                proc_q_seqs.append(q_seq[i:i + max_seq_len - 1])
                proc_r_seqs.append(r_seq[i:i + max_seq_len - 1])

                i += max_seq_len

            #while을 거치지 않은 경우는 바로, while을 거친 경우 남은 데이터에 대해서만 처리하게 됨
            proc_q_seqs.append(
                np.concatenate(
                    [
                        q_seq[i:], #while을 거치지 않았다면, 처음부터 끝까지, while을 거쳤다면 남은부분만
                        np.array([pad_val] * (i + max_seq_len - len(q_seq))) #총 100개로 만들기, 대신 남은 부분은 -1로 채움
                    ]
                )
            )
            proc_r_seqs.append(
                np.concatenate(
                    [
                        r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        return proc_q_seqs, proc_r_seqs
