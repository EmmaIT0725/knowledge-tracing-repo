import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DATASET_DIR = "../datasets/assistments17/preprocessed_df.csv"


class ASSIST2017(Dataset):
    def __init__(self, max_seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx = self.preprocess() # 가장 아래에서 각각의 요소를 가져옴
            # preprocess 메서드를 호출하여 데이터를 전처리하고, 전처리된 데이터를 클래스 속성에 저장

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
        # 인덱스를 입력받아 해당 인덱스의 질문과 답변 시퀀스를 반환
        # DataLoader에서 사용

    def __len__(self):
        return self.len
    # 데이터셋 크기

    # 데이터 전처리 수행
    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1", sep='\t')
        #df = df[(df["correct"] == 0).values + (df["correct"] == 1).values]
        df = df[(df["correct"] == 0) | (df["correct"] == 1)]

        u_list = np.unique(df["user_id"].values) #중복되지 않은 user의 목록
        q_list = np.unique(df["skill_id"].values) #중복되지 않은 question의 목록
        r_list = np.unique(df["correct"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)} #중복되지 않은 user에게 idx를 붙여준 딕셔너리
        q2idx = {q: idx for idx, q in enumerate(q_list)} #중복되지 않은 question에 idx를 붙여준 딕셔너리

        q_seqs = [] #로그 기준으로 각 user별 질문 목록을 담은 리스트
        r_seqs = [] #로그 기준으로 각 user별 정답 목록을 담은 리스트

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values]) # 판다스로 짜는게 좋음
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx

    # 수정할 것
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


# ## print()로 찍어서 값 확인해보기
# # ASSIST2017 인스턴스 생성
# dataset = ASSIST2017(max_seq_len=100, dataset_dir=DATASET_DIR)
# print(dataset)

# # 속성 값 확인 (값이 7개 반환)
# print(dataset.q_seqs[:5])  # 처음 5개의 질문 시퀀스 (각 user별 질문 목록을 담은 리스트)
# print(dataset.q_seqs[0])
# print(len(dataset.q_seqs[0]))   # 99개
# '''
# [339 339 338 338  44  44  44  59  55  55  55 314 314 314 314  34 356 356
#  356  60  60 365 365 155 366 366 366 374  36  36  36  36  36  36  36  36
#   36  20 177 177 177 177 381 381 381 124 363 109 113 363 356  39 356 356
#  196 196 196 194 194 194 262 262 262 262 131 119  72  47  43  47 384 379
#  379 375 375 385 385 385 385 385 377 377 380 384 384 222 384 384 384 384
#  384 384 379 379 384 384 222 222 376]
# '''
# print()
# print(dataset.r_seqs[:5])  # 처음 5개의 답변 시퀀스 (각 user별 정답 목록을 담은 리스트)
# print(dataset.r_seqs[0])
# '''
# [0 1 0 0 1 0 1 0 0 0 1 0 1 0 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 0 1 1 1 1 1 1 1
#  1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
# '''
# print()
# print(dataset.q_list[:5])  # 처음 5개의 고유 질문 ID
# print(dataset.u_list[:5])  # 처음 5개의 고유 사용자 ID
# print(dataset.r_list)      # 고유한 정답 값 (0 또는 1)
# print(dataset.q2idx)       # 질문 ID → 인덱스 매핑
# print(dataset.u2idx)       # 사용자 ID → 인덱스 매핑
# print(dataset.u_list.shape[0])  # 1708