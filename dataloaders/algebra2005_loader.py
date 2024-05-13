import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DATASET_DIR = "../datasets/algebra05/preprocessed_df.csv"

# user_id	item_id	  timestamp	  correct	skill_id

class ALGEBRA2005(Dataset):
    def __init__(self, max_seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir

        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.q2idx, \
            self.u2idx = self.preprocess()  
        # 가장 아래에서 각각의 요소를 가져옴
        # def preprocess(self):
        
        self.num_u = self.u_list.shape[0]       # 게임으로 비유하면 특징, 혹은 장착된 도구 등
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]

        # match_seq_len은 경우에 따라 설정하기 -> 사용하려면 parameter에 seq_len을 추가해야 함
        # match_seq_len을 거치면, 모든 데이터는 101개로 통일되고, 빈칸인 부분은 -1로 전처리되어있음
        self.q_seqs, self.r_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, max_seq_len) #아래 method를 한번 거치도록 처리
        
        self.len = len(self.q_seqs)

    def __getitem__(self, index):   # 각각의 구동되는 함수는 특징을 이용해서 움직이는 movement라고 생각하면됨. .move
        # 출력되는 벡터는 모두 101개로 전처리되어있고, 만약 빈칸이 있는 데이터의 경우에는 -1로 채워져있음
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):      # 각각의 구동되는 함수는 특징을 이용해서 움직이는 movement라고 생각하면됨. .move
        return self.len

    def preprocess(self):       
        # 각각의 구동되는 함수는 특징을 이용해서 움직이는 movement라고 생각하면됨. .move
        df = pd.read_csv(self.dataset_dir, sep='\t').sort_values(by=["timestamp"])

#        user_id  item_id  timestamp  correct  skill_id
# 290766      250   159661         27        0        81
# 252884      213   159661         38        1        81
# 343466      308   159661         46        1        81

        u_list = np.unique(df["user_id"].values) # 중복되지 않은 user의 목록
        # print(u_list)
        # .values는 np array 값을 가져오는 것
        q_list = np.unique(df["skill_id"].values) # 중복되지 않은 skill의 목록
        # print(q_list)
        r_list = np.unique(df["correct"].values) # 중복되지 않은 correct의 목록
        # print(r_list): [0, 1]

        u2idx = {u: idx for idx, u in enumerate(u_list)} # 중복되지 않은 user에게 idx를 붙여준 딕셔너리
        q2idx = {q: idx for idx, q in enumerate(q_list)} # 중복되지 않은 question에 idx를 붙여준 딕셔너리
        # print(q2idx)

        q_seqs = [] # 로그 기준으로 각 user별 질문 목록을 담은 리스트
        r_seqs = [] # 로그 기준으로 각 user별 정답 목록을 담은 리스트

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values]) # 판다스로 짜는게 좋음
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx



# 수정할 것
# match_seq_len
    def match_seq_len(self, q_seqs, r_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []

        for q_seq, r_seq in zip(q_seqs, r_seqs):

            # max_seq_len(100)보다 작거나 같은 데이터는 넘기고, 100보다 큰 데이터는 while문을 통과하게 됨
            # while을 통과할 경우, 100개씩 데이터를 떼서 proc에 넣음
            i = 0 # i는 while을 통과할 경우 추가되고, 아니면 추가되지 않음
            while i + max_seq_len < len(q_seq): # 첫반복: 100보다 큰 경우, 두번째 반복: 200보다 큰 경우
                proc_q_seqs.append(q_seq[i:i + max_seq_len - 1])
                proc_r_seqs.append(r_seq[i:i + max_seq_len - 1])

                i += max_seq_len

            #while을 거치지 않은 경우는 바로, while을 거친 경우 남은 데이터에 대해서만 처리하게 됨
            proc_q_seqs.append(
                np.concatenate(
                    [
                        q_seq[i:], # while을 거치지 않았다면, i 변화가 없으므로 처음부터 끝까지, while을 거쳤다면 남은부분만
                        np.array([pad_val] * (i + max_seq_len - len(q_seq))) #총 100개로 만들기, 대신 남은 부분은 -1로 채움
                    ]
                )
            )
            '''
            두 배열을 합치는 과정에서는 np.concatenate 함수가 사용되는데, 
            이 함수는 입력된 배열들을 연결하여 하나의 배열로 만들어 줍니다. 
            따라서 위 코드는 q_seq 배열의 일부분과 부족한 부분을 pad_val 값으로 채운 배열을 합쳐서 
            하나의 배열로 만드는 작업을 수행합니다.
            '''
            proc_r_seqs.append(
                np.concatenate(
                    [
                        r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        return proc_q_seqs, proc_r_seqs
    

'''
## preprocess 함수 이해하기
df = pd.read_csv(DATASET_DIR, sep='\t').sort_values(by=['timestamp'])
print(df)

###
        user_id  item_id  timestamp  correct  skill_id
290766      250   159661         27        0        81
252884      213   159661         38        1        81
343466      308   159661         46        1        81
###

u_list = np.unique(df["user_id"].values) # 중복되지 않은 user의 목록
# print(u_list)
q_list = np.unique(df["skill_id"].values) # 중복되지 않은 skill의 목록
# print(q_list)
r_list = np.unique(df["correct"].values) # 중복되지 않은 correct의 목록
# print(r_list): [0, 1]

u2idx = {u: idx for idx, u in enumerate(u_list)} # 중복되지 않은 user에게 idx를 붙여준 딕셔너리
q2idx = {q: idx for idx, q in enumerate(q_list)} # 중복되지 않은 question에 idx를 붙여준 딕셔너리
# print(q2idx)

q_seqs = [] # 로그 기준으로 각 user별 질문 목록을 담은 리스트
r_seqs = [] # 로그 기준으로 각 user별 정답 목록을 담은 리스트

for u in u_list:
    df_u = df[df["user_id"] == u]

    q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values]) # 판다스로 짜는게 좋음
    r_seq = df_u["correct"].values

    q_seqs.append(q_seq)
    r_seqs.append(r_seq)

# print(df_u)
# print()
# print(q_seq)
# print()
# print(r_seq)
'''

'''
### match_seq_len 이해하기
df = pd.read_csv(DATASET_DIR, sep='\t').sort_values(by=['timestamp'])
# print(df)

###
#         user_id  item_id  timestamp  correct  skill_id
# 290766      250   159661         27        0        81
# 252884      213   159661         38        1        81
# 343466      308   159661         46        1        81
###

u_list = np.unique(df["user_id"].values) # 중복되지 않은 user의 목록
# print(u_list)
q_list = np.unique(df["skill_id"].values) # 중복되지 않은 skill의 목록
# print(q_list)
r_list = np.unique(df["correct"].values) # 중복되지 않은 correct의 목록
# print(r_list): [0, 1]

u2idx = {u: idx for idx, u in enumerate(u_list)} # 중복되지 않은 user에게 idx를 붙여준 딕셔너리
q2idx = {q: idx for idx, q in enumerate(q_list)} # 중복되지 않은 question에 idx를 붙여준 딕셔너리
# print(q2idx)

q_seqs = [] # 로그 기준으로 각 user별 질문 목록을 담은 리스트
r_seqs = [] # 로그 기준으로 각 user별 정답 목록을 담은 리스트

for u in u_list:
    df_u = df[df["user_id"] == u]

    q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values]) # 판다스로 짜는게 좋음
    r_seq = df_u["correct"].values

    q_seqs.append(q_seq)
    r_seqs.append(r_seq)

proc_q_seqs = []
proc_r_seqs = []

for q_seq, r_seq in zip(q_seqs, r_seqs):

    # max_seq_len(100)보다 작거나 같은 데이터는 넘기고, 100보다 큰 데이터는 while문을 통과하게 됨
    # while을 통과할 경우, 100개씩 데이터를 떼서 proc에 넣음
    i = 0 
    # i는 while을 통과할 경우 추가되고, 아니면 추가되지 않음
    # 즉, while을 통과하지 않는 경우 계속 i = 0
    max_seq_len = 100

    while i + max_seq_len < len(q_seq): # 첫 반복: 100보다 큰 경우, 두번째 반복: 200보다 큰 경우
        proc_q_seqs.append(q_seq[i:i + max_seq_len - 1])
        proc_r_seqs.append(r_seq[i:i + max_seq_len - 1])

        i += max_seq_len
    # max_seq_len보다 작은 범위 내에서 q_seq 시퀀스를 순회

    # while 반복문은 주어진 조건이 참(True)인 동안 반복하여 코드 블록을 실행. 
    # 조건이 거짓(False)이 되면 반복문이 종료. 
    # 따라서 while 반복문은 조건을 만족하는 동안 계속해서 반복 작업을 수행.

    ## while 조건:
    # 조건이 참일 때 실행할 코드 블록
    # 반복적으로 실행될 코드
    print(np.array([-1] * (i + max_seq_len - len(q_seq))))
'''