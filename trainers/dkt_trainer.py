# For utils.py

# 다른 폴더에 있는 모듈 가져오기
import sys
sys.path.append('/Users/mj/Deep_Knowledge_Tracing_Baseline')
    
# 다른 폴더의 경로를 시스템 경로에 추가

# 다른 폴더에 있는 모듈에서 필요한 클래스 또는 함수 가져오기
# from other_module import SomeClass, some_function

import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint

from utils import EarlyStopping

class DKT_trainer():        # DKT_trainer 클래스 정의 : 이 클래스는 DKT 모델을 학습하고 평가하기 위한 Trainer를 구현
    def __init__(           # 클래스의 초기화 메서드
        self,
        model,              # 학습할 모델
        optimizer,          # 사용할 옵티마이저
        n_epochs,           # 학습할 에폭 수
        device,             # 학습할 디바이스 (CPU 또는 GPU)
        num_q,              # 유일한 질문의 개수
        crit,               # 손실 함수 (binary_cross_entropy 또는 rmse)
        max_seq_len,        # 최대 시퀀스 길이
        grad_acc=False,     # 그래디언트 누적 여부
        grad_acc_iter=4     # 그래디언트 누적을 위한 반복 횟수
        ):

        # 입력된 매개변수들을 객체의 속성으로 설정
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit
        self.max_seq_len = max_seq_len
        self.grad_acc=grad_acc
        self.grad_acc_iter=grad_acc_iter
    
    # train 함수 정의 
    # 학습용 데이터로 모델을 학습하는 내부 메서드
    def _train(self, train_loader, metric_name):
            # train_loader: 학습 데이터 로더
            # metric_name: 평가 지표 이름 (AUC 또는 RMSE)
        
        # AUC 점수, 실제 값(y_trues), 예측 값(y_scores), 손실 값(loss_list)을 저장하기 위한 변수를 초기화
        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []


#### RNN 같이 공부하면서 이해하기
        
        for idx, data in enumerate(tqdm(train_loader)):
            # 학습 데이터로더를 순회하면서 데이터 가져오기

            # 모델을 학습 모드로 전환(Pytorch)
            self.model.train()

            # 데이터를 풀어서 가져오기. (질문 시퀀스, 정답 시퀀스, 이동된 질문 시퀀스, 이동된 정답 시퀀스, 마스크 시퀀스)
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
            
            # 데이터를 디바이스로 옮기기
            q_seqs = q_seqs.to(self.device) #|q_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            r_seqs = r_seqs.to(self.device) #|r_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            qshft_seqs = qshft_seqs.to(self.device) #|qshft_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            rshft_seqs = rshft_seqs.to(self.device) #|rshft_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            mask_seqs = mask_seqs.to(self.device) #|mask_seqs| = (bs, sq) -> [[True,  True,  True,  ..., False, False, False], [True,  True,  True,  ..., False, False, False]..]

            # 모델에 입력 데이터를 전달하여 예측을 수행
            y_hat = self.model( q_seqs.long(), r_seqs.long() ) #|y_hat| = (bs, sq, self.num_q) -> tensor([[[0.6938, 0.7605, ..., 0.7821], [0.8366, 0.6598,  ..., 0.8514],..)
            #=> 각 sq별로 문항의 확률값들이 담긴 벡터들이 나오게 됨

            #|qshft_seqs| = (bs, sq) -> tensor([[43., 43., 79.,  ..., -0., -0., -0.], [59., 15., 47.,  ..., -0., -0., -0.],...])
            #|self.num_q| = 100
            
            # 이동된 질문 시퀀스를 one-hot 벡터로 변환
            one_hot_vectors = one_hot(qshft_seqs.long(), self.num_q) #|one_hot_vectors| = (bs, sq, self.num_q) -> tensor([[[0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0],..], [[]])
            #=> qshft는 한칸뒤의 벡터임, 각 seqeunce 별로 웟핫 벡터를 순서대로 만듦
            # 원핫벡터는 다 0이고 하나만 1이니까, 내가 원하는 위치에 있는 애만 살아남음
            # qshft는 한칸뒤의 벡터임 >> 즉, 한 칸 옮긴 것들의 값들만 살아남음


            # 예측 값(y_hat)에 one-hot 벡터를 곱하고, 축을 기준으로 합산하여 각 시퀀스의 예측 확률을 구함
            # 내가 예측한 다음 값들만, 즉 지금 입력한 것들의 다음 시점을 예측하는 확률들만 남게 됨.
            y_hat = ( y_hat * one_hot_vectors ).sum(-1) #|y_hat| = (bs, sq) -> tensor([[0.5711, 0.7497, 0.8459,  ..., 0.6606, 0.6639, 0.6702], [0.5721, 0.6495, 0.6956,  ..., 0.6677, 0.6687, 0.6629],
            # => 각 문항별 확률값만 추출해서 담고, 차원을 축소함

            # 마스크를 적용하여 실제로 문제를 푼 경우의 예측 값과 정답 값을 추출
            y_hat = torch.masked_select(y_hat, mask_seqs) #|y_hat| = () -> tensor([0.7782, 0.8887, 0.7638,  ..., 0.8772, 0.8706, 0.8831])
            #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 확률값만 추출

            correct = torch.masked_select(rshft_seqs, mask_seqs) #|correct| = () -> tensor([0., 1., 1.,  ..., 1., 1., 1.])
            #=> y_hat은 다음 값을 예측하게 되므로, 한칸 뒤의 정답값인 rshft_seqs을 가져옴
            #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 정답값만 추출

            # 손실 함수를 계산
            loss = self.crit(y_hat, correct) #|loss| = () -> ex) 0.5432


            # grad_accumulation
            # 그래디언트 누적이 활성화되어 있다면 일정 반복 횟수마다 그래디언트를 업데이트하고, 
            # 그렇지 않다면 그때 그때 그래디언트를 업데이트
            if self.grad_acc == True:
                loss.backward()     # 자동미분이 됨.
                if (idx + 1) % self.grad_acc_iter == 0:     # self.grad_acc_iter 의 배수마다 loss 값 도출
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 예측 값, 실제 값, 손실 값을 저장
            y_trues.append(correct)
            y_scores.append(y_hat)
            loss_list.append(loss)

        # 리스트로 저장된 값을 하나의 배열로 변환하고, CPU로 이동.
        y_trues = torch.cat(y_trues).detach().cpu().numpy() #|y_tures| = () -> [0. 0. 0. ... 1. 1. 1.]
        y_scores = torch.cat(y_scores).detach().cpu().numpy() #|y_scores| = () ->  tensor(0.5552)
        # y_scores 리스트에 추가되는 각 텐서의 크기를 확인합니다.



        # AUC 점수를 계산
        auc_score += metrics.roc_auc_score( y_trues, y_scores ) #|metrics.roc_auc_score( y_trues, y_scores )| = () -> 0.6203433289463159

        # 손실 결과를 계산
        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result

    # def _train 과 과정은 동일       
    def _validate(self, valid_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

    # validate 는 학습을 하면 안되기 때문에 no_grad
        with torch.no_grad():
            for data in tqdm(valid_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat = self.model( q_seqs.long(), r_seqs.long() )
                y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(rshft_seqs, mask_seqs)

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result
        
    # test 는 학습을 하면 안되기 때문에 no_grad        
    def _test(self, test_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat = self.model( q_seqs.long(), r_seqs.long() )
                y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(rshft_seqs, mask_seqs)

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result
      
    # auc용으로 train
    def train(self, train_loader, valid_loader, test_loader, config):
        
        if config.crit == "binary_cross_entropy":
            best_valid_score = 0
            best_test_score = 0
            metric_name = "AUC"
        elif config.crit == "rmse":
            best_valid_score = float('inf')
            best_test_score = float('inf')
            metric_name = "RMSE"
        
        #출력을 위한 기록용
        train_scores = []
        valid_scores = []
        test_scores = []
        #best_model = None

        # early_stopping 선언
        early_stopping = EarlyStopping(metric_name=metric_name,
                                    best_score=best_valid_score)

        # Train and Valid Session
        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            # Training Session
            train_score = self._train(train_loader, metric_name)
            valid_score = self._validate(valid_loader, metric_name)
            test_score = self._test(test_loader, metric_name)

            # train, test record 저장
            train_scores.append(train_score)
            valid_scores.append(valid_score)
            test_scores.append(test_score)

            # early stop
            train_scores_avg = np.average(train_scores)
            valid_scores_avg = np.average(valid_scores)
            early_stopping(valid_scores_avg, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if config.crit == "binary_cross_entropy":
                if test_score >= best_test_score:
                    best_test_score = test_score
                    #best_model = deepcopy(self.model.state_dict())
            elif config.crit == "rmse":
                if test_score <= best_test_score:
                    best_test_score = test_score
                    #best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) result: train_score=%.4f  valid_score=%.4f test_score=%.4f best_test_score=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_score,
                valid_score,
                test_score,
                best_test_score,
            ))

        print("\n")
        print("The Best Test Score(" + metric_name + ") in Testing Session is %.4f" % (
                best_test_score,
            ))
        print("\n")
        
        # 가장 최고의 모델 복구
        #self.model.load_state_dict(best_model)
        self.model.load_state_dict(torch.load("/Users/mj/Deep_Knowledge_Tracing_Baseline/checkpoints/checkpoint.pt"))

        return train_scores, valid_scores, \
            best_valid_score, best_test_score
    '''
    Gradient accumulation(그라디언트 누적)은 GPU 메모리 부족 문제를 해결하기 위한 기법 중 하나. 
    일반적으로 모델의 매개변수 업데이트를 위해 사용되는 그라디언트를 한 번에 계산하고 적용하는 대신, 
    여러 미니배치를 사용하여 그라디언트를 누적한 후에 업데이트를 수행. 
    이 기법은 GPU 메모리를 효율적으로 사용하고 더 큰 미니배치 크기를 처리할 수 있게 해준다.

    Gradient accumulation은 다음과 같은 단계로 작동:

    미니배치 반복: 학습 데이터를 미니배치로 나누어 각 미니배치에 대해 순회합니다.
    그라디언트 계산: 각 미니배치에 대해 손실 함수의 그라디언트를 계산합니다.
    그라디언트 누적: 계산된 그라디언트를 메모리에 누적합니다.
    업데이트 스텝: 정해진 누적 횟수에 도달하면(일반적으로 매 n번째 미니배치마다), 
    누적된 그라디언트를 사용하여 모델의 매개변수를 업데이트합니다.
    그라디언트 초기화: 업데이트가 수행된 후에는 누적된 그라디언트를 초기화합니다.

    Gradient accumulation은 주로 다음과 같은 상황에서 유용합니다:

    GPU 메모리가 제한적인 경우: 더 큰 미니배치 크기를 사용하여 모델을 효율적으로 학습할 수 있습니다.
    미니배치 크기가 너무 작아 모델이 제대로 학습되지 않는 경우: 그라디언트를 누적하여 더 많은 데이터로 학습할 수 있습니다.
    학습 속도를 조절해야 하는 경우: 그라디언트 업데이트를 자주 수행하지 않고 누적할 수 있습니다.

    Gradient accumulation은 주로 파이토치(PyTorch)와 같은 딥러닝 프레임워크에서 지원되며, 
    optimizer.step()을 호출하기 전에 누적된 그라디언트를 사용하여 매개변수를 업데이트하는 방식으로 구현됩니다.
    
    '''
