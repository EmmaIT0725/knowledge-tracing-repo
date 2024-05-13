from torch.nn import Module, Embedding, LSTM, Sequential, Linear, Sigmoid
# Torch의 nn 모듈에서 Module, Embedding, LSTM, Sequential, Linear, Sigmoid 클래스 가져오기
# 이들은 딥러닝 모델을 구축하는 데 사용되는 PyTorch의 클래스들

class DKT(Module):  # DKT 클래스 정의 - 이 클래스는 Module을 상속받으므로 Pytorch 모델을 나타냄
    # Module을 상속받아서 쓰면 좋은점: def forward를 사용할 수 있다.
    # num_q: 유일한 질문의 갯수
    # emb_size: 100
    # hidden_size: 100
    def __init__(   # DKT 클래스의 생성자 메서드: 이 메서드는 DKT 객체가 생성될 때 호출.
        self,
        num_q,  # 유일한 질문의 갯수
        emb_size,   # Embedding 벡터의 크기
        hidden_size,    # LSTM 레이어의 hidden state 크기
        n_layers = 4,   # LSTM 레이어의 층 수(기본값은 4)
        dropout_p = .2  # LSTM 레이어의 드롭아웃 확률(기본값은 0.2)
    ):
        super().__init__()      # 상위 클래스(Module)의 생성자를 호출

        self.num_q = num_q #100     # 입력된 매개변수들을 객체의 속성으로 설정
        self.emb_size = emb_size #100
        self.hidden_size = hidden_size #100
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        #|self.interaction_emb| = (200, 100) 
        # -> 즉, 전체 문항의 맞고 틀림을 고려해서 문항수*2만큼의 행이 만들어지고, 각 행들은 embedding값으로 채워짐
        self.interaction_emb = Embedding( 
            self.num_q * 2, self.emb_size
        )   
        # Embedding 레이어 정의: 학생의 질문 및 응답 이력을 Embedding하는 데 사용
        # Embedding의 크기는 (self.num_q * 2, self.emb_size)

        # LSTM 레이어 정의
        # LSTM은 시퀀스 데이터를 처리하는 데 사용
        # 입력크기는 Embedding 크기 / 은닉 상태 크기는 hidden_size
        # 100을 받아서 100이 나옴
        self.lstm_layer = LSTM( 
            input_size = self.emb_size,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.n_layers,
            dropout = self.dropout_p
        )  
        #100을 받아서 100이 나옴
        # 출력레이어 정의
        # LSTM의 출력을 입력으로 받아서 다음 질문에 대한 예측 확률을 출력
        # LSTM의 출력 크기를 num_q로 변환하는 선형 레이어를 적용
        # 그 결과에 시그모이드 활성화 함수를 적용
        self.out_layer = Sequential(
            Linear(self.hidden_size, self.num_q),
            Sigmoid()
        )

    # forward 메서드는 신경망의 순전파 연산을 정의
    # q_seqs와 r_seqs는 각각 질문 시퀀스와 학생의 응답 시퀀스
    def forward(self, q_seqs, r_seqs):
        #|q_seqs| = (bs, sq), |r_seqs| = (bs, sq) 
        # bs: 행 - 배치 사이즈
        # sq: 열 - 시퀀스

        # 세로줄 즉, 행 전체가 배치사이즈
        # 가로줄 한 줄이 학생 시퀀스 하나
        # ex) bs:100 - 학생 100명의 시퀀스를 가져온 것
        '''
        q_seqs: tensor([[94, 94, 94,  ...,  0,  0,  0],
        [96, 96,  0,  ...,  0,  0,  0],
        [57, 62, 93,  ...,  0,  0,  0],
        ...,
        [71, 71, 71,  ...,  0,  0,  0],
        [43, 43, 43,  ...,  0,  0,  0],
        [79, 31, 32,  ...,  0,  0,  0]], device='cuda:0')        
        '''
        # 입력데이터 준비: 학생의 질문 및 응답 이력을 하나의 텐서로 결합
        x = q_seqs + self.num_q * r_seqs #|x| = (bs, sq)

        # Embedding 레이어를 통과시켜서 학생의 질문 및 응답 이력을 Embedding 벡터로 변환
        interaction_emb = self.interaction_emb(x) #|interaction_emb| = (bs, sq, self.emb_size) -> 각각의 x에 해당하는 embedding값이 정해짐

        # LSTM 레이어를 통과시켜서 시퀀스 데이터를 처리하고, 결과인 z를 얻음.
        z, _ = self.lstm_layer( interaction_emb ) #|z| = (bs, sq, self.hidden_size)

        # 출력 레이어를 통과시켜서 각 시퀀스의 다음 질문에 대한 예측 확률을 얻음
        y = self.out_layer(z) #|y| = (bs, sq, self.num_q) -> 통과시키면 확률값이 나옴

        # 예측 확률을 반환
        return y
