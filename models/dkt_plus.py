# dkt.py 와 과정 동일
'''
위 코드에서 사용된 각 줄은 다음과 같은 역할을 한다.

    필요한 모듈 및 클래스를 가져오기.
    DKT_plus 클래스를 정의.
    클래스 생성자에서 모델의 파라미터를 초기화.
    Embedding 레이어를 정의하여 입력 데이터를 임베딩.
    LSTM 레이어를 정의하여 시퀀스 데이터를 처리.
    출력 레이어를 정의하여 LSTM의 출력을 다음 질문에 대한 예측 확률로 변환.
    forward 메서드를 정의하여 순전파 연산을 수행. 이 과정에서 데이터를 준비하고 모델의 각 레이어를 통과.
    최종 예측 확률을 반환.

이러한 구성은 학습 데이터를 모델에 입력하고 예측을 생성하는 데 사용.
'''
from torch.nn import Module, Embedding, LSTM, Sequential, Linear, Sigmoid

class DKT_plus(Module):
    #num_q: 유일한 질문의 갯수
    #emb_size: 100
    #hidden_size: 100
    def __init__(
        self,
        num_q,
        emb_size,
        hidden_size,
        n_layers = 4,
        dropout_p = .2
    ):
        super().__init__()

        self.num_q = num_q #100
        self.emb_size = emb_size #100
        self.hidden_size = hidden_size #100
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        #|self.interaction_emb| = (200, 100) -> 즉, 전체 문항의 맞고 틀림을 고려해서 문항수*2만큼의 행이 만들어지고, 각 행들은 embedding값으로 채워짐
        self.interaction_emb = Embedding( 
            self.num_q * 2, self.emb_size
        ) 
        #100을 받아서 100이 나옴
        self.lstm_layer = LSTM( 
            input_size = self.emb_size,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.n_layers,
            dropout = self.dropout_p
        )  
        #100을 받아서 100이 나옴
        self.out_layer = Sequential(
            Linear(self.hidden_size, self.num_q),
            Sigmoid()
        )

    def forward(self, q_seqs, r_seqs):
        #|q_seqs| = (bs, sq), |r_seqs| = (bs, sq)
        x = q_seqs + self.num_q * r_seqs #|x| = (bs, sq)

        interaction_emb = self.interaction_emb(x) #|interaction_emb| = (bs, sq, self.emb_size) -> 각각의 x에 해당하는 embedding값이 정해짐

        z, _ = self.lstm_layer( interaction_emb ) #|z| = (bs, sq, self.hidden_size)

        y = self.out_layer(z) #|y| = (bs, sq, self.num_q) -> 통과시키면 확률값이 나옴

        return y
