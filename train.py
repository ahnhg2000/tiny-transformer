import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 결과의 일관성을 위해 랜덤 시드 고정
tf.random.set_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset & Hyperparameters (학습 데이터 및 하이퍼파라미터 정의)
# =====================================================================
print("# 1. Dataset & Hyperparameters")

# 6개의 간단한 학습용 데이터셋 (긍정 3개, 부정 3개)
raw_sentences = [
    "i love this movie",    # 긍정 (1)
    "this movie is great",   # 긍정 (1)
    "i liked this film",    # 긍정 (1)
    "i hate this movie",    # 부정 (0)
    "this movie is bad",     # 부정 (0)
    "i disliked this film"   # 부정 (0)
]
labels = tf.constant([[1], [1], [1], [0], [0], [0]], dtype=tf.float32)

# 단어 사전(Vocabulary) 자동 구축
vocab = {"<PAD>": 0}
for sentence in raw_sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
seq_len = 4    # 문장 길이 고정
d_model = 16   # GPT/BERT 구조의 표현력을 위해 차원을 16으로 설정

# 문장을 토큰 ID 배열로 변환
input_data = [[vocab[word] for word in sent.split()] for sent in raw_sentences]
input_ids = tf.constant(input_data, dtype=tf.int32)

print(f"단어 사전(Vocab): {vocab}")
print(f"입력 데이터 Shape: {input_ids.shape} -> [Batch_Size(6), Seq_Len(4)]\n")


# =====================================================================
# 2. Transformer Model Definition (트랜스포머 모델 클래스 정의)
# =====================================================================
print("# 2. Transformer Model Definition")

class TransformerClassifier(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 레이어 정의
        self.token_emb_layer = layers.Embedding(input_dim=vocab_size, output_dim=d_model, name="token_emb")
        self.pos_emb_layer = layers.Embedding(input_dim=seq_len, output_dim=d_model, name="pos_emb")
        self.position_indices = tf.range(start=0, limit=seq_len, delta=1)
        
        # 텐서플로우 공식 MultiHeadAttention 라이브러리 사용
        self.transformer_attention = layers.MultiHeadAttention(num_heads=2, key_dim=d_model, name="attention")
        
        # 잔차 연결 및 정규화 레이어
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6, name="ln_1")
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6, name="ln_2")
        
        # 피드 포워드 네트워크(FFN) 레이어
        self.ffn_layer1 = layers.Dense(units=32, activation='relu', name="ffn_1") # 16차원 -> 32차원 확장
        self.ffn_layer2 = layers.Dense(units=d_model, name="ffn_2")              # 32차원 -> 16차원 복원
        
        # 최종 분류 헤드
        self.classification_layer = layers.Dense(units=1, activation='sigmoid', name="clf_head")

    def call(self, ids):
        # 토큰 및 포지셔널 임베딩 결합
        w_emb = self.token_emb_layer(ids)
        p_emb = self.pos_emb_layer(self.position_indices)
        x = w_emb + tf.expand_dims(p_emb, axis=0)
        
        # 셀프 어텐션 (Query, Key, Value 자리에 모두 똑같은 'x' 입력)
        attention_output = self.transformer_attention(query=x, key=x, value=x)
        
        # 첫 번째 잔차 연결 + 레이어 정규화
        out_1 = self.layer_norm_1(x + attention_output)
        
        # 피드 포워드 + 두 번째 잔차 연결 + 레이어 정규화
        ffn_out = self.ffn_layer2(self.ffn_layer1(out_1))
        out_2 = self.layer_norm_2(out_1 + ffn_out)
        
        # 마지막 토큰 추출 (BERT/GPT 스타일의 문장 요약)
        last_token = out_2[:, -1, :]
        
        # Dense 연결 및 예측 확률 반환
        pred = self.classification_layer(last_token)
        return pred

# 모델 인스턴스 생성 및 컴파일
model = TransformerClassifier(vocab_size, seq_len, d_model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# =====================================================================
# 3. Model Training (모델 학습 진행)
# =====================================================================
print("# 3. Model Training")

# 60번 반복 학습을 통해 모델이 단어들의 관계를 스스로 학습하도록 유도합니다.
# 주피터 노트북 스타일에서 가볍게 진행하기 위해 verbose=1로 설정하여 상태를 출력합니다.
model.fit(input_ids, labels, epochs=60, batch_size=6, verbose=1)
print("✔ 훈련 데이터 기반 바닥부터 학습 완료!\n")


# =====================================================================
# 4. Inference on New Data (새로운 데이터 예측 테스트)
# =====================================================================
print("# 4. Inference on New Data")

# 한 번도 정답을 알려준 적 없는 새로운 조합의 문장으로 테스트
test_sentences = [
    "i liked this movie",  # 긍정 단어(liked) 패턴 인식 확인
    "i hate this film"     # 부정 단어(hate) 패턴 인식 확인
]

# 새로운 테스트 문장을 토큰 ID로 변환 (기존 단어 사전 기준)
test_input_data = [[vocab[word] for word in sentence.split()] for sentence in test_sentences]
test_input_ids = tf.constant(test_input_data, dtype=tf.int32)

# 모델 추론 수행
final_predictions = model(test_input_ids)

for i, sentence in enumerate(test_sentences):
    prob = final_predictions.numpy()[i][0]
    result = "긍정 (Positive)" if prob > 0.5 else "부정 (Negative)"
    print(f"입력 문장: \"{sentence}\"")
    print(f" └ 예측 확률: {prob:.4f} -> 최종 판정 결과: {result}")
