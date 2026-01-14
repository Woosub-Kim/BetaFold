# BetaFold: small peptide structure predictor

좋아 — **“ESM2(8M) 동결 + Tiny Transformer head”**는 M4 맥미니에서 *현실적으로* 돌리면서도 Transformer를 제대로 체득하기에 딱 좋은 선택이야.
아래는 지금까지의 로드맵을 **Transformer 학습/이해를 중심축**으로 다시 엮은 **종합(학습+개발) 로드맵**이야.

---

### 최종 목표(현실 버전)

* 입력: peptide 서열(길이 10–30)
* 특징: ESM2 8M 임베딩(동결, 가능하면 **선계산 캐시**)
* 모델: **Tiny Transformer(2–4층)**
* 출력(1차): residue별 토션(φ/ψ/ω) = sin/cos
* 출력(2차, 선택): distogram(Cα–Cα 거리) + confidence
* 재구성: 토션 → backbone 좌표(NeRF 방식)
* 평가: 토션 오차 + Kabsch RMSD + distogram 오차(선택)

---

## Transformer 중심 “학습 & 개발” 종합 로드맵 (추천 8주)

각 주차는 **(학습 포인트 → 구현 산출물 → 성공 체크)**로 구성했어.

---

### 1주차: Transformer 핵심 개념을 “코드로” 고정하기 (P0)

#### 학습 포인트

* Scaled Dot-Product Attention 수식/shape (Q,K,V: `[B,L,d]`)
* 마스킹(패딩 마스크/어텐션 마스크)의 의미와 구현
* Residual + (Pre-LN vs Post-LN) 차이, 왜 Pre-LN이 학습 안정적인지
* FFN(MLP) 역할, dropout 위치

#### 개발 산출물

* **Tiny Transformer block**을 직접 구현(최소 1개 블록)
* toy task로 sanity check:

  * copy task / shift task 같이 “학습이 되면 attention이 맞게 동작한다”를 확인할 수 있는 것

#### 성공 체크

* toy task에서 loss가 뚝뚝 떨어지고, 마스크 넣었을 때/뺐을 때 결과가 달라짐이 명확함

> 여기서 “Transformer가 진짜로 뭘 하는지” 감이 잡히면 이후 단백질로 옮겼을 때 디버깅이 압도적으로 쉬워져.

---

### 2주차: 데이터 파이프라인(10–30) + 라벨(토션) 만들기 (P0)

#### 학습 포인트

* 구조 데이터에서 제일 많이 터지는 버그: residue indexing / 결손 마스크 / chain 추출
* 토션(φ/ψ/ω) 계산과 마스크 처리(첫/끝 residue 등)
* 누수 방지의 최소 조건(중복 서열 제거라도 반드시)

#### 개발 산출물

* `dataset_builder.py`

  * (서열, N/CA/C 좌표, 토션 라벨, 마스크)을 샘플 단위로 저장
* `Dataset/Dataloader`

  * 길이 10–30 패딩 + 패딩 마스크 생성

#### 성공 체크

* 1000개 내외 샘플에서 로딩/배치가 안정
* 토션 라벨에 NaN/이상치가 거의 없음

---

### 3주차: ESM2 임베딩 “선계산 캐시” 파이프라인 (P0)

#### 학습 포인트

* “비싼 부분(ESM forward)”과 “학습 대상(head)”을 분리하는 엔지니어링
* float16 저장, shard 저장(파일 너무 많아지는 문제 예방)

#### 개발 산출물

* `embed_cache.py`

  * 서열 → ESM 임베딩 `(L,320)` 저장
* 학습 시에는 임베딩만 읽어서 사용(ESM은 호출 안 함)

#### 성공 체크

* head 학습이 훨씬 빨라지고 반복 실험이 가능해짐(이게 맥에서는 결정적)

---

### 4주차: Tiny Transformer head로 “토션 예측” 베이스라인 완성 (P0→P1)

#### 학습 포인트(Transformer를 단백질에 적용)

* **Positional encoding** 선택(여기선 쉬운 게 정답)

  * 길이 10–30 고정이니 **learned absolute position embedding(최대 32)**가 가장 간단하고 잘 먹힘
* 패딩 마스크를 attention에 올바르게 적용하기

#### 개발 산출물(권장 아키텍처)

* 입력: ESM 임베딩 `(B,L,320)`
* * pos embedding `(B,L,320)`
* Transformer encoder: **2–4층, n_heads=8**(320/8=40이라 딱 떨어짐), FFN=4×320 정도
* 출력 head: `Linear(320 → 6*3)` = (φ,ψ,ω 각각 sin/cos)

#### 성공 체크

* train loss 감소 + val에서도 개선
* “구조가 비교적 단순한 샘플(짧은 helix 성향)”에서 토션 오차가 눈에 띄게 줄어듦

---

### 5주차: 토션 → 3D 재구성 + RMSD 평가까지 “end-to-end” 연결 (P0 핵심)

#### 학습 포인트

* 좌표 비교는 **정렬(Kabsch) 후 RMSD**가 기본
* 토션은 맞는데 좌표가 깨질 수 있음 → 디버깅 지표를 분리해서 봐야 함

  * (1) 토션 MAE
  * (2) Cα 거리행렬 오차
  * (3) RMSD

#### 개발 산출물

* `nerf_backbone.py` (토션 → N/CA/C 좌표 재구성)
* `metrics.py` (Kabsch RMSD, 거리행렬 오차)
* 샘플 몇 개를 PDB로 내보내서 시각적으로 확인하는 툴(강추)

#### 성공 체크

* RMSD 자체가 아직 높아도 OK
* 대신 **“언제/왜 망가지는지”**(끝단 폭주, 특정 아미노산 패턴 등) 설명 가능해짐

---

### 6주차: Transformer를 더 “구조적으로” 만들기 위한 1차 업그레이드 (P1)

여기서부터가 Transformer 이해가 성능으로 직결돼.

#### 학습 포인트

* attention head 수/층 수가 성능과 안정성에 미치는 영향(작은 모델에서는 과적합도 쉬움)
* dropout, weight decay, lr 스케줄이 실제로 어떤 변화를 주는지

#### 개발 산출물(권장 실험 3개만)

* (A) layers 2 vs 4 비교
* (B) positional encoding: learned vs sinusoidal 비교
* (C) Pre-LN 고정 후 dropout만 0.0 / 0.1 비교

#### 성공 체크

* 어떤 설정이 안정적인지 “패턴”이 보이기 시작함(이게 실력)

---

### 7주차: distogram 멀티태스크 추가(Transformer 이해 확장) (P1)

#### 학습 포인트

* residue feature → pair feature로 확장하는 대표 패턴(outer/concat)
* 멀티태스크 loss 밸런싱

#### 개발 산출물(가벼운 방식)

* Transformer 출력 `H: (B,L,320)`에서

  * pair feature `P_ij = concat(H_i, H_j, H_i*H_j)` 같은 단순 조합
  * 작은 MLP로 distogram 예측(회귀 또는 bin 분류)

#### 성공 체크

* distogram이 맞아질수록 3D 재구성이 덜 깨지는 경향이 생김(안정화)

---

### 8주차: Confidence(신뢰도) 헤드 + 캘리브레이션 (P1)

#### 학습 포인트

* “맞출 수 없는 샘플”이 존재할 때, 모델이 불확실성을 표현하도록 만드는 법
* 신뢰도와 실제 오차의 상관관계 체크

#### 개발 산출물

* residue별 predicted error(또는 sample-level quality) 예측 헤드
* “신뢰도 상위 20%”의 실제 RMSD가 유의미하게 낮아지는지 평가

#### 성공 체크

* 맞는 건 자신있게, 틀린 건 자신없게(이것만 돼도 실전 감각 크게 늘어)

---

## Tiny Transformer head 추천 스펙(맥 현실 최적화)

* d_model = **320**(ESM2 8M 그대로)
* n_heads = **8**
* n_layers = **2부터 시작 → 4로 확장**
* FFN dim = **1280(=4×320)**
* dropout = **0.1**
* 학습: head만(ESM 동결), 배치 크게 가능(길이가 짧으니까)
* 가장 큰 최적화: **ESM 임베딩 캐시 후 head만 학습**

---

## “Transformer 이해”에 직접 도움 되는 체크리스트(필수 5개)

프로젝트 진행 중 아래를 꼭 해봐. 이해가 확 빨라져.

1. **attention map을 한두 샘플에서 시각화**(길이 30이면 눈으로 보기 좋음)
2. 패딩 마스크를 일부러 틀리게 넣어보고 성능이 어떻게 망가지는지 관찰
3. positional encoding을 바꾸면 어떤 샘플에서 좋아/나빠지는지 분석
4. layer 수 늘리면 언제 과적합이 빨리 오는지 관찰
5. distogram을 넣으면 학습이 왜 안정화되는지(gradient/오류 패턴) 확인

---
* **B) 핵심 모델 코드 중심(Tiny Transformer 구현부터)**

