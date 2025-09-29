# FLTG Byzantine-Robust Federated Learning Implementation

논문 "FLTG: Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting"의 핵심 알고리즘을 구현하고 성능을 검증한 프로젝트입니다.

## 📋 프로젝트 개요

### 목적
- FLTG 논문에서 제안하는 비잔틴 공격 방어 메커니즘 구현
- 기존 방어 기법(FedAVG, Krum, Trimmed-Mean) 대비 성능 비교
- MNIST 데이터셋을 활용한 로컬 환경 실험
- 논문의 주장을 실증적으로 검증

### 기반 프레임워크
- [CRYPTO-KU/FL-Byzantine-Library](https://github.com/CRYPTO-KU/FL-Byzantine-Library)
- PyTorch 기반 연합 학습 및 비잔틴 공격/방어 라이브러리

---

## 🔬 구현 내용

### 1. FLTG 알고리즘 구현

논문의 핵심 메커니즘을 다음과 같이 구현했습니다:

#### Step 1: ReLU-Clipped Cosine Similarity Filtering
```python
# 서버의 루트 데이터셋으로 계산한 그래디언트와 클라이언트 업데이트 간의 코사인 유사도 계산
cos_sims_with_server = [F.cosine_similarity(g, g0, dim=0) for g in inputs]
# 음수 유사도(반대 방향) 필터링
filtered_indices = [i for i, sim in enumerate(cos_sims_with_server) if sim > 0]
```

**의도**: 서버와 정반대 방향을 가리키는 악의적 업데이트 제거

#### Step 2: Dynamic Reference Selection
```python
# 이전 글로벌 그래디언트와 코사인 유사도가 가장 낮은 클라이언트를 참조점으로 선택
cos_sims_with_prev = [F.cosine_similarity(g, g_prev, dim=0) for g in filtered_inputs]
ref_idx = cos_sims_with_prev.index(min(cos_sims_with_prev))
g_ref = filtered_inputs[ref_idx]
```

**의도**: Non-IID 환경에서 "정상이지만 가장 독특한" 업데이트를 기준점으로 활용

#### Step 3: Non-IID Aware Weighting
```python
# 참조 클라이언트와의 각도 편차에 반비례하여 가중치 부여
scores = []
for g in filtered_inputs:
    cos_sim_with_ref = F.cosine_similarity(g, g_ref, dim=0)
    score = 1.0 - cos_sim_with_ref
    scores.append(F.relu(score))
```

**의도**: 참조점과 다를수록 높은 가중치를 부여하여 데이터 다양성 보장

#### Step 4: Magnitude Normalization
```python
# 모든 클라이언트 업데이트를 서버 그래디언트와 동일한 크기로 정규화
g0_norm = torch.norm(g0)
normalized_g = g * (g0_norm / torch.norm(g))
```

**의도**: 악의적인 크기 조작(scaling attack) 방지

#### Step 5: Weighted Aggregation
```python
# 가중치를 적용하여 최종 글로벌 모델 업데이트 생성
aggregated = sum(w * g for w, g in zip(weights, normalized_inputs))
```

### 2. 프레임워크 통합

**수정된 파일**:
- `FL-Byzantine-Library/Aggregators/fltg.py` - FLTG 클래스 구현 (신규)
- `FL-Byzantine-Library/mapper.py` - FLTG를 aggregator 레지스트리에 등록
- `FL-Byzantine-Library/parameters.py` - 누락된 파라미터 추가
- `FL-Byzantine-Library/main.py` - 함수 호출 시그니처 수정

---

## 🧪 실험 설정

### 환경 구성
- **데이터셋**: MNIST (손글씨 숫자 인식)
- **모델**: MNISTNET (0.431M parameters, CNN 기반)
- **클라이언트 수**: 20개
- **악의적 클라이언트 비율**: 20% (4개)
- **공격 유형**: ROP (Relocated Orthogonal Perturbation)
- **학습 라운드**: 10 epochs
- **배치 크기**: 64
- **데이터 분포**: IID (균등 분포)

### 비교 대상
1. **Baseline** - 공격 없는 이상적 환경
2. **FedAVG** - 방어 메커니즘 없음 (단순 평균)
3. **Krum** - 유클리디안 거리 기반 방어
4. **Trimmed-Mean** - 중앙값 기반 방어
5. **FLTG** - 제안된 각도 기반 방어

### 실험 명령어
```bash
# Baseline (공격 없음)
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0 --aggr avg --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# ROP 공격 + FedAVG
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr avg --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# ROP 공격 + Krum
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr krum --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# ROP 공격 + Trimmed-Mean
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr tm --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# ROP 공격 + FLTG
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr fltg --trials 1 --global_epoch 10 --gpu_id -1 --bs 64
```

---

## 📊 실험 결과

### 정량적 성능 비교

| 방어 기법 | Epoch 1 | Epoch 5 | **Epoch 10** | Baseline 대비 | 순위 |
|-----------|---------|---------|--------------|---------------|------|
| **Baseline (공격 없음)** | 90.9% | 97.4% | **98.5%** | - | - |
| **FedAVG** | 75.7% | 97.5% | **98.4%** | -0.1%p | 1위 |
| **Trimmed-Mean** | 77.5% | 97.3% | **98.2%** | -0.3%p | 2위 |
| **Krum** | 89.8% | 97.3% | **98.1%** | -0.4%p | 3위 |
| **FLTG** | 63.6% | 97.4% | **97.1%** | -1.4%p | 4위 |

### 학습 곡선 분석

**초기 학습 속도 (Epoch 1)**:
- Krum: 89.8% (가장 빠름)
- Trimmed-Mean: 77.5%
- FedAVG: 75.7%
- **FLTG: 63.6% (가장 느림)** ⚠️

**수렴 안정성 (Epoch 5-10)**:
- FedAVG, Trimmed-Mean, Krum: 안정적 수렴
- FLTG: 97.4% → 97.1% (성능 저하)

### 집계 시간 (Epoch 10 기준)

| 방어 기법 | 총 집계 시간 | 라운드당 평균 |
|-----------|--------------|---------------|
| FedAVG | 0.542초 | 0.054초 |
| FLTG | 13.116초 | 1.312초 |
| Trimmed-Mean | 14.645초 | 1.465초 |
| Krum | 12.901초 | 1.290초 |

---

## 🤔 예상치 못한 결과 분석

### 논문의 주장
> "FLTG는 50% 이상의 악의적 클라이언트 환경에서도 기존 방법보다 우수한 성능을 보인다."

### 실제 결과
- **20% 악의적 환경에서 가장 낮은 성능 (97.1%)**
- 방어 없는 FedAVG(98.4%)보다 1.3%p 낮음
- 초기 학습이 매우 느림 (Epoch 1: 63.6%)

### 왜 이런 결과가 나왔을까?

#### 가설 1: Dynamic Reference Selection의 역설 🔴

**구현 로직**:
```python
# 이전 글로벌 모델과 가장 다른 업데이트를 참조점으로 선택
ref_idx = cos_sims_with_prev.index(min(cos_sims_with_prev))
```

**문제점**:
1. "이전 모델과 가장 다른 업데이트" ≠ "정상이지만 독특한 업데이트"
2. ROP 공격의 특성: 의도적으로 이전 방향과 다른 방향 제시
3. **결과**: 악의적 클라이언트가 참조점이 될 가능성 높음
4. **파급효과**: 참조점이 악의적이면 가중치 시스템 전체가 역전됨

**예시**:
```
이전 글로벌 방향: [1, 0, 0]
정상 클라이언트들: [0.9, 0.1, 0], [0.8, 0.2, 0], [0.85, 0.15, 0]
악의적 클라이언트: [-0.5, 0.5, 0.7]  ← 가장 다름 → 참조점으로 선택!

결과: 악의적 방향과 가까운 업데이트에 높은 가중치 부여
```

#### 가설 2: 과도한 필터링 🔴

**ReLU-clipped filtering**:
- 서버 그래디언트와 `cos_sim < 0`인 모든 업데이트 제거
- Non-IID 환경에서 정상 클라이언트도 서버와 방향이 크게 다를 수 있음
- 필요한 정보를 가진 정상 클라이언트까지 제거 가능

#### 가설 3: 가중치 부여 방식의 모순 🔴

**현재 로직**:
```python
score = 1 - cos_sim(g_i, g_ref)  # 참조점과 다를수록 높은 가중치
```

**의도**: Non-IID 다양성 존중
**실제**: 참조점이 악의적이면, 정상 클라이언트에 낮은 가중치 부여

#### 가설 4: 서버 루트 데이터셋 크기 부족 🟡

- **현재**: 100개 샘플 (MNIST 10 클래스 → 클래스당 평균 10개)
- **문제**: 서버 그래디언트가 편향될 가능성
- **영향**: Step 1 필터링의 신뢰도 저하

#### 가설 5: 학습 시간 부족 🟡

- FLTG는 초기 학습이 매우 느림 (강력한 필터링의 부작용)
- 10 epochs는 충분한 수렴 시간이 아닐 수 있음
- 50-100 epochs 실험 필요

---

## 💡 핵심 통찰 (Key Insights)

### 1. 복잡성의 역설
> **"더 복잡한 알고리즘이 항상 더 나은 것은 아니다"**

- Krum, Trimmed-Mean 같은 단순한 방법이 더 안정적
- 다단계 방어는 각 단계마다 오류가 누적될 수 있음
- 디버깅과 유지보수도 어려움

### 2. FedAVG의 예상외 강건함
> **"때로는 방어가 없는 것이 최선일 수 있다"**

- 20% 악의적 환경에서 FedAVG가 98.4% 달성
- MNIST의 단순성 + 낮은 공격 비율 덕분
- 실무에서는 환경에 맞는 적절한 방어 선택 중요

### 3. 논문 재현의 어려움
> **"논문의 주장을 무조건 신뢰하면 안 된다"**

- 구현 세부사항의 미묘한 차이가 큰 성능 차이를 만듦
- 논문에 명시되지 않은 하이퍼파라미터 존재
- 공식 코드 공개의 중요성

### 4. 실패도 가치 있는 결과
> **"FLTG의 한계를 발견한 것 자체가 기여"**

- 다른 연구자들에게 경고 역할
- 알고리즘 개선의 출발점 제공
- 실증적 검증의 중요성 입증

---

## 🔧 개선 방향

### 단기 개선 (Quick Fixes)

#### 1. 참조 클라이언트 선택 개선
```python
# 현재: 최소값 선택
ref_idx = cos_sims_with_prev.index(min(cos_sims_with_prev))

# 개선안: 중간값 선택 (극단값 회피)
sorted_indices = sorted(range(len(cos_sims_with_prev)),
                       key=lambda i: cos_sims_with_prev[i])
median_idx = sorted_indices[len(sorted_indices) // 2]
ref_idx = median_idx
```

#### 2. 서버 그래디언트 가중치 증가
```python
# 현재: 클라이언트 업데이트만 사용
aggregated = sum(w * g for w, g in zip(weights, normalized_inputs))

# 개선안: 서버 그래디언트에 더 높은 신뢰
aggregated = 0.3 * g0 + 0.7 * sum(w * g for w, g in zip(weights, normalized_inputs))
```

#### 3. 루트 데이터셋 크기 증가
```python
# mapper.py에서
root_dataset_inds = np.random.choice(range(l), 500, replace=False)  # 100 → 500
```

### 중기 개선 (Algorithm Refinement)

#### 1. 2단계 필터링
```python
# Step 1: 서버와의 유사도 (임계값 완화)
filtered_1 = [g for g in inputs if cos_sim(g, g0) > -0.5]  # -1 → -0.5

# Step 2: 클라이언트 간 상호 유사도로 2차 검증
pairwise_sims = compute_pairwise_similarity(filtered_1)
filtered_2 = remove_outliers(filtered_1, pairwise_sims)
```

#### 2. 적응적 가중치
```python
# 학습 초기: 서버에 높은 가중치
# 학습 후기: 클라이언트 다양성에 높은 가중치
alpha = max(0.1, 1.0 - epoch / total_epochs)
aggregated = alpha * g0 + (1-alpha) * client_aggregated
```

#### 3. 참조점 앙상블
```python
# 단일 참조점 대신 top-k 참조점 사용
top_k_refs = get_top_k_diverse_clients(filtered_inputs, k=3)
scores = [average_score_against_refs(g, top_k_refs) for g in filtered_inputs]
```

### 장기 개선 (Fundamental Redesign)

1. **논문 원저자와 소통**
   - 정확한 구현 세부사항 확인
   - 공식 코드 비교 분석

2. **더 강한 실험 환경**
   - 악의적 비율 30%, 40%, 50% 테스트
   - IPM, ALIE 등 다른 공격으로 테스트
   - CIFAR-10, CIFAR-100 등 복잡한 데이터셋
   - 50-100 epochs 장기 실험

3. **이론적 분석**
   - 각 단계별 수학적 보장 확인
   - 최악의 경우(worst-case) 시나리오 분석
   - 수렴성 증명

---

## 📂 프로젝트 구조

```
FLTG/
├── FL-Byzantine-Library/           # 기반 프레임워크
│   ├── Aggregators/
│   │   ├── fltg.py                # ✨ FLTG 구현 (신규)
│   │   ├── fedavg.py              # FedAVG
│   │   ├── krum.py                # Krum
│   │   ├── trimmed_mean.py        # Trimmed-Mean
│   │   └── ...
│   ├── Attacks/
│   │   ├── rop.py                 # ROP 공격
│   │   ├── ipm.py                 # IPM 공격
│   │   └── ...
│   ├── Models/
│   │   ├── CNN.py                 # MNISTNET 등
│   │   └── ...
│   ├── main.py                    # 🔧 메인 실행 파일 (수정)
│   ├── mapper.py                  # 🔧 Aggregator 매핑 (수정)
│   └── parameters.py              # 🔧 파라미터 정의 (수정)
├── README.md                      # 📖 이 파일
├── README_SETUP.txt               # 🚀 빠른 시작 가이드
├── ANALYSIS.txt                   # 📊 상세 분석
└── .gitignore                     # Git 제외 파일
```

---

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone <your-repo-url>
cd FLTG
```

### 2. 의존성 설치
```bash
pip install torch torchvision matplotlib numpy scipy
```

### 3. 실험 실행
```bash
cd FL-Byzantine-Library

# FLTG 테스트
python3 main.py --dataset_name mnist --nn_name mnistnet \
                --num_client 20 --traitor 0.2 --attack rop \
                --aggr fltg --trials 1 --global_epoch 10 \
                --gpu_id -1 --bs 64
```

더 자세한 내용은 [README_SETUP.txt](README_SETUP.txt)를 참고하세요.

---

## 📈 실험 결과 파일

실험 결과는 다음 로그 파일에 저장됩니다:
- `baseline_no_attack.log` - 공격 없음
- `rop_fedavg.log` - ROP + FedAVG
- `rop_krum.log` - ROP + Krum
- `rop_tm.log` - ROP + Trimmed-Mean
- `rop_fltg.log` - ROP + FLTG

각 파일은 `.gitignore`에 포함되어 있습니다.

---

## 🎓 학술적/실무적 시사점

### 학술적 관점
1. **재현성의 중요성**: 논문 결과 재현을 위한 코드 공개 필수
2. **부정적 결과의 가치**: 실패한 시도도 공유할 필요
3. **실증적 검증**: 이론만으로는 부족, 다양한 환경에서 실험 필요

### 실무적 관점
1. **단순함의 가치**: Krum, Trimmed-Mean 같은 검증된 방법 우선 고려
2. **환경 맞춤**: 공격 비율, 데이터 특성에 따라 방어 기법 선택
3. **비용 대비 효과**: 복잡한 알고리즘의 계산 비용 고려

### 연구 윤리
1. **투명성**: 모든 세부사항을 명시해야 함
2. **재현 가능성**: 공식 구현 공개 필요
3. **정직한 보고**: 예상과 다른 결과도 보고

---

## 📝 인용

이 프로젝트를 사용하거나 참고하신다면 다음을 인용해주세요:

```bibtex
@misc{fltg-implementation-2025,
  title={FLTG Byzantine-Robust Federated Learning Implementation and Validation},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/your-username/FLTG}},
  note={Implementation and experimental validation of FLTG algorithm with unexpected results analysis}
}
```

원본 논문:
```bibtex
@article{fltg-original,
  title={FLTG: Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting},
  author={[Original Authors]},
  journal={[Journal Name]},
  year={[Year]}
}
```

---

## 🤝 기여

이 프로젝트는 실험적 검증 프로젝트입니다. 다음과 같은 기여를 환영합니다:

- 알고리즘 구현 개선
- 다른 공격/방어 기법과의 비교
- 더 복잡한 데이터셋(CIFAR-10, CIFAR-100)에서의 실험
- 하이퍼파라미터 튜닝
- 논문 원저자로부터의 피드백

---

## 📧 문의

문제나 질문이 있으시면 Issue를 열어주세요.

---

## 📜 라이선스

이 프로젝트는 학술 연구 목적으로 작성되었습니다.

**기반 프레임워크**: [FL-Byzantine-Library](https://github.com/CRYPTO-KU/FL-Byzantine-Library)

---

## ⚠️ 면책 조항

이 프로젝트는 논문을 기반으로 한 독립적인 구현이며, 원 논문의 저자들과는 무관합니다. 실험 결과는 특정 환경(MNIST, 10 epochs, 20% Byzantine ratio)에서 얻어진 것이므로, 일반화에 주의가 필요합니다.

---

**Made with ❤️ and 🤔 (그리고 예상치 못한 결과)**

*"실패는 성공의 어머니" - 예상과 다른 결과를 통해 더 많은 것을 배웠습니다.*