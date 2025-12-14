# FLTG 실험 브랜치

이 브랜치는 FLTG(Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting) 알고리즘을 직접 구현하고, 논문 속 핵심 시나리오들을 재현·확장하기 위한 전용 실험 공간이다. 메인 브랜치 설명 대신, 여기에서 다룬 실험과 산출물을 중심으로 정리한다.

---

## 폴더 요약

- `1-implementation/` – FLTG 코드를 실제로 넣어 둔 곳입니다. 어떤 파일을 고쳤고 왜 필요한지 메모도 함께 들어 있습니다.
- `2-experiments/` – 실험을 돌리는 스크립트 모음입니다. MNIST, CIFAR10, 편향 데이터 생성 도구까지 전부 이 폴더에서 실행합니다.
- `3-results/` – 실험을 돌리고 남은 로그와 요약 파일을 모아둔 폴더입니다.
- `4-analysis/` – 실험 결과를 해석하고 배운 점을 정리한 문서들이 여기에 있습니다. 개선 아이디어도 함께 적어 두었습니다.
- `5-visualization/` – 실험 로그를 그래프로 바꿔 주는 플롯 스크립트가 들어 있습니다. 결과를 보고서나 발표 자료로 쓰기 쉽게 만들어 줍니다.
- `docs/` – 전체 프로젝트 보고서와 발표 자료, 참고 문헌 목록이 모여 있는 문서 폴더입니다.

---

## 구현 범위

- `FL-Byzantine-Library/Aggregators/fltg.py`에 5단계 방어 메커니즘을 추가하고, 매퍼·파라미터 파일을 통해 `--aggr fltg` 플래그로 선택 가능하도록 등록했다 (`1-implementation/implementation_notes.md:5`).
- 기존 라이브러리 코드를 건드리지 않고 실험 자동화를 위해 외부 쉘/파이썬 스크립트를 별도로 두었다.

---

## 주요 실험 세트

### 1. MNIST 극한 Non-IID 스트레스 테스트
- 기본 설정: MNIST, MNISTNET, 클라이언트 100명, 글로벌 에폭 80, 배치 64, CPU 모드 (`2-experiments/mnist/run_extreme_experiments.sh:18-23`).
- 시나리오
  - Dirichlet α ∈ {0.01, 0.1} 와 Byzantine 비율 {0.3, 0.5} 조합 (`2-experiments/mnist/run_extreme_experiments.sh:51`, `2-experiments/mnist/run_extreme_experiments.sh:56`).
  - 정렬·분할 기반 Class Imbalance(클라이언트당 2개의 숫자 클래스) (`2-experiments/mnist/run_extreme_experiments.sh:104-128`).
  - ROP·IPM 공격 각각에 대한 비교 (`2-experiments/mnist/run_extreme_experiments.sh:144`).
- 모든 시나리오에서 FedAVG, Krum, Trimmed-Mean, FLTG를 순차 실행하며 로그를 동일 디렉터리에 적재한다 (`2-experiments/mnist/run_extreme_experiments.sh:61`, `2-experiments/mnist/run_extreme_experiments.sh:85`).
- 총 21개의 조합을 자동 실행하도록 구성되어 있으며, 예상 소요 시간은 약 12시간이다 (`2-experiments/mnist/run_extreme_experiments.sh:183`, `2-experiments/mnist/run_extreme_experiments.sh:189`).

### 2. MNIST Figure 3 재현(루트 데이터 편향)
- Dirichlet 기반 데이터 분할과 Root bias JSON을 인자로 사용하며, 편향도 {0.1, 0.5, 0.8}, Byzantine 비율 {0.2, 0.5, 0.8, 0.95}, 공격은 Min-Max를 그대로 따른다 (`2-experiments/mnist/run_fig3_mnist.py:95-99`).
- 기본값이 CIFAR10/ResNet-20으로 설정돼 있으나, MNIST 재현 시 `--dataset mnist --model mnistnet --clients 100 --epochs 20` 처럼 값을 덮어써 사용한다 (`2-experiments/mnist/run_fig3_mnist.py:88-93`).
- 비교 대상은 FL-Trust와 FLTG 두 가지이다 (`2-experiments/mnist/run_fig3_mnist.py:97`).

### 3. MNIST Figure 4 재현(Dirichlet α 스윕)
- α ∈ {0.1, 0.5, 1.0}, Byzantine 비율 {0.2, 0.5, 0.8, 0.9, 0.95}, 공격은 ROP로 고정했다 (`2-experiments/mnist/run_fig4_mnist.py:90-100`).
- 동일하게 FL-Trust와 FLTG를 비교 대상 aggregator로 두었으며, 각 α마다 베이스라인(IID, 공격 없음)도 함께 기록한다 (`2-experiments/mnist/run_fig4_mnist.py:66`, `2-experiments/mnist/run_fig4_mnist.py:86`).

### 4. CIFAR-10 Figure 3 확장
- ResNet-20, 클라이언트 80명, 50 에폭, Bias 레벨 {0.1, 0.5, 0.8} 설정을 사용하며 공격은 Min-Max로 고정했다 (`2-experiments/cifar10/new_run_fig3_cifar.py:86-97`).
- MNIST 버전과 동일하게 FL-Trust vs FLTG만 비교한다 (`2-experiments/cifar10/new_run_fig3_cifar.py:95`).

---

## 데이터 생성 유틸

- `mnist_bias_utils.py`는 Dirichlet 분할과 Root bias 인덱스를 생성해 JSON으로 저장하며, `mnist_bias_configs` 폴더를 자동 생성한다 (`2-experiments/utils/mnist_bias_utils.py:21`, `2-experiments/utils/mnist_bias_utils.py:37`, `2-experiments/utils/mnist_bias_utils.py:69-79`).
- CIFAR-10도 동일한 방식으로 `new_cifar_bias_utils.py`를 통해 구성할 수 있다 (`2-experiments/cifar10/new_run_fig3_cifar.py:99`).

---

## 결과 하이라이트

- 극한 Non-IID 8개 시나리오에서 FLTG가 4회 승리(50%), FedAVG 2회, Krum 2회, Trimmed-Mean 0회로 집계되었다 (`3-results/extreme_noniid/EXTREME_RESULTS_SUMMARY.txt:4-11`).
- Ultra Extreme Non-IID(α=0.01) + 50% Byzantine 환경에서는 FLTG가 10.7%→75.3%까지 회복하며 FedAVG(21.6%→66.6%)보다 8.7%p 높았다 (`README.md:193-194`, `README.md:288`).
- Class Imbalance + 50% Byzantine 실험에서는 FLTG가 31.4%→96.0%까지 상승한 반면, FedAVG·Trimmed-Mean는 10%대에 머물렀다 (`README.md:201-202`, `README.md:289`).
- ROP·IPM 공격 모두에서 50% Byzantine 시 FLTG가 각각 95.3%, 96.2%로 FedAVG 대비 7–11%p 우위를 보였다 (`README.md:210`, `README.md:216`, `README.md:290-291`).
- 30% Byzantine 수준에서는 Krum이나 FedAVG가 우세한 경우도 있어, FLTG의 이점이 극단적 조건에 집중돼 있음을 확인했다 (`README.md:223-225`).
- IID에 가까운 조건에서는 네 방법 모두 98%대 정확도로 수렴하며, FLTG는 초기 1 에폭 수렴 속도가 느린 편이다 (`4-analysis/ANALYSIS.md:6-15`).

---

## 실행 방법

1. (1회) `SETUP.md`에 맞춰 FL-Byzantine-Library 의존성을 설치한다.
2. 극한 Non-IID 실험  
   ```bash
   bash 2-experiments/mnist/run_extreme_experiments.sh
