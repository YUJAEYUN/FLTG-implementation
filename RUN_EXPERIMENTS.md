# 종합 실험 실행 가이드

## 🚀 빠른 실행 (권장)

```bash
cd /Users/smartnewbie/Desktop/FLTG

# 백그라운드로 실험 실행 (터미널 닫아도 계속 실행)
nohup bash run_focused_experiments.sh > experiment_output.log 2>&1 &

# 실행 상태 확인
tail -f experiment_output.log

# 또는 실시간으로 진행 상황 보기
watch -n 5 'ls -lh results/ | tail -20'
```

## 📊 실험 내용

### 30 epochs × 26개 실험 = 약 3-4시간 소요

1. **고강도 비잔틴 공격** (3개 비율 × 4개 방어 = 12 runs)
   - 30% Byzantine (6/20 clients)
   - 40% Byzantine (8/20 clients)
   - 50% Byzantine (10/20 clients) ← 논문의 핵심 주장 테스트!

2. **Non-IID 데이터** (2개 수준 × 4개 방어 = 8 runs)
   - Highly Non-IID (Dirichlet α=0.1)
   - Moderate Non-IID (Dirichlet α=0.5)

3. **다양한 공격 유형** (2개 공격 × 4개 방어 = 8 runs)
   - ROP (Relocated Orthogonal Perturbation)
   - IPM (Inner Product Manipulation)

4. **Baseline** (1 run)
   - 공격 없는 이상적 환경

## 🔍 실험 중 모니터링

### 진행 상황 확인
```bash
# 현재까지 완료된 실험 개수
ls results/*.log | wc -l

# 가장 최근 실험 결과
tail results/*.log | grep "Epoch 30"

# 특정 실험 실시간 보기
tail -f results/high_byz_0.5_fltg.log
```

### 예상 시간
- 실험 1개당: 약 5-10분
- 전체 26개: 약 3-4시간

## 📈 결과 분석

실험 완료 후:

```bash
cd /Users/smartnewbie/Desktop/FLTG
python3 analyze_focused_results.py
```

출력 예시:
```
===========================================
 FLTG COMPREHENSIVE EXPERIMENTAL RESULTS
===========================================

📊 Baseline (No Attack, 30 epochs): 98.73%

🔴 HIGH BYZANTINE RATIO EXPERIMENTS
Byzantine Ratio: 30% (6/20 clients)
Method          Accuracy     vs Baseline      Rank
------------------------------------------------------
Trimmed-Mean    98.20%       -0.53%p         1 ⭐
FLTG            98.10%       -0.63%p         2
Krum            97.90%       -0.83%p         3
FedAVG          97.50%       -1.23%p         4

Byzantine Ratio: 50% (10/20 clients)
Method          Accuracy     vs Baseline      Rank
------------------------------------------------------
FLTG            97.30%       -1.43%p         1 ⭐  ← 여기서 FLTG가 이겨야 함!
...
```

## 🎯 핵심 검증 포인트

### 논문의 주장을 검증하려면:

1. **50% Byzantine에서 FLTG가 1위?**
   ✅ Yes → 논문 주장 입증
   ❌ No → 구현 문제 or 논문 과장

2. **Non-IID 환경에서 FLTG 우위?**
   ✅ Yes → Non-IID aware weighting 효과 있음
   ❌ No → 메커니즘 재검토 필요

3. **다양한 공격에서 일관성?**
   ✅ Yes → 범용 방어 능력
   ❌ No → 특정 공격에만 효과적

## 💻 더 강력한 테스트 (선택사항)

시간이 더 있다면:

```bash
# CIFAR-10으로 테스트 (더 복잡한 데이터셋)
cd FL-Byzantine-Library
python3 main.py --dataset_name cifar10 --nn_name resnet20 \
  --num_client 20 --traitor 0.5 --attack rop \
  --aggr fltg --trials 1 --global_epoch 50 \
  --gpu_id -1 --bs 64
```

## 🛑 실험 중단

```bash
# 실행 중인 프로세스 찾기
ps aux | grep python3

# 종료
kill <PID>

# 또는 전체 Python 프로세스 종료 (주의!)
killall python3
```

## 📝 결과 저장

```bash
# 실험 완료 후 결과를 저장하고 커밋
git add results/ experiment_output.log COMPREHENSIVE_RESULTS.md
git commit -m "Add comprehensive experimental results (30 epochs, 26 scenarios)"
```

## ⚡ 빠른 테스트 (10 epochs)

시간이 부족하면:

```bash
# run_focused_experiments.sh에서 EPOCHS=30을 EPOCHS=10으로 변경
sed -i '' 's/EPOCHS=30/EPOCHS=10/g' run_focused_experiments.sh

# 실행 (약 1시간)
bash run_focused_experiments.sh
```

## 📊 예상 결과

**만약 논문이 맞다면:**
- 50% Byzantine: FLTG >> 다른 방법들
- Non-IID: FLTG가 상대적 강점
- 다양한 공격: FLTG가 안정적

**만약 우리 초기 실험처럼 나온다면:**
- 50% Byzantine: 여전히 FLTG가 약함
- → 구현 로직 재검토 필요
- → 논문 저자에게 문의 필요