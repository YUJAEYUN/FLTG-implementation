# FLTG 구현 설명

## 수정된 파일

### 1. Aggregators/fltg.py (신규)
- FLTG 알고리즘의 핵심 구현
- 5단계 방어 메커니즘 포함

### 2. mapper.py
- FLTG를 aggregator 레지스트리에 등록
- `get_aggregator()` 함수에 'fltg' 케이스 추가

### 3. parameters.py
- 누락된 파라미터 추가
- FLTG 관련 설정 추가

## 주요 구현 내용

자세한 내용은 상위 폴더의 README.md를 참고하세요.
