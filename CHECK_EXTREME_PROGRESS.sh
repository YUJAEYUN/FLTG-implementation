#!/bin/bash

echo "======================================"
echo "🔥 극단적 Non-IID 실험 진행 상황"
echo "======================================"
echo ""

TOTAL=21
COMPLETED=$(ls results/*.log 2>/dev/null | wc -l | tr -d ' ')

echo "완료: $COMPLETED / $TOTAL 실험"

if [ $COMPLETED -gt 0 ]; then
    echo ""
    echo "최근 3개 결과 (Final epoch):"
    for file in $(ls -t results/*.log | head -3); do
        name=$(basename $file .log)
        last_line=$(grep "Epoch 20" $file | tail -1)
        if [ ! -z "$last_line" ]; then
            acc=$(echo $last_line | grep -o "Accuracy [0-9.]*" | awk '{print $2}')
            echo "  ${name}: ${acc}%"
        fi
    done
fi

echo ""
PROGRESS=$((COMPLETED * 100 / TOTAL))
echo "진행률: $PROGRESS%"
echo -n "["
for i in $(seq 1 42); do
    if [ $((i * 100 / 42)) -le $PROGRESS ]; then
        echo -n "="
    else
        echo -n " "
    fi
done
echo "]"

echo ""
if [ $COMPLETED -eq $TOTAL ]; then
    echo "✅ 모든 실험 완료!"
    echo ""
    echo "🔍 결과 분석:"
    echo "  python3 analyze_extreme_results.py"
else
    REMAINING=$((TOTAL - COMPLETED))
    EST_MIN=$((REMAINING * 5))
    echo "⏳ $REMAINING개 남음 (약 ${EST_MIN}분)"
    echo ""
    echo "실시간 로그:"
    echo "  tail -f extreme_output.log"
fi
echo ""