#!/bin/bash
# Compare 768 training progress against 512 baseline epoch-by-epoch.

LOG_768=/tmp/comicml_train_768.log
LOG_512=/tmp/comicml_train.log

# Extract val_px series (one per line, indexed by epoch)
extract() {
  grep -E "^epoch.*val_px" "$1" | sed -E 's/.*val_px=[ ]*([0-9.]+).*/\1/'
}

mapfile -t v512 < <(extract "$LOG_512")
mapfile -t v768 < <(extract "$LOG_768")

n768=${#v768[@]}
last_line=$(grep -E "^epoch.*val_px" "$LOG_768" | tail -1)
running=$(ps -p 17620 -o etime= 2>/dev/null | xargs)

best_512=$(printf '%s\n' "${v512[@]}" | sort -n | head -1)
best_768=$(printf '%s\n' "${v768[@]}" | sort -n | head -1)

echo "=== 768 training: epoch $n768/60   (elapsed ${running:-stopped}) ==="
echo "Last: $last_line"
echo
printf "%-7s  %-10s  %-10s  %-10s\n" "epoch" "512_val" "768_val" "delta"
echo "----------------------------------------------"
start=$((n768 - 8)); [ $start -lt 0 ] && start=0
for ((i=start; i<n768; i++)); do
  ep=$((i+1))
  a=${v512[i]:-"-"}
  b=${v768[i]:-"-"}
  if [[ "$a" != "-" && "$b" != "-" ]]; then
    delta=$(python3 -c "print(f'{float($b)-float($a):+.2f}')")
  else
    delta="-"
  fi
  printf "%-7s  %-10s  %-10s  %-10s\n" "$ep/60" "$a" "$b" "$delta"
done
echo
echo "Best so far:  512=$best_512   768=$best_768"
