#!/usr/bin/env python3
"""Compare current training run vs 768 regression baseline."""
import re, subprocess, sys
from pathlib import Path

LOG_CURR = Path("/tmp/comicml_train_502_120ep.log")  # current: 502-page 120-epoch
LOG_BASE = Path("/tmp/comicml_train_502.log")        # baseline: 502-page 60-epoch
PID_CURR = 34109
LABEL_CURR = "120 ep"
LABEL_BASE = "60 ep"

PAT = re.compile(r"^epoch\s+(\d+)/\d+.*val_px=\s*([\d.]+)")

def extract(path):
    if not path.exists(): return []
    return [(int(m.group(1)), float(m.group(2)))
            for line in path.read_text().splitlines()
            if (m := PAT.match(line))]

vbase = dict(extract(LOG_BASE))
vcurr = dict(extract(LOG_CURR))

if not vcurr:
    print("No epochs yet."); sys.exit(0)

n = max(vcurr)
running = subprocess.run(["ps","-p",str(PID_CURR),"-o","etime="],
                        capture_output=True, text=True).stdout.strip()
status = f"running ({running})" if running else "STOPPED"

best_base = min(vbase.values()) if vbase else float("inf")
best_curr = min(vcurr.values())
best_curr_ep = min(vcurr, key=vcurr.get)

print(f"=== {LABEL_CURR}: epoch {n}/60  [{status}] ===")
print(f"{'epoch':>6s}  {LABEL_BASE:>10s}  {LABEL_CURR:>10s}  {'delta':>8s}  {'best?':>6s}")
print("-" * 50)
start = max(1, n - 9)
for ep in range(start, n + 1):
    a = vbase.get(ep)
    b = vcurr.get(ep)
    delta = f"{b-a:+.2f}" if (a is not None and b is not None) else "-"
    a_str = f"{a:.2f}" if a is not None else "-"
    b_str = f"{b:.2f}" if b is not None else "-"
    star = "*" if b == best_curr else ""
    print(f"{ep:>4d}/60  {a_str:>10s}  {b_str:>10s}  {delta:>8s}  {star:>6s}")

print()
print(f"Best {LABEL_BASE} (60 ep): {best_base:6.2f} px")
print(f"Best {LABEL_CURR} so far:  {best_curr:6.2f} px (epoch {best_curr_ep})")
if best_curr < best_base:
    print(f"{LABEL_CURR} beats {LABEL_BASE} by {best_base-best_curr:.2f} px")
else:
    print(f"{LABEL_CURR} needs to drop {best_curr-best_base:.2f} more px to beat {LABEL_BASE}")
