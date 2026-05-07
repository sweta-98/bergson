"""Stream stdin → stdout, prefixing each line with HH:MM:SS + monotonic-elapsed.

Used as a wrapper for run_comparison.py so we get truly real-time per-line
timestamps. awk's stdin buffering hid early-phase timings; this avoids that.
"""

import sys
import time

start = time.monotonic()
for line in sys.stdin:
    elapsed = time.monotonic() - start
    sys.stdout.write(f"[{time.strftime('%H:%M:%S')}+{elapsed:7.2f}s] {line}")
    sys.stdout.flush()
