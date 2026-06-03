#!/usr/bin/env bash
# GPU Monitor — samples nvidia-smi every INTERVAL seconds to a CSV file.
# Usage: gpu_monitor.sh <output_csv> [interval_seconds]
# Stop with: kill $PID  or  touch <output_csv>.stop

OUTPUT="${1:?Usage: gpu_monitor.sh <output.csv> [interval]}"
INTERVAL="${2:-1}"

echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,power_w,temp_c,pcie_tx_kbps,pcie_rx_kbps" > "$OUTPUT"

while true; do
    [ -f "${OUTPUT}.stop" ] && break
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    DATA=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$DATA" ]; then
        echo "${TS},${DATA}" >> "$OUTPUT"
    fi
    sleep "$INTERVAL"
done
