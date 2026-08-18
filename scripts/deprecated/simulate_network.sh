#!/bin/bash
# =============================================================================
# DEPRECATED (2026-08-18): retired per the Topic-10 code walkthrough (J12,
# MyDocs/registration/CODE_ASSERTION_2026-08.md). The campaign design
# (MyDocs/PUBLICATION.md) has no tc-based network-emulation arm: pilot phases
# never ran it, and Phase-3 interconnect work uses REAL RDMA/RoCE hardware, not
# netem. Kept only so any historical reference stays auditable. Do NOT run
# against campaign data. See scripts/deprecated/README.md.
# =============================================================================
# Network Emulation Script for CAGE Framework
# Uses Linux Traffic Control (tc) to simulate HPC network conditions.
# Intended to be run inside Docker containers (privileged mode required).

set -euo pipefail
# shellcheck source=scripts/lib/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/_common.sh"

# Default settings
INTERFACE="${INTERFACE:-eth0}"
# HPC Interconnect (InfiniBand simulation): 100Gbps, 0.05ms latency
DELAY="${DELAY:-0.05ms}"
RATE="${RATE:-100gbit}"
JITTER="${JITTER:-0.01ms}"
LOSS="${LOSS:-0%}"

# Check for tc
require_cmd tc "ensure iproute2 is installed"

printf 'Setting up network simulation on %s\n' "$INTERFACE"
printf '  Delay: %s +/- %s\n' "$DELAY" "$JITTER"
printf '  Rate:  %s\n' "$RATE"
printf '  Loss:  %s\n' "$LOSS"

# Clear existing rules
tc qdisc del dev "$INTERFACE" root 2> /dev/null || true

# Add root qdisc (Hierarchical Token Bucket)
tc qdisc add dev "$INTERFACE" root handle 1: htb default 11

# Add class with rate limit
tc class add dev "$INTERFACE" parent 1: classid 1:1 htb rate "$RATE"

# Add NetEm qdisc for delay and loss
tc qdisc add dev "$INTERFACE" parent 1:1 handle 10: netem delay "$DELAY" "$JITTER" loss "$LOSS"

echo "Network simulation applied."
tc qdisc show dev "$INTERFACE"
