#!/usr/bin/env bash
# Script to check GPU availability across SLURM partitions

echo "=========================================="
echo "SLURM GPU Information"
echo "=========================================="
echo ""

# Show partition summary with GPU info
echo "=== Partition Summary ==="
sinfo -o "%15P %5D %10F %12G %25N" | head -30

echo ""
echo "=== VEU Partition Details ==="
echo ""

# Show detailed info for each veu node
for node in veuc01 veuc05 veuc09 veuc10 veuc11 veuc12; do
    echo "--- Node: $node ---"
    scontrol show node $node | grep -E "State=|Gres=|CPUTot=|RealMemory=|AllocTRES=" | sed 's/^/  /'
    echo ""
done

echo "=== Legend ==="
echo "State: IDLE (available), ALLOCATED (in use), MIXED (partially used), DOWN (offline)"
echo "Gres: Generic Resources (gpu:N = number of GPUs)"
echo ""

echo "=== Quick GPU Check ==="
echo "Available GPUs in veu partition:"
sinfo -p veu -o "%n %G %a %T" | column -t

echo ""
echo "=========================================="
echo "To request specific GPUs, use:"
echo "  #SBATCH --gres=gpu:2"
echo "=========================================="
