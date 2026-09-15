#!/usr/bin/env bash
# Bounded single-GPU smoke; these settings are not a benchmark recipe.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
method="${1:-t2}"
environment="${2:-craftax_classic}"
case "$method" in
    t2|e3b|icm|rnd|e3b_rnd) ;;
    *) echo "Usage: $0 [t2|e3b|icm|rnd|e3b_rnd] [craftax_classic|trilab|quadlab]" >&2; exit 2 ;;
esac
case "$environment" in
    craftax_classic|trilab|quadlab) ;;
    *) echo "Unsupported smoke environment: $environment" >&2; exit 2 ;;
esac
export PUFFER_OMP_LIB="${PUFFER_OMP_LIB:--lgomp}"
binary="build/intrinsic_${environment}"
./build.sh "$environment" "$binary" --t2 --float
exec "./$binary" train \
    --intrinsic.method="$method" --intrinsic.intrinsic_only=1 \
    --t2.intrinsic_only=1 --base.seed=73 \
    --train.gpus=1 --train.total_timesteps=2097152 \
    --vec.total_agents=256 --vec.num_buffers=2 --vec.num_threads=4 \
    --train.horizon=64 --train.minibatch_size=8192 \
    --t2.wm_batch=256 --t2.wm_steps=8 --t2.reservoir=128
