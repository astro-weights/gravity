#!/bin/sh
# Merge a gravity PR only after all its checks pass: scripts/merge.sh <PR number>
# (GitHub Free cannot require checks on a private repo; see .github/workflows/gate.yml.)
set -eu
n="${1:?usage: scripts/merge.sh <PR number>}"
gh pr checks "$n" --watch --fail-fast
gh pr merge "$n" --merge
