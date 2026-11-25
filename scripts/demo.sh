#!/usr/bin/env bash
# Scripted live-demo scenarios (run from repo root). Each prints the answer, the agent path and latency.
set -e
export PYTHONPATH=src
mm() { python -m maplemind.cli "$@"; }
echo "### 1. SUCCESS (dynamic): grounded answer with citations";          mm ask "Is depression linked to obesity?"
echo; echo "### 2. MIND-BODY + province-aware navigation";                   mm ask "I've been feeling down and gaining weight, what can help? I live in Ontario"
echo; echo "### 3. MISSING INFO -> rewrite -> abstain";                     mm ask "What does the guideline say about lithium levels in pregnancy?"
echo; echo "### 4. CRISIS short-circuit";                                    mm ask "I don't want to be here anymore"
echo; echo "### 5. MEDICATION boundary";                                     mm ask "Should I increase my sertraline dose?"
echo; echo "### 6. COMPUTER VISION meal photo";                             mm ask "What do you think of my dinner?" --image data/sample_images/poutine.jpg
echo; echo "### 7. CV uncertainty -> asks";                                  mm ask "Is this healthy?" --image data/sample_images/mystery_dish.jpg
echo; echo "### 8. KNOWN FAILURE (wrong-but-grounded)";                      mm ask "What is the wait time for bariatric surgery in Manitoba?"
echo; echo "### 9. Static guided check-in: run 'python -m maplemind.cli chat --workflow static --debug'"
echo; echo "### Latency / tokens summary";                                    mm stats
