# ShortV: Freezing Visual Tokens in Ineffective Layers of Multimodal Large Language Models

Anonymous Research Repository | Technical Documentation

## Overview
This repository provides a standardized pipeline for evaluating ShortV models on structured visual understanding tasks. The implementation preserves anonymity in compliance with double-blind review requirements.

## Environment Setup

### Conda Environment
```bash
conda create -n shortv python=3.10 -y
conda activate shortv
pip install --upgrade pip  # Enable PEP 660 support
pip install -e .
```

## Evaluation
```bash
bash eval.sh
```
