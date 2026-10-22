<div align="center">

# Generative Conformal Prediction with Optimized Coverage Allocation

**Minxing Zheng** · **Shixiang Zhu**
<br>
Carnegie Mellon University

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-102137?style=flat-square)](https://neurips.cc/virtual/2026/poster/149988)
[![arXiv](https://img.shields.io/badge/arXiv-2410.13735-b31b1b?style=flat-square)](https://arxiv.org/abs/2410.13735)
[![Interactive demo](https://img.shields.io/badge/Interactive_demo-live-007d70?style=flat-square)](https://minxing-zheng.github.io/orca/)
[![License: MIT](https://img.shields.io/badge/License-MIT-6b7280?style=flat-square)](LICENSE)

<a href="https://minxing-zheng.github.io/orca/">
  <img src="assets/demo-preview.png" alt="Interactive demo comparing Standard CP, PCP and ORCA on the same generated samples" width="100%">
</a>

<sub>Standard CP, PCP and ORCA on the same generated samples. <a href="https://minxing-zheng.github.io/orca/">Try the interactive demo</a>.</sub>

</div>

## Overview

In this paper, we propose **ORCA** (Optimized Ranking and Coverage Allocation), a method that turns samples from a generative model into prediction sets with guaranteed coverage. Instead of one radius for every sample, ORCA learns a separate radius for each density rank, which gives compact, disconnected sets that follow the shape of the distribution.

1. **Explore.** Draw samples from the generative model and rank them by local density, densest first.
2. **Optimize.** On an exploration split, learn one radius per rank: the smallest balls that still cover the target share of outcomes.
3. **Calibrate.** On a separate calibration split, scale every radius by one common factor, which gives finite-sample marginal coverage.

## Real data: NYC bike trips

<img src="assets/bike-comparison.jpg" alt="Prediction sets for a bike trip destination from ORCA, PCP, CHR and R2CCP" width="100%">

Prediction sets for the destination of one bike trip (red marker). ORCA and PCP return unions of balls around likely destinations, while CHR and R2CCP return a single rectangle. ORCA shrinks the balls around isolated samples, which gives tighter sets at the same coverage. Map data © OpenStreetMap contributors.

## Getting started

```bash
git clone https://github.com/Minxing-Zheng/ORCA.git
cd ORCA
pip install -e ".[notebook]"
jupyter notebook notebooks/orca_toy.ipynb
```

The [toy notebook](notebooks/orca_toy.ipynb) runs ORCA end to end on synthetic S-shape data: it fits a generative model, ranks the generated samples, optimizes the radii, calibrates them, and compares the resulting sets with PCP at a 90% coverage target. On 1,000 test points it gives:

| Method | Coverage | Set size |
| :--- | :---: | :---: |
| ORCA | 0.916 | **0.430** |
| PCP | 0.896 | 0.510 |

Set size is the average length of the prediction set, so smaller is better.

## Citation

```bibtex
@inproceedings{zheng2026orca,
  title     = {Generative Conformal Prediction with Optimized Coverage Allocation},
  author    = {Zheng, Minxing and Zhu, Shixiang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Released under the [MIT License](LICENSE). The original PCP-VCR implementation is kept at the [`pcp-vcr`](https://github.com/Minxing-Zheng/ORCA/tree/pcp-vcr) tag.
