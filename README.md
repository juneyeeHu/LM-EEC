# Robust Ego-Exo Correspondence with Long-Term Memory

[![NeurIPS 2025](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)](https://neurips.cc/)
[![Status](https://img.shields.io/badge/status-coming--soon-orange)](https://github.com/yourusername/Robust-Ego-Exo-Correspondence)

## Overview

This repository hosts the official code for our work:

**"Robust Ego-Exo Correspondence with Long-Term Memory"**

Establishing object-level correspondences between **egocentric** and **exocentric** views is crucial for AI assistants to provide precise and intuitive visual guidance. However, this task is challenging due to extreme viewpoint variations, occlusions, and the presence of numerous small objects. Existing methods, often adapted from video object segmentation approaches such as XSegTx and XView-XMem, struggle to handle these challenges effectively.

Recently, the Segment Anything Model 2 (SAM 2) has demonstrated strong generalization capabilities and impressive performance in video object segmentation. Nevertheless, directly applying SAM 2 to the ego-exo correspondence (EEC) task encounters key limitations, including suboptimal feature fusion across views and inefficient memory management for long video sequences.

To address these issues, we propose a novel EEC framework based on SAM 2 with **long-term memories**, introducing a **dual-memory system** and an **adaptive feature routing module inspired by Mixture-of-Experts (MoE)**. Specifically, our approach features:

1. a **Memory-View Mixture-of-Experts module** which consists of a dual-branch routing mechanism to adaptively assign contribution weights to each expert feature along both channel and spatial dimensions.  
2. a **dual-memory bank system** with a dedicated **compression strategy** to retain critical long-term information while eliminating redundancy. 

Extensive experiments on the challenging **EgoExo4D benchmark** demonstrate that our method, dubbed **LM-EEC**, establishes new state-of-the-art performance, significantly outperforming existing approaches and the SAM 2 baseline, and exhibiting strong generalization across diverse scenarios.


## Code Release

The code is **coming soon**.
