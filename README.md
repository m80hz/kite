# KITE: Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis

### [Paper](https://arxiv.org/abs/XXXX.XXXXX) | [Project Page](https://m80hz.github.io/kite/)

**IEEE International Conference on Robotics and Automation (ICRA), 2026**

[Mehdi Hosseinzadeh](https://m80hz.github.io/), [King Hang Wong](#), [Feras Dayoub](https://ferasdayoub.com/)

The Australian Institute for Machine Learning (AIML), Adelaide University, Australia

---

<p align="center">
  <img src="docs/kite_teaser.png" alt="KITE Overview" width="100%">
</p>

## Abstract

We present **KITE**, a *training-free*, keyframe-anchored, layout-grounded front-end that converts long robot-execution videos into compact, interpretable tokenized evidence for vision-language models (VLMs). KITE distills each trajectory into a small set of motion-salient keyframes with open-vocabulary detections and pairs each keyframe with a schematic bird's-eye-view (BEV) representation that encodes relative object layout, axes, timestamps, and detection confidence. These visual cues are serialized with robot-profile and scene-context tokens into a unified prompt, allowing the same front-end to support failure detection, identification, localization, explanation, and correction with an off-the-shelf VLM.

On the RoboFAC benchmark, KITE with Qwen2.5-VL substantially improves over vanilla Qwen2.5-VL in the training-free setting, with especially large gains on simulation failure detection, identification, and localization, while remaining competitive with a RoboFAC-tuned baseline. A small QLoRA fine-tune further improves explanation and correction quality.

## Code Release

> **🚧 Code coming soon.** We are preparing the codebase for public release. Stay tuned!

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hosseinzadeh2025kite,
  title     = {KITE: Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis},
  author    = {Hosseinzadeh, Mehdi and Wong, King Hang and Dayoub, Feras},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```
