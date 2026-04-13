<p align="center">
  <h1 align="center">KITE [ICRA 2026]</h1>
  <h3 align="center">Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis</h3>
 <p align="center">
    <a href="https://m80hz.github.io/" target="_blank">Mehdi Hosseinzadeh</a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="#">King Hang Wong</a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ferasdayoub.com/" target="_blank">Feras Dayoub</a>
  </p>
 
 <h5 align="center">The Australian Institute for Machine Learning (AIML), Adelaide University, Australia</h5>

  <p align="center">
    <a href="https://m80hz.github.io/kite/" target="_blank">
      <img src="https://img.shields.io/badge/🌐_Project_Page-007ACC?style=for-the-badge" alt="Project Page" />
    </a>
    &nbsp;
    <a href="https://arxiv.org/abs/2604.07034" target="_blank">
      <img src="https://img.shields.io/badge/📄_Paper-B31B1B?style=for-the-badge" alt="arXiv" />
    </a>
    &nbsp;
    <a href="https://github.com/m80hz/kite" target="_blank">
      <img src="https://img.shields.io/badge/💻_Code-181717?style=for-the-badge&amp;logo=github" alt="GitHub" />
    </a>
    &nbsp;
    <a href="#demo">
      <img src="https://img.shields.io/badge/🎮_Demo-8A2BE2?style=for-the-badge" alt="Demo" />
    </a>
    &nbsp;
    <a href="https://huggingface.co/m80hz/KITE-7B-Instruct" target="_blank">
      <img src="https://img.shields.io/badge/🤗_Model-FFD21E?style=for-the-badge" alt="HuggingFace" />
    </a>
  </p>
</p>

<p align="center">
  <img src="docs/kite_teaser.png" alt="KITE Overview" width="100%">
</p>

---


> **🚧 ToDo**
>
> - [ ] QLoRA fine-tuning scripts & training recipes
> - [x] [Pre-trained weights](https://huggingface.co/m80hz/KITE-7B-Instruct)
>

## Quick Start

1. Clone with submodules (GroundingDINO, Depth-Anything-V2):
```bash
git clone --recursive https://github.com/m80hz/kite.git
cd kite
```
If you already cloned without `--recursive`, run: `git submodule update --init`

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Start a VLM server. A convenience launcher for [vLLM](https://github.com/vllm-project/vllm) is included:
```bash
# Pass an HF repo id or a local model directory
bash scripts/run_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct
# Or use our fine-tuned model:
bash scripts/run_vllm.sh m80hz/KITE-7B-Instruct
```
See `scripts/run_vllm.sh` for optional env vars (`TP`, `PORT`, `GPU_MEM_UTIL`, etc.).

4. Ensure you have the **dataset_folder** and **test_file** JSON (e.g. RoboFAC dataset).

> **Note on RoboFAC annotations:** The original RoboFAC dataset has some mismatched / incorrect file paths in certain annotation files. See [MINT-SJTU/RoboFAC#2](https://github.com/MINT-SJTU/RoboFAC/issues/2) for details and how to fix them before running evaluation.

5. Run evaluation (full pipeline with keyframes, 2D/3D scene context, BEV, and narrative):
```bash
python -m kite.cli \
  --dataset_folder ./datasets/robofac/simulation_data \
  --test_file ./datasets/robofac/test_qa_sim/test_detect_identify_locate.json \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --model_url http://127.0.0.1:8000/v1 \
  --out_dir ./outputs/kite_run
```

Outputs:
- Per-task JSONs under `<out_dir>/` and a consolidated `stats_data.json` with MCQ scores and descriptive metrics.
- Storyboard images: `*_storyboard_all_keyframes.jpg` and, when enabled, `*_storyboard_bev.jpg` for BEV alignment.
- Optional final narrative text per video: `*_final_narrative.txt`.

## Demo

Launch the interactive Gradio app to explore the full pipeline on a single video:

```
python app.py
```

The demo lets you step through keyframes, view 2D detections, per-keyframe BEV maps, colored depth, an interactive Plotly 3D point-cloud viewer, and run QA queries against the VLM. A few example video sequences are included under `examples/` to get started quickly.


## Notes

- Training-free by default; depth/3D are optional but recommended for BEV and spatial grounding.
- Any model exposing an OpenAI-compatible **/chat/completions** endpoint works out of the box.

## Evaluation

We score MCQs with normalized string containment and compute non-LLM text similarity metrics for descriptive QAs. See `docs/Evaluation.md` for details.

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
