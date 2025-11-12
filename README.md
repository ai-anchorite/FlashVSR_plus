This repo has a [Pinokio](https://pinokio.co/) 1-click-installer available [here:](https://pinokio.co/item.html?uri=https%3A%2F%2Fgithub.com%2Fai-anchorite%2FFlashVSR_plus_pinokio&parent_frame=&theme=null)

Installing outside of Pinokio will require ffmepg on PATH and self-installed pytorch. Torch install info can be seen [here:](https://github.com/ai-anchorite/FlashVSR_plus_pinokio/blob/main/torch.js)

Forked from: [lihaoyun6/FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus)

Original Project: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR )

# FlashVSR: Efficient & High-Quality Video Super-Resolution
A user-friendly fork of FlashVSR, enhanced and packaged for the Pinokio community. This version is optimized for consumer-grade hardware, enabling users to access powerful video and image upscaling without the demanding VRAM requirements of the original project.

<summary>Tab Screenshots</summary>
<table>
  <tr>
    <td valign="top">
      <a href="https://raw.githubusercontent.com/ai-anchorite/FlashVSR_plus/main/inputs/flashvsr_video_screen.png">
        <img src="https://raw.githubusercontent.com/ai-anchorite/FlashVSR_plus/main/inputs/flashvsr_video_screen.png" alt="flashvsr_video_screen" width="250">
      </a>
    </td>
    <td valign="top">
      <a href="https://raw.githubusercontent.com/ai-anchorite/FlashVSR_plus/main/inputs/flashvsr_img_screen.png">
        <img src="https://raw.githubusercontent.com/ai-anchorite/FlashVSR_plus/main/inputs/flashvsr_img_screen.png" alt="flashvsr_img_screen" width="250">
      </a>
    </td>
    <td valign="top">
      <a href="https://raw.githubusercontent.com/ai-anchorite/FlashVSR_plus/main/inputs/flashvsr_toolbox_screen.png">
        <img src="https://raw.githubusercontent.com/ai-anchorite/FlashVSR_plus/main/inputs/flashvsr_toolbox_screen.png" alt="flashvsr_toolbox_screen" width="250">
      </a>
    </td>
  </tr>
</table>

## Project Background
FlashVSR was generously released via [OpenImagingLab](https://github.com/OpenImagingLab/FlashVSR) to the open-source community. Their team's README is detailed below!

This project builds upon this excellent fork [lihaoyun6/FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus), which introduced several key optimizations to the original FlashVSR project

### Features from Upstream Fork
The FlashVSR_plus fork laid the groundwork with several notable enhancements, including:
* Replaced Block-Sparse-Attention with Sparse_SageAttention.
* Added DiT tiling and other memory optimizations to significantly reduce VRAM requirements.
* Implemented the initial Gradio user interface.

## Enhancements in This Version
This fork further refines the user experience and expands functionality with a focus on quality-of-life improvements and adds several new tools.

* Enhanced Gradio UI: The interface has been redesigned for a more intuitive workflow, including dedicated tabs for additional tasks.
* Improved Memory Management and optimizations and internal fixes to ensure smooth operation on consumer hardware.
* Chunked Video Processing: Easily upscale longer videos without running into memory limitations.
* Image Upscaling: A new feature that brings the power of FlashVSR to still images.
* Post-Processing Toolbox: A suite of useful post-processing tools for RIFE frame interpolation, seamless video looping, and extra compression/export options.


# Original Project Details below: 
# ⚡ FlashVSR

**Towards Real-Time Diffusion-Based Streaming Video Super-Resolution**

**Authors:** Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue

<a href='http://zhuang2002.github.io/FlashVSR'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/JunhaoZhuang/FlashVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1)-blue"></a> &nbsp;
<a href="https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1.1)-blue"></a> &nbsp;
<a href="https://huggingface.co/datasets/JunhaoZhuang/VSR-120K"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a> &nbsp;
<a href="https://arxiv.org/abs/2510.12747"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

**Your star means a lot for us to develop this project!** :star:

---

### 🌟 Abstract

Diffusion models have recently advanced video restoration, but applying them to real-world video super-resolution (VSR) remains challenging due to high latency, prohibitive computation, and poor generalization to ultra-high resolutions. Our goal in this work is to make diffusion-based VSR practical by achieving **efficiency, scalability, and real-time performance**. To this end, we propose **FlashVSR**, the first diffusion-based one-step streaming framework towards real-time VSR. **FlashVSR runs at ∼17 FPS for 768 × 1408 videos on a single A100 GPU** by combining three complementary innovations: (i) a train-friendly three-stage distillation pipeline that enables streaming super-resolution, (ii) locality-constrained sparse attention that cuts redundant computation while bridging the train–test resolution gap, and (iii) a tiny conditional decoder that accelerates reconstruction without sacrificing quality. To support large-scale training, we also construct **VSR-120K**, a new dataset with 120k videos and 180k images. Extensive experiments show that FlashVSR scales reliably to ultra-high resolutions and achieves **state-of-the-art performance with up to ∼12× speedup** over prior one-step diffusion VSR models.

---
### 🛠️ Method

The overview of **FlashVSR**. This framework features:

* **Three-Stage Distillation Pipeline** for streaming VSR training.
* **Locality-Constrained Sparse Attention** to cut redundant computation and bridge the train–test resolution gap.
* **Tiny Conditional Decoder** for efficient, high-quality reconstruction.
* **VSR-120K Dataset** consisting of **120k videos** and **180k images**, supports joint training on both images and videos.

<img src="./examples/WanVSR/assets/flowchart.jpg" width="1000" />

---

### 🤗 Feedback & Support

We welcome feedback and issues. Thank you for trying **FlashVSR**!

---

### 📄 Acknowledgments

We gratefully acknowledge the following open-source projects:

* **DiffSynth Studio** — [https://github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* **Block-Sparse-Attention** — [https://github.com/mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
* **taehv** — [https://github.com/madebyollin/taehv](https://github.com/madebyollin/taehv)

---

### 📞 Contact

* **Junhao Zhuang**
  Email: [zhuangjh23@mails.tsinghua.edu.cn](mailto:zhuangjh23@mails.tsinghua.edu.cn)

---

### 📜 Citation

```bibtex
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}
```
