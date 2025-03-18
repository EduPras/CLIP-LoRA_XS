# CLIP-LoRA_XS

Low-Rank Adaptation Extremely Small for Vision-Language Models

**Author**: [Eduardo Prasniewski](https://www.linkedin.com/in/edupras/)

This repository contains the implementation of **CLIP-LoRA-XS**, an improved method for fine-tuning large vision-language models (CLIP) using Low-Rank Adaptation (LoRA) with extremely small parameter counts. The work builds on [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA) and introduces [LoRA-XS](https://github.com/MohammadrezaBanaei/LoRA-XS), a novel technique to reduce trainable parameters while maintaining performance. The core implementation of **LoRA-XS** modifies the original LoRA method by introducing an intermediate matrix **R**, which is initialized with near-zero values.

<div align="center">
  
  
|       Template       | Rank | No Params (%) |
|:--------------------:|:----:|:-------------:|
|      **CLIP-LoRA**       |  2   |     0.12      |
|    **CLIP-LoRA_XS**    |  32  |     0.5       |
|    **CLIP-LoRA_XS**     |  45  |     0.10      |

_Table 1: Number of trainable parameters in comparison to full fine-tuning_

</div>

Although [LoRA-XS](https://github.com/MohammadrezaBanaei/LoRA-XS) provides an implementation using the Hugging Face library, this project uses the original [LoRA](https://github.com/microsoft/LoRA) implementation and only updates what is necessary.


<div align="center">
  
![image](https://github.com/user-attachments/assets/70d85df0-b3bc-45f4-9661-f7ef560018f4)
_Figure 1: CLIP-LoRA_XS Few-Shot performance vs. CLIP-LoRA_

</div>

The new trainable matrix **R** ∈ ℝ (r × r) is placed between the frozen matrices **A** ∈ ℝ (n × r) and **B** ∈ ℝ (r × n).  
Using Singular Value Decomposition (SVD) on the original weights **W** ∈ ℝ (n × n):

$$
W = U\Sigma V^T
$$

$$
A = U_r \Sigma_r \quad \text{and} \quad B = V_r^T
$$

The traditional *forward* pass of LoRA is defined as:

$$
h = xW + x\Delta W = xW + xAB
$$

where **A** and **B** are trainable. In LoRA-XS:

$$
h = xW + x\Delta W = xW + xARB
$$

**R** is initialized with values very close to zero, following a normal distribution:  
**R** ~ N(0, σ²) with **σ = 10⁻⁵**.

---

## References:
1. Zanella, M., & Ben Ayed, I. (2024). Low-rank few-shot adaptation of vision-language models. _In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 1593-1603)_.[https://arxiv.org/abs/2405.18541](1)
2. Bałazy, K., Banaei, M., Aberer, K., & Tabor, J. (2024). Lora-xs: Low-rank adaptation with extremely small number of parameters. _arXiv preprint arXiv:2405.17604_. [https://arxiv.org/abs/2405.17604](2)
