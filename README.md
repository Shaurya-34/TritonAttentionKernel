# Triton Attention Kernel

A ladder of GPU kernels written from scratch in Triton: vector addition,
fused ReLU + dropout, fused softmax (including one real correctness bug
found and fixed), and a naive self-attention kernel validated against
`F.scaled_dot_product_attention` across several shapes.

The attention kernel loads all of K and V into SRAM at once, so it doesn't
scale to long sequences the way tiled, online-softmax kernels (FlashAttention)
do. That's named honestly as the next step, not implemented here.

## Run it

Open `TritonAttentionKernel.ipynb` in Colab, or any Jupyter environment with
a CUDA GPU, and run the cells top to bottom.

Full writeup: [Programming an attention kernel in Triton](https://sslog.dpdns.org/programming-an-attention-kernel-in-triton.html)
