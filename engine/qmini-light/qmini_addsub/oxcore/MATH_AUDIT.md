# OxCore mathematical audit and implementation gates

This document separates algebra that can be implemented now from ideas that
need a corrected derivation or a measured training ablation. Formula numbers
refer to the OxCore proposal; unverified speed/quality claims are not accepted
as engineering requirements.

## Formula 1 — Mirror-63 and deferred split accumulator

**Status: implemented and exhaustively tested for the scalar code mapping, row-wise linear projections, and causal depthwise convolution.**

For each triplet (q_j \in \{-2,-1,0,1,2\}),

[
v=25q_0+5q_1+q_2,\qquad -62\leq v\leq 62.
]

Balanced-base-5 representation is unique. One byte stores octave, sign and
magnitude. Across the two octave values, there are 250 valid encodings and
six unused byte patterns:
`0x3f, 0x40, 0x7f, 0xbf, 0xc0, 0xff`.
The implementation assigns `0x3f` as an explicit all-zero block skip marker
and rejects the other five. Zero uses only the non-negative sign encoding;
this canonicalization is necessary for the six-code count.

For (w_j=\tfrac12 q_j 2^o), the kernel accumulates
[
N=\sum_j q_j x_j 2^o
]
without an intermediate right shift. It returns integer (N); the caller
forms (\alpha_w\alpha_xN/2) once. This removes truncation caused by
per-term integer halving. It does **not** prove every later floating-point
conversion or quantization step is unbiased. The native linear projection
returns one numerator for every batch/output pair; partial input triplets are
zero-padded. The causal depthwise convolution uses per-channel weights,
tap-zero-oldest ordering and zero left padding, returning an int64 numerator
for each timestep/channel. Both operators validate all packed weight bytes
and conservative int64 accumulator bounds before writing outputs. These
implementations are portable scalar C++ kernels, not optimized SIMD.

## Formula 2 — FWHT incoherence transform

**Status: integer unnormalized FWHT implemented; random-sign/outlier transform
and model reparameterization not yet integrated.**

The native primitive computes the unnormalized Walsh-Hadamard transform
using only pairwise additions and subtractions:
[
H_1=[1],\quad
H_{2m}=\begin{pmatrix}H_m&H_m\\H_m&-H_m\end{pmatrix}.
]
It satisfies (H_nH_n^T=nI), so applying it twice returns (nx). An
orthonormal transform additionally needs a factor (1/\sqrt n), which is
not performed by the integer primitive; a scale may be deferred and tracked
separately.

For a fixed vector and random diagonal signs, the stated
(\ell_\infty) concentration bound is a high-probability statement over
the random signs, not a deterministic bound for every sign vector. Exact
linear equivalence requires the inverse transform and matching reparameterized
weights. Quantizing after the transform changes the function and must be
evaluated as an approximation.

## Formula 3 — Hysteresis and Tequila STE

**Status: deferred; no training autograd has been added.**

The proposed mode selector is stateful. Its octave state, thresholds, tie
rules and RNG (if randomized) must be checkpointed to make resume exact.
A state transition at a threshold is not an ordinary smooth derivative.

The proposed Tequila term adds
(\lambda_T\sum_{j\in\mathcal D_i}w_{ij}) to each output. Its claimed
gradient only follows when the dead-zone set is treated as fixed during the
backward pass. The extra forward bias can change model outputs; it needs
finite-difference/gradient checks and matched ablation against the current
scaled STE before it can be enabled. The damping term also needs an explicit
definition of quantizer scale, clipping range and normalization.

## Formula 4 — mHC, nGPT and phantom gradients

**Status: 4x4 Sinkhorn, L2 normalization and a Q15 diagonal SSM recurrence
primitive implemented; residual-stream mixing and autograd remain deferred.**

For a positive matrix, alternating row/column normalization approximates the
Birkhoff polytope. A nonnegative doubly-stochastic matrix has spectral norm at
most one. Twenty iterations are not a proof of exact convergence; this
implementation accepts a requested iteration count and tests approximate
row/column sums.

The native normalization primitive computes (u/(\|u\|_2+\epsilon)).
The proposed residual interpolation followed by normalization places the
result on the unit sphere, but is not by itself an exact geodesic/Slerp update.

The native SSM cache primitive accepts precomputed Q15 decay coefficients
(a_t in [0, 32768]) and int32 injection values:
[
s_t=\\operatorname{round}_{\\mathrm{even}}(a_t s_{t-1}/2^{15})+b_t.
]
Its scaling routine decomposes the coefficient into binary shifts and sums,
then applies explicit nearest-even rounding; a conservative range bound is
checked before output writes. It returns all token states and the final cache.
This is only the diagonal recurrence/update kernel. It does not compute the
model's drive/decay/gate projections, tanh/sigmoid lookup tables, the
(1-a_t)drive injection, output projection or residual gating; those tensors
must still be supplied by a future integer operator pipeline. The floating
PyTorch state-space path has not been replaced and numerical parity against a
trained checkpoint has not been established.

Saving only the last (k) recurrent steps is truncated backpropagation through
time: activation memory is (O(k)), and the gradient is biased by omitted
earlier steps. It is not an (O(1))-memory exact implicit gradient. A true
implicit method would solve a linear system involving (I-J_F^T) at a
converged fixed point and requires contraction/stability checks plus a
validated vector-Jacobian product. No 8x–16x memory claim is made.

## Formula 5 — DeltaNet / WY memory

**Status: deferred pending orientation-correct reference implementation.**

Under the convention that (S\in\mathbb R^{d_v\times d_k}) maps a key to a
value by (Sk\approx v), gradient descent on
(\tfrac12\|Sk-v\|_2^2) gives the rank-one correction
[
S_t=\alpha_tS_{t-1}
+\eta_t(v_t-S_{t-1}k_t)k_t^T
=\alpha_tS_{t-1}(I-(\eta_t/\alpha_t) k_tk_t^T)
+\eta_t v_tk_t^T
]
for nonzero (alpha_t), with alpha_t and eta_t kept as separate coefficients.
In this column-vector convention the projection multiplies (S) on the **right**.
The proposed ((I-\beta kk^T)S) has the wrong side/dimensions for this
definition. A different matrix convention is possible, but it must be stated
before implementation.

A chunkwise WY representation can compactly represent a product of rank-one
updates, but its (W_C) coefficients must be constructed recursively. The
simple product is not automatically a single rank-one correction. A
sequential reference and chunkwise implementation must agree in tests before
GPU/parallel code is attempted.

## Formula 6 — KL halting and stochastic trajectories

**Status: deferred.**

Small (D_{KL}(p_t\|p_{t-1})) means the output distribution changed little; it
does not mean the answer is correct or that additional reasoning is useless.
A halt policy requires calibration on held-out easy/hard examples, minimum and
maximum step guards, and a measured quality/compute trade-off.

Gaussian perturbations in latent space are continuous stochastic search, not
a discrete diffusion process by themselves. A Q-head can only rank candidates
reliably after training on independently verifiable outcomes or rewards.
Neither a halt policy nor a Q-head is present in v0.1.

## Formula 7 — Muon / Newton-Schulz optimizer

**Status: deferred.**

A quintic Newton-Schulz iteration approximates the polar factor under
conditions on matrix scaling, aspect ratio and coefficient choice. Five
iterations with one fixed coefficient triple are not automatically stable
for every matrix shape or precision. Before adoption, compare against a
high-precision SVD polar factor across tall, wide and square matrices, test
finite gradients/updates, then run matched optimizer ablations on the actual
Add/Sub task.

Muon has a momentum-like matrix state; Adam commonly has two moment states.
The memory difference can be meaningful, but a universal 2x compute or 50%
end-to-end memory reduction is not established by the formula alone.

## Framework and performance statement

A missing built-in sub-byte tensor dtype does not prevent a framework from
running packed weights: custom C++/CUDA/Triton kernels can consume packed byte
storage directly. Python dispatch and eager intermediate tensors can be
bottlenecks for small recurrent kernels, but graph capture/fusion and custom
autograd can mitigate them. For v0.1, OxCore is an independently testable
native primitive layer; PyTorch remains the Add/Sub training implementation
until end-to-end parity and measured benefits justify a controlled migration.
