# Optimizer-Agnostic Gradient Transformation

The optimizer is never edited. Instead, the gradient is rewritten before and
after `optimizer.step()`, using the two hooks `torch.optim.Optimizer` exposes:

```py
optimizer.register_step_pre_hook(pre)    # runs just before step()
optimizer.register_step_post_hook(post)  # runs just after  step()
```

`pre` walks `optimizer.param_groups` and, for every parameter with a
gradient, replaces `p.grad` in place by $T(g) = g|g|$, saving a copy of the parameter. 
The optimizer then executes its ordinary `step()`, reading `p.grad` as usual, so it sees only
$T(g)$. `post` restores the original gradient by copying it back.

## Optimizer-agnostic

The only interface an optimizer has with the outside world is to read
`p.grad` for each `p` in `param_groups` during `step()`. The hooks operate
solely on that interface and don't touch optimizer's state, its
hyper-parameters, or its update rule, and they are attached through the
base-class hook API, which every optimizer inherits. 

Thus the same approach works for SGD, AdamW, and even 3rd-party optimizers as long
as they inherit the same base-class protocol. 

The test verifies this for SGD 
and AdamW by checking, to $10^{-12}$, that the hooked run matches a manual
run in which `p.grad` is set to $T(g)$ by hand and the same `step()` is
called.

## Assumptions, limitations, unsupported cases

- This ransforms the accumulated gradient. Because $T$ is nonlinear,
  $T(g_1 + g_2) \ne T(g_1) + T(g_2)$. Applying $T$ at `step()` time is
  therefore the correct semantics for gradient accumulation and for
  parameters used several times in a graph. A per-contribution tensor hook
  (`p.register_hook`) would be wrong for these cases.
- Ordering with other gradient processing is the user's responsibility. For example, 
  gradient clipping applied in the training loop would see $g$, not $T(g)$.
- Parameters with `grad is None` are skipped (which matches optimizer behaviour).
- Restoring is only actually observable if something reads `p.grad` between
  `step()` and the next `zero_grad()`. If nothing does, `post` can be
  dropped.

An alternate way (since $T$ is a bijection) to do this would be to directly perform the transform $T$ on `p.grad` in `pre` and then perform the inverse transform $T^{-1}(g) = \text{sign}(g) \sqrt{|g|}$ in `post`. This would have no (O(1)) memory overhead as everything would be in place, but still have $O(N)$ time complexity. This would also suffer from floating point rounding issues - the restored gradients wouldn't be exactly the same as they previously were, and precision will be lost through the process. 

## Overhead

Let $N$ be the total number of parameters.

- Time: The pre-hook performs one elementwise `abs` and one `mul_` per
  parameter; the post-hook performs a `copy_`. These are all
  $O(N)$ memory-bound kernels, launched once per parameter tensor. This is the same order as the
  optimizer's own update and is negligible next to forward/backward passes.
- Memory: It holds one clone of every gradient, i.e. $O(N)$ extra
  elements in the gradient dtype, which live only for the duration of `step()`
  and are freed in the post-hook.
