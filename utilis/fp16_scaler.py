"""Dynamic loss scaling for a model whose parameters are FP16 (train.py --cast FP16).

torch.amp.GradScaler does dynamic loss scaling. It works like this:
  * Before backward, it multiplies the loss by S, so that small gradients do
    not underflow (become zero).
  * If any gradient is inf or NaN, it skips the step and halves S.
  * After N steps with no problems, it doubles S.
Its unscale_() refuses FP16 *parameter* gradients. The reason is that AMP
keeps the parameters in FP32, so it assumes that FP16 gradients are a mistake.

With --cast FP16, the parameters are FP16 on purpose. So this subclass
repeats GradScaler.unscale_() with allow_fp16=True. Everything else
(scale(), step(), update(), state_dict()) is inherited with no change.

Limitation: loss scaling protects the gradients during backward. But after
unscaling, they are stored in FP16 again, so gradients below about 6e-8
still become zero. This is part of pure FP16, and it is exactly what the
experiment studies.

Note: this calls the private method GradScaler._unscale_grads_(). If a future
PyTorch version changes it, this class fails with a clear error
(TypeError or AttributeError).
"""

import torch
from torch.amp.grad_scaler import OptState


class FP16ParamGradScaler(torch.amp.GradScaler):
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._enabled:
            return
        self._check_scale_growth_tracker("unscale_")
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)
        optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, True)
        optimizer_state["stage"] = OptState.UNSCALED
