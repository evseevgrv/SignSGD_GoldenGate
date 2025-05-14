from torch.optim import AdamW
from torch.optim.optimizer import _use_grad_for_differentiable

class AdamWDelayed(AdamW):
    def __init__(self, *args, delay_start_step=10000, grad_taylor_approx=False, taylor_linear_coef=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_start_step = delay_start_step
        self.grad_taylor_approx = grad_taylor_approx
        self.taylor_linear_coef = taylor_linear_coef

    @_use_grad_for_differentiable
    def step(self, closure=None):
        for group in self.param_groups:
            if not group.get("is_proj_params", False):
                continue
            for p in group["params"]:
                if self.state[p].get("step", 0) < self.delay_start_step:
                    continue
                if not hasattr(p, 'prev_grad'):
                    p.prev_grad = None
                p.grad, p.prev_grad = p.prev_grad, p.grad

                if self.grad_taylor_approx:
                    if p.grad is not None:
                        diff = p.prev_param.sub_(p)
                        correction = p.grad * diff * (p.grad)
                        p.grad.add_(correction, alpha=-self.taylor_linear_coef)
                    p.prev_param = p.data.clone()

        super().step(closure=closure)