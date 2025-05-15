from typing import cast, List, Optional, Union, Tuple

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _use_grad_for_differentiable,
    Optimizer,
)

from torch.optim.adamw import (
    AdamW,
    adamw
)

def convert_to_tensor(obj, device=None, dtype=None):
    if not isinstance(obj, torch.Tensor):
        obj = torch.tensor(obj)
    if device is not None:
        obj = obj.to(device)
    if dtype is not None:
        obj = obj.to(dtype)
    return obj

class AIDsignSGD(Optimizer):
    def __init__(
        self,
        params,
        L_inf: float,
        lower_bound: Optional[float] = None,
        d_0: Optional[float] = None,
        weight_decay: float = 0,
        clamp_level: Optional[float] = None,
        update_gap: int = 1,

        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,

        lr=None,
        warmup_steps=0,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        assert d_0 is not None or lower_bound is not None, "One have to specify either d_0 (for option 1) or lower_bound (for option 2)."
        assert d_0 is None or d_0 > 0.0, "d_0 should be positive value."

        if lr is None:
            lr = 1e-3
        if isinstance(lr, float):
            lr = torch.tensor(lr)

        defaults = dict(
            d=d_0 if d_0 is None else torch.tensor(d_0),
            L_inf=L_inf,
            lower_bound=lower_bound,
            weight_decay=weight_decay,
            f_diff=None,
            tilde_d=None,
            prev_gamma=None,
            lambda_denom_sum=None,
            update_gap=update_gap,

            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,

            lr=lr,
            warmup_steps=warmup_steps,

            clamp_level=1e-3 if clamp_level is None else clamp_level,

            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):   
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, prev_grad_list, prev_param_list=None, loss=None, momentum_buffer_list=None):
        has_sparse_grad = False
        device, dtype = group["params"][0].device, group["params"][0].dtype
        if group["lambda_denom_sum"] is None:
            group["lambda_denom_sum"] = convert_to_tensor(group["L_inf"], device=device, dtype=dtype)
            group["clamp_level"] = convert_to_tensor(group["clamp_level"], device=device, dtype=dtype)
        if group["d"] is not None and group["tilde_d"] is None:
            group["d"] = convert_to_tensor(group["d"], device=device, dtype=dtype)
            group["tilde_d"] = torch.tensor(0.0, device=device, dtype=dtype)
            group["prev_gamma"] = []
        elif group["lower_bound"] is not None and group["f_diff"] is None:
            assert loss is not None, "One have to pass f_0 (loss) to the first step of algorithm."
            group["f_diff"] = convert_to_tensor(loss, device=device, dtype=dtype) - convert_to_tensor(group["lower_bound"], device=device, dtype=dtype) 
            group["prev_gamma"] = []
        if "step" not in group:
            group["step"] = 0

        for p in group['params']:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'prev_grad' not in state:
                    prev_grad_list.append(None)
                else:
                    prev_grad_list.append(state['prev_grad'])

                if 'prev_param' not in state:
                    prev_param_list.append(None)
                else:
                    prev_param_list.append(state['prev_param'])

                if group["momentum"] != 0:
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

    def _get_loss(self, loss=None):
        return loss

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            prev_grad_list: List[Optional[Tensor]] = []
            prev_param_list: List[Optional[Tensor]] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, prev_grad_list, prev_param_list, loss, momentum_buffer_list
            )

            aid_sign_sgd(
                params,
                grads,
                prev_grad_list,
                prev_param_list,
                d=group["d"],
                tilde_d=group["tilde_d"],
                prev_gamma=group["prev_gamma"],
                lambda_denom_sum=group["lambda_denom_sum"],
                f_diff=group["f_diff"],
                weight_decay=group["weight_decay"],

                momentum_buffer_list=momentum_buffer_list,
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],

                lr=group["lr"],
                warmup_steps=group["warmup_steps"],

                clamp_level=group["clamp_level"],
                step=group["step"],
                update_gap=group["update_gap"],

                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )
            group["step"] += 1

            for p, prev_grad, prev_param in zip(params, prev_grad_list, prev_param_list):
                state = self.state[p]
                state["prev_grad"] = prev_grad
                state["prev_param"] = prev_param

        return loss

def aid_sign_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    prev_grad_list: List[Optional[Tensor]],
    prev_param_list: List[Optional[Tensor]],
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    d: torch.tensor,
    tilde_d: torch.tensor,
    prev_gamma,

    momentum_buffer_list,
    momentum,
    lr,
    dampening,
    nesterov,
    warmup_steps,

    lambda_denom_sum: torch.tensor,
    f_diff: Optional[float],
    clamp_level: torch.tensor,
    step: int,
    update_gap: int,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    if foreach is None:
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        raise NotImplementedError("`foreach` option is not implemented")
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        raise NotImplementedError("`fused` option is not implemented")
        func = _fused_sgd
    else:
        func = _single_tensor_aid_sign_sgd

    func(
        params,
        d_p_list,
        prev_grad_list,
        prev_param_list,
        weight_decay=weight_decay,
        d=d,
        tilde_d=tilde_d,
        prev_gamma=prev_gamma,
        lambda_denom_sum=lambda_denom_sum,
        f_diff=f_diff,

        momentum_buffer_list=momentum_buffer_list,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        warmup_steps=warmup_steps,

        clamp_level=clamp_level,
        step=step,
        update_gap=update_gap,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_aid_sign_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    prev_grad_list: List[Optional[Tensor]],
    prev_param_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    d: torch.tensor,
    tilde_d: torch.tensor,
    prev_gamma,
    lambda_denom_sum: torch.tensor,
    f_diff: Optional[float],

    momentum_buffer_list,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    warmup_steps,

    clamp_level: torch.tensor,
    step: int,
    update_gap: int,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    if step >= warmup_steps:
        if step % update_gap == 0:
            if step > warmup_steps:
                numerator_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [torch.linalg.vector_norm(g - prev_g, 1) for g, prev_g in zip(grads, prev_grad_list)]
                    ), 1
                )
                denominator_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [torch.linalg.vector_norm(p - prev_p, float("inf")) for p, prev_p in zip(params, prev_param_list)]
                    ), float("inf")
                )
                lambda_denom_sum.add_(numerator_norm / denominator_norm)

            if tilde_d is not None:
                if step > warmup_steps:
                    tilde_d.add_(sum((g * prev_g.sign() * pr_gamma).sum() for g, prev_g, pr_gamma in zip(grads, prev_grad_list, prev_gamma)))
                d = torch.max(d, tilde_d, out=d)
                gamma = (d.div(lambda_denom_sum)).sqrt()
            else:
                gamma = (f_diff.div(lambda_denom_sum)).sqrt()
            gamma.clamp_(max=clamp_level)
            prev_gamma.append(gamma.clone())
        lr.copy_(prev_gamma[-1])

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if step % update_gap == 0:
            prev_grad_list[i] = torch.clone(grads[i].detach())
            prev_param_list[i] = torch.clone(param.detach())

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        param.add_(grad.sign(), alpha=-lr)

class AIDWithAdam(Optimizer):  
    def __init__(
        self,
        params,
        L_inf: float,
        lower_bound: Optional[float] = None,
        d_0: Optional[float] = None,
        weight_decay: float = 0,
        clamp_level: Optional[float] = None,
        update_gap: int = 1,


        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,

        adam_lr: Union[float, Tensor] = 1e-3,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        adam_amsgrad: bool = False,
        
        warmup_steps=0, 
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        assert d_0 is not None or lower_bound is not None, "One have to specify either d_0 (for option 1) or lower_bound (for option 2)."
        assert d_0 is None or d_0 > 0.0, "d_0 should be positive value."
        defaults = dict()

        if isinstance(adam_lr, float):
            adam_lr = torch.tensor(adam_lr)
        
        super().__init__([group for group in params], defaults)
        del self.param_groups
        aid_params = [group for group in params if self._is_sgd_group(group)]
        adam_params = [group for group in params if not self._is_sgd_group(group)]

        aid_sign_sgd = AIDsignSGD(
            aid_params, 
            L_inf=L_inf,
            lower_bound=lower_bound,
            d_0=d_0,
            weight_decay=weight_decay,
            clamp_level=clamp_level,
            update_gap=update_gap,
            foreach=False,
            fused=False,
            differentiable=False,

            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,

            lr=adam_lr,
            warmup_steps=warmup_steps,
        )

        adam = torch.optim.AdamW(
            adam_params,
            lr=adam_lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
            amsgrad=adam_amsgrad,

            maximize=maximize,
            foreach=foreach,
            fused=fused,
            differentiable=differentiable,
        )
        self.param_groups = aid_sign_sgd.param_groups + adam.param_groups

        self._init_group_aid = AIDsignSGD._init_group.__get__(self)
        self._init_group_adam = AdamW._init_group.__get__(self)

        self._get_loss = aid_sign_sgd._get_loss

    def _is_sgd_group(self, group):
        return group.get("is_proj_params", False) or group.get("is_sgd_params", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self._is_sgd_group(group):
                params: List[Tensor] = []
                grads: List[Tensor] = []
                prev_grad_list: List[Optional[Tensor]] = []
                prev_param_list: List[Optional[Tensor]] = []
                momentum_buffer_list: List[Optional[Tensor]] = []

                has_sparse_grad = self._init_group_aid(
                    group, params, grads, prev_grad_list, prev_param_list, loss, momentum_buffer_list
                )

                aid_sign_sgd(
                    params,
                    grads,
                    prev_grad_list,
                    prev_param_list,
                    d=group["d"],
                    tilde_d=group["tilde_d"],
                    prev_gamma=group["prev_gamma"],
                    lambda_denom_sum=group["lambda_denom_sum"],
                    f_diff=group["f_diff"],
                    weight_decay=group["weight_decay"],

                    clamp_level=group["clamp_level"],
                    step=group["step"],
                    update_gap=group["update_gap"],

                    momentum_buffer_list=momentum_buffer_list,
                    momentum=group["momentum"],
                    dampening=group["dampening"],
                    nesterov=group["nesterov"],

                    lr=group["lr"],
                    warmup_steps=group["warmup_steps"],

                    maximize=group["maximize"],
                    has_sparse_grad=has_sparse_grad,
                    foreach=group["foreach"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                )
                group["step"] += 1

                for p, prev_grad, prev_param in zip(params, prev_grad_list, prev_param_list):
                    state = self.state[p]
                    state["prev_grad"] = prev_grad
                    state["prev_param"] = prev_param
                
            else:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []
                amsgrad = group["amsgrad"]
                beta1, beta2 = group["betas"]

                self._init_group_adam(
                    group,
                    params_with_grad,
                    grads,
                    amsgrad,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                )

                for p in group["params"]:
                    state = self.state[p]
                    state["prev_param"] = p.detach().clone()

                adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                )

        return loss

class AdamLike(Optimizer):
    def __init__(
        self,
        params,
        L_inf: float,
        lower_bound: Optional[float] = None,
        d_0: Optional[float] = None,
        weight_decay: float = 0,
        clamp_level: Optional[float] = None,
        update_gap: int = 1,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        adam_lr: Union[float, Tensor] = 1e-3,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        adam_amsgrad: bool = False,
        warmup_steps: int = 0,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        assert d_0 is not None or lower_bound is not None, "Specify either d_0 or lower_bound."
        assert d_0 is None or d_0 > 0.0, "d_0 must be positive."
        
        defaults = dict(
            L_inf=L_inf,
            lower_bound=lower_bound,
            d_0=d_0,
            weight_decay=weight_decay,
            clamp_level=clamp_level,
            update_gap=update_gap,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            lr=adam_lr,
            warmup_steps=warmup_steps,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
            betas=adam_betas,
            eps=adam_eps,
            amsgrad=adam_amsgrad,
            capturable=False
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                device = p.device
                state.setdefault('step', torch.tensor(0., device=device))
                state.setdefault('exp_avg', torch.zeros_like(p))
                state.setdefault('exp_avg_sq', torch.zeros_like(p))
                state.setdefault('prev_grad', None)
                state.setdefault('r', torch.tensor(1.0, device=device))
                state.setdefault('d', torch.tensor(1.0, device=device))
                if group.get('amsgrad', False):
                    state.setdefault('max_exp_avg_sq', torch.zeros_like(p))

    def _is_sgd_group(self, group):
        return group.get("is_proj_params", False) or group.get("is_sgd_params", False)

    def _get_loss(self, loss=None):
        return loss
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self._is_sgd_group(group):
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_steps = []
                prev_grads = []
                rs = []
                ds = []

                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                maximize = group['maximize']
                warmup_steps = group['warmup_steps']

                lr_scalar = lr.item() if isinstance(lr, Tensor) else lr

                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad)
                        state = self.state[p]
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        prev_grad = state['prev_grad']
                        rs.append(state['r'])
                        ds.append(state['d'])
                        state_steps.append(state['step'])

                for i, param in enumerate(params_with_grad):
                    grad = grads[i] 
                    step_t = state_steps[i].item()
                    exp_avg = exp_avgs[i]
                    exp_avg_sq = exp_avg_sqs[i]
                    r = rs[i]
                    d = ds[i]
                    prev_grad = self.state[param]['prev_grad']             

                    if prev_grad is None:
                        prev_grad = torch.ones_like(grad)
                        self.state[param]['prev_grad'] = prev_grad

                    if step_t > 0:
                        dot_product = (grad.abs() * prev_grad.abs()).sum().item()
                        new_r = r.item() * (beta2**0.5) + (1 - beta2**0.5) * d.item() * dot_product
                        new_d = max(d.item(), new_r)
                        r.fill_(new_r)
                        d.fill_(new_d)

                    d_scalar = d.item()
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1) * d_scalar)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2) * (d_scalar**2))

                    m_hat = exp_avg   
                    v_hat = exp_avg_sq 
                    denom = v_hat - m_hat.square()  
                    denom = denom / (m_hat.square() + eps)
                    if warmup_steps > 0 and step_t < warmup_steps:
                        d_scalar = 1
                    step_size = (d_scalar**2 / (1 + denom)).sqrt()
                    
                    step_size *= lr_scalar
                    param.add_(-exp_avg.sign() * step_size)
                    
                    if weight_decay != 0:
                        param.mul_(1 - lr_scalar * weight_decay)  

                    self.state[param]['prev_grad'].copy_(grad)
                    self.state[param]['d'].copy_(d)
                    self.state[param]['r'].copy_(r)
                    self.state[param]['exp_avg'].copy_(exp_avg)
                    self.state[param]['exp_avg_sq'].copy_(exp_avg_sq)
                    state_steps[i] += 1

            else:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []

                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                amsgrad = group['amsgrad']
                maximize = group['maximize']

                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad)
                        state = self.state[p]
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        if amsgrad:
                            max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                        else:
                            max_exp_avg_sqs.append(None)
                        state_steps.append(state['step'])

                torch.optim._functional.adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    maximize=maximize,
                    foreach=False,
                    capturable=group['capturable'],
                    differentiable=group['differentiable'],
                    fused=False,
                )
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['r'] = state['r'].to(p.device)
                state['d'] = state['d'].to(p.device)
                if state['prev_grad'] is not None:
                    state['prev_grad'] = state['prev_grad'].to(p.device)
        return loss


class ProdigyWithAdam(Optimizer):
    def __init__(
        self,
        params,
        L_inf: float = None,         
        lower_bound: Optional[float] = None,
        d_0: Optional[float] = None,
        weight_decay: float = 0,
        clamp_level: Optional[float] = None,
        update_gap: int = 1,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        amsgrad: bool = False,       
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        defaults = dict(
            L_inf=L_inf,
            lower_bound=lower_bound,
            d_0=d_0,
            weight_decay=weight_decay,
            clamp_level=clamp_level,
            update_gap=update_gap,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            warmup_steps=warmup_steps,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
            amsgrad=amsgrad,
            capturable=False
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                device = p.device
                state.setdefault('step', torch.tensor(0., device=device))
                state.setdefault('m', torch.zeros_like(p))
                state.setdefault('v', torch.zeros_like(p))
                state.setdefault('r', torch.tensor(0.0, device=device))
                state.setdefault('s', torch.zeros_like(p))
                state.setdefault('d', torch.tensor(group['d_0'] if group['d_0'] is not None else 1e-6, device=device))
                state.setdefault('x0', p.data.clone().detach())
                
                state.setdefault('exp_avg', torch.zeros_like(p))
                state.setdefault('exp_avg_sq', torch.zeros_like(p))
                if group.get('amsgrad', False):
                    state.setdefault('max_exp_avg_sq', torch.zeros_like(p))

    def _is_sgd_group(self, group):
        return group.get("is_proj_params", False) or group.get("is_sgd_params", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self._is_sgd_group(group):
                beta1, beta2 = group['betas']
                lr = group['lr']
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']
                warmup_steps = group['warmup_steps']
                maximize = group['maximize']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    if maximize:
                        grad = -grad

                    state = self.state[p]
                    m, v = state['m'], state['v']
                    r, s = state['r'], state['s']
                    d, x0 = state['d'], state['x0']
                    step_t = state['step']
                    step = step_t.item()

                    step_t.add_(1)

                    lr_current = lr
                    m.mul_(beta1).add_(grad * d, alpha=(1 - beta1))
                    v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2) * (d**2))

                    delta_x = x0 - p.data
                    dot_product = torch.sum(grad * delta_x)

                    beta2_sqrt = torch.sqrt(torch.tensor(beta2, device=r.device, dtype=r.dtype))
                    r_new = beta2_sqrt * r + (1 - beta2_sqrt) * lr_current * (d**2) * dot_product
                    s_new = beta2_sqrt * s + (1 - beta2_sqrt) * lr_current * (d**2) * grad

                    r.copy_(r_new)
                    s.copy_(s_new)

                    s_l1_norm = torch.sum(torch.abs(s_new))
                    hat_d = r_new / (s_l1_norm + epsilon)

                    new_d = torch.max(d, hat_d)
                    d.copy_(new_d)

                    if warmup_steps > 0 and step < warmup_steps:
                        d = 1
                    denominator = torch.sqrt(v) + d * epsilon
                    step_update = (lr_current * d * m) / denominator
                    p.data.sub_(step_update)
                    
                    state['d'].copy_(d)
                    state['r'].copy_(r)
                    state['m'].copy_(m)
                    state['v'].copy_(v)

                    if weight_decay != 0:
                        p.data.mul_(1 - lr_current * weight_decay)

            else:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []

                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['epsilon']
                weight_decay = group['weight_decay']
                amsgrad = group['amsgrad']
                maximize = group['maximize']

                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad)
                        state = self.state[p]
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        if amsgrad:
                            max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                        else:
                            max_exp_avg_sqs.append(None)
                        state_steps.append(state['step'])

                torch.optim._functional.adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    maximize=maximize,
                    foreach=False,
                    capturable=group['capturable'],
                    differentiable=group['differentiable'],
                    fused=False,
                )

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                for key in state:
                    if isinstance(state[key], torch.Tensor):
                        state[key] = state[key].to(p.device)
        return loss
