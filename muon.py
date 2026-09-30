"""Muon optimizer with AdamW updates for parameters outside Muon's scope.

TUNE the constants marked below for your experiments. Muon is used for
2-D matrix weights; embeddings, biases, and normalization parameters use AdamW.
"""
import torch


class Muon(torch.optim.Optimizer):
    def __init__(self, params, *, muon_lr=0.02, adamw_lr=0.0003,
                 momentum=0.95, weight_decay=0.01, ns_steps=5):
        # TUNE: Muon learning rate for matrix weights.
        # TUNE: AdamW learning rate for embeddings and 1-D parameters.
        # TUNE: momentum controls smoothing of Muon's gradient direction.
        # TUNE: weight_decay is applied to both parameter groups.
        # TUNE: ns_steps controls Newton-Schulz orthogonalization accuracy.
        defaults = dict(muon_lr=muon_lr, adamw_lr=adamw_lr,
                        momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, beta1=0.9, beta2=0.95, eps=1e-8)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(matrix, steps):
        """Approximate the orthogonal factor of a 2-D update using Newton-Schulz."""
        transposed = matrix.size(0) < matrix.size(1)
        x = matrix.T if transposed else matrix
        x = x / (x.norm() + 1e-7)
        # TUNE: coefficients are the standard stable quintic iteration constants.
        for _ in range(steps):
            a = x @ x.T
            b = -4.7750 * a + 2.0315 * (a @ a)
            x = 3.4445 * x + b @ x
        return x.T if transposed else x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients.')
                state = self.state[param]

                if group.get('use_muon', param.ndim == 2):
                    if param.ndim != 2:
                        raise ValueError('Muon updates require 2-D matrix parameters.')
                    momentum = state.setdefault('momentum_buffer', torch.zeros_like(grad))
                    momentum.mul_(group['momentum']).add_(grad)
                    update = self._orthogonalize(momentum, group['ns_steps'])
                    # Normalize for matrix aspect ratio, following common Muon practice.
                    update.mul_(max(1., param.size(0) / param.size(1)) ** 0.5)
                    if group['weight_decay']:
                        param.mul_(1. - group['muon_lr'] * group['weight_decay'])
                    param.add_(update, alpha=-group['muon_lr'])
                else:
                    # AdamW for embeddings and vector parameters.
                    if not state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(param)
                        state['exp_avg_sq'] = torch.zeros_like(param)
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.mul_(group['beta1']).add_(grad, alpha=1 - group['beta1'])
                    exp_avg_sq.mul_(group['beta2']).addcmul_(grad, grad, value=1 - group['beta2'])
                    bias1 = 1 - group['beta1'] ** state['step']
                    bias2 = 1 - group['beta2'] ** state['step']
                    denom = exp_avg_sq.sqrt().div_(bias2 ** 0.5).add_(group['eps'])
                    if group['weight_decay']:
                        param.mul_(1. - group['adamw_lr'] * group['weight_decay'])
                    param.addcdiv_(exp_avg, denom, value=-group['adamw_lr'] / bias1)
        return loss
