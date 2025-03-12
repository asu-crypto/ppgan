#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch
from .optimizer import Optimizer
import math

class Adam(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        grad_threshold (float, optional): imposes a threshold on the magnitude of gradient values.
            Gradient values with magnitude above the threshold will be replaced with 0.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr,
        betas=(0.9,0.999),
        eps= 1e-8,
        weight_decay=0,
        grad_threshold=None,
    ):
        if not isinstance(lr, (int, float)) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not isinstance(betas[0], (int, float)) or betas[0] < 0.0:
            raise ValueError("Invalid beta value: {}".format(betas))
        if not isinstance(betas[1], (int, float)) or betas[1] < 0.0:
            raise ValueError("Invalid beta value: {}".format(betas))
        if not isinstance(eps, (int, float)):
            raise ValueError("Invalid epsilon value {}".format(eps))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        
        # Compute thresholding based on square value since abs is more expensive
        self.square_threshold = grad_threshold
        if self.square_threshold is not None:
            self.square_threshold *= self.square_threshold
        self.step_ = 0
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
    
    # def _init_group(
    #     self,
    #     group,
    #     params_with_grad,
    #     grads,
    #     exp_avgs,
    #     exp_avg_sqs,
    #     max_exp_avg_sqs,
    #     state_steps
    # ):
    #     for p in group['params']:
    #         if p.grad is not None:
    #             params_with_grad.append(p)
    #             grads.append(p.grad)

    #             state = self.state[p]
    #             # Lazy state initialization
    #             if len(state) == 0:
    #                 state['step'] = (
    #                     torch.tensor(0.)
    #                 )
    #                 # Exponential moving average of gradient values
    #                 expavg = torch.zeros(p.shape)#, memory_format=torch.preserve_format)
    #                 state['exp_avg'] = crypten.cryptensor(expavg)
    #                 # Exponential moving average of squared gradient values
    #                 expavgsq = torch.zeros(p.shape)#, memory_format=torch.preserve_format)
    #                 state['exp_avg_sq'] = crypten.cryptensor(expavgsq)
    #             exp_avgs.append(state['exp_avg'])
    #             exp_avg_sqs.append(state['exp_avg_sq'])

                
    #             state_steps.append(state['step'])

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with crypten.no_grad():
            self._cuda_graph_capture_health_check()

            loss = None
            t = self.step_+ 1
            if closure is not None:
                with crypten.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                # weight_decay = group["weight_decay"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lr  = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if self.square_threshold is not None:
                        d_p = p.grad.mul(p.grad.square().lt(self.square_threshold))
                    else:
                        d_p = p.grad
                    d_p_square = d_p.square()
                    state = self.state[p]
                    if len(state) == 0:
                        state['m'] = d_p.mul(0)
                        state['v'] = d_p.mul(0)
                        
                    state['m'] = state['m'].mul(beta1) + d_p.mul(1-beta1)
                    state['v'] = state['v'].mul(beta2) + d_p_square.mul(1-beta2)
                    # mhat = state['m'].div(1-beta1**t)
                    # vhat = state['v'].div(1-beta2**t)
                    alpha = group['lr']*math.sqrt(1-beta2**t)/(1-beta1**t)
                    p.sub_((state['m'].mul(alpha)).div(state['v'].sqrt()+group['eps']))
            self.step_ += 1


            # for group in self.param_groups:
            #     weight_decay = group["weight_decay"]
            #     beta1, beta2 = group["betas"]
            #     eps = group["eps"]
            #     lr  = group['lr']

            #     params_with_grad = []
            #     grads = []
            #     exp_avgs = []
            #     exp_avg_sqs = []
            #     max_exp_avg_sqs = []
            #     state_steps = []
            #     self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)

            #     for i, param in enumerate(params_with_grad):

            #         grad = grads[i]
            #         # if param.grad is None:
            #         #     continue

            #         # Threshold gradients to prevent gradient explosion
            #         if self.square_threshold is not None:
            #             grad = grad.mul(grad.square().lt(self.square_threshold))
            #         # else:
            #         #     d_p = p.grad
            #         exp_avg = exp_avgs[i]
            #         exp_avg_sq = exp_avg_sqs[i]
            #         step_t = state_steps[i]

            #         # update step
            #         step_t += 1

            #         if weight_decay != 0:
            #             grad = grad.add(param, alpha=weight_decay)

                    
            #         gradconj = crypten.cryptensor(grad.get_plain_text().conj())
            #         # Decay the first and second moment running average coefficient
            #         exp_avg.mul_(beta1).add_(grad.mul(1 - beta1))
            #         exp_avg_sq.mul_(beta2).add_((grad*gradconj).mul(1 - beta2))

            #         step = step_t#_get_value(step_t)

            #         bias_correction1 = 1 - beta1 ** step
            #         bias_correction2 = 1 - beta2 ** step

            #         step_size = lr / bias_correction1

            #         bias_correction2_sqrt = bias_correction2.sqrt()

                    
            #         denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            #         param.add_((exp_avg/denom).mul(-step_size))
                    # param.addcdiv_(exp_avg, denom, value=-step_size)

                # for p in group["params"]:
                #     if p.grad is None:
                #         continue

                #     # Threshold gradients to prevent gradient explosion
                #     if self.square_threshold is not None:
                #         d_p = p.grad.mul(p.grad.square().lt(self.square_threshold))
                #     else:
                #         d_p = p.grad

                #     if weight_decay != 0:
                #         d_p = d_p.add(p.mul(weight_decay))
                #     if momentum != 0:
                #         param_state = self.state[id(p)]
                #         if "momentum_buffer" not in param_state:
                #             buf = param_state["momentum_buffer"] = d_p.clone().detach()
                #         else:
                #             buf = param_state["momentum_buffer"]
                #             buf.mul_(momentum).add_(d_p.mul(1 - dampening))
                #         if nesterov:
                #             d_p = d_p.add(buf.mul(momentum))
                #         else:
                #             d_p = buf

                #     p.sub_(d_p.mul(group["lr"]))

            return loss
