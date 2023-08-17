from torch.optim import Optimizer
import torch
import math

# Note: for all algorithms with momentum prarameter, dampening is 1-momentum


class AlgorithmBase:
    @staticmethod
    def update(paras, grad, state, group_state, *args, **kwargs):
        raise NotImplementedError


class Algorithm(Optimizer):
    """
    Note that kwargs should include all the parameters, such as lr, momentum, etc.
    Parameters are grouped and gradient normalization is applied group-wise.
    """

    def __init__(self, params, algo, vr=True, **kwargs):
        super(Algorithm, self).__init__(params, kwargs)
        self._params = self.get_params()
        self.iter = 0
        self.vr = vr
        assert issubclass(algo, AlgorithmBase)
        self.algo = algo

    def get_params(self):
        """
        gets all parameters in all param_groups with gradients requirements
        """
        return [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]

    def _save_past_info(self, vlist, key):
        """
        saving momentum
        """
        if key == "old_p":
            with torch.no_grad():
                for idx, p in enumerate(self._params):
                    if p not in self.state:
                        self.state[p] = {}
                    self.state[p][key] = vlist[idx].clone()
        else:
            with torch.no_grad():
                for idx, p in enumerate(self._params):
                    self.state[p][key] = vlist[p]

    def get_past_info(self, key):
        info = {p: self.state[p][key] for p in self._params if key in self.state[p]}
        return info

    @torch.no_grad()
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _clone_grad(self):
        return {p: p.grad.detach().clone() for p in self._params}

    def get_grad(self, bool_switch=False, closure=None):
        # copy of params at last step
        p_copy = self._clone_param()

        if self.vr:
            # load params at last step
            p_old = self.get_past_info("old_p")

            # load gradient at last step
            g_old = self.get_past_info("old_g")

            # q1 is the interval
            if bool_switch:
                loss = closure()
                # Compute the sum of gradients at each point in S1
                g_i = self._clone_grad()
                for key in g_i:
                    g_old[key] = g_i[key]
            else:
                # Update the previous gradient by
                #   adding the difference of gradients at each point in S2
                g_i = self._clone_grad()
                self._set_param(p_old)
                loss_old = closure()
                g_i_old = self._clone_grad()

                for key in g_i:
                    if key not in g_old:
                        g_old[key] = g_i[key] - g_i_old[key]
                    else:
                        g_old[key] += g_i[key] - g_i_old[key]

            self._set_param(p_copy)
            loss = closure()
            # g = self.normalize(g_old)
            g = g_old
        else:
            loss = closure()
            g = self._clone_grad()

        self._save_past_info(g, "old_g")
        self._save_past_info(p_copy, "old_p")
        self.iter += 1
        return g

    def step(self, bool_switch=False, closure=None):
        """
        bool_switch: if True then compute `larger batch gradient` else use the difference
        """
        if closure is None:
            raise ValueError("must provide a closure")

        loss = closure()
        for id, group in enumerate(self.param_groups):
            ps = [p for p in group["params"] if p.grad is not None]
            grad = self.get_grad(bool_switch=bool_switch, closure=closure)
            if "wd" in group and group["wd"] != 0:
                for p in ps:
                    grad[p] = grad[p] + p.data * group["wd"]
                    # p.grad.data.add_(p.data, alpha=group['wd'])
            self.algo.update(
                ps, grad, self.state, self.state["group" + str(id)], **group
            )
        return loss


def sum_of_square_grad(grads):
    return sum([p.view(-1).dot(p.view(-1)) for p in grads])


def update_momentum(grad, state, momentum):
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(grad, device=grad.device)
    buf = state["momentum_buffer"]
    buf.mul_(momentum).add_(1 - momentum, grad)
    return buf


class SGD(AlgorithmBase):
    # g = momentum * g + (1 - momentum) * grad
    # x = x - lr * g
    @staticmethod
    def update(paras, grad, state, group_state, lr, momentum=0, **kwargs):
        for p in paras:
            d_p = grad[p]
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            p.data.add_(-lr, d_p)


class NormalizedSGD(AlgorithmBase):
    # g = momentum * g + (1 - momentum) * grad
    # x = x - lr * g / |g|
    @staticmethod
    def update(paras, grad, state, group_state, lr, momentum=0, eps=1e-6, **kwargs):
        d_ps = []
        for p in paras:
            d_p = grad[p]
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            d_ps.append(d_p)
        sum = math.sqrt(sum_of_square_grad(d_ps))
        for p, d_p in zip(paras, d_ps):
            p.data.add_(-lr / (sum + eps), d_p)


class SGDClip(AlgorithmBase):
    # g = momentum * g + (1 - momentum) * grad
    # x = x - min(lr, gamma / |g|) * g
    @staticmethod
    def update(paras, grad, state, group_state, lr, gamma, momentum=0, **kwargs):
        d_ps = []
        for p in paras:
            d_p = grad[p]
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            d_ps.append(d_p)
        sum = math.sqrt(sum_of_square_grad(d_ps))
        for p, d_p in zip(paras, d_ps):
            p.data.add_(-min(lr, gamma / sum), d_p)


class Adagrad(AlgorithmBase):
    # g^2 = g^2 + |grad|^2
    # x = x - lr / g * grad
    @staticmethod
    def update(paras, state, group_state, lr, b0, **kwargs):
        sum = sum_of_square_grad([p.grad.data for p in paras])
        if "sum_buffer" not in group_state:
            group_state["sum_buffer"] = sum.new_ones(1) * b0 ** 2
        group_state["sum_buffer"].add_(sum)
        for p in paras:
            p.data.add_(-lr / math.sqrt(group_state["sum_buffer"]), p.grad.data)


class MixClip(AlgorithmBase):
    # see the paper
    @staticmethod
    def update(
        paras, grad, state, group_state, lr, gamma, momentum=0.999, nu=0.7, **kwargs
    ):
        d_ps = []
        for p in paras:
            d_p = grad[p]
            if momentum != 0:
                d_p = update_momentum(d_p, state[p], momentum)
            d_ps.append(d_p)
        sum = math.sqrt(sum_of_square_grad(d_ps))
        sum2 = math.sqrt(sum_of_square_grad([grad[p] for p in paras]))
        for p, d_p in zip(paras, d_ps):
            p.data.add_(-lr * nu / (1 + sum / gamma * lr), d_p)
            p.data.add_(-lr * (1 - nu) / (1 + sum2 / gamma * lr), grad[p])


class MomClip(AlgorithmBase):
    @staticmethod
    def update(paras, grad, state, group_state, lr, gamma, momentum=0.9, **kwargs):
        MixClip.update(
            paras,
            grad,
            state,
            group_state,
            lr,
            gamma,
            momentum=momentum,
            nu=1,
            **kwargs
        )
