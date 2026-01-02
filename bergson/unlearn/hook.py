from functools import partial

from bergson.unlearn.utils import stable_rank


class ActivationCapture:
    def __init__(self, model, target_module_names, accelerator=None):
        self.model = model
        self.target_module_names = set(target_module_names)
        self.activations = {}
        self._handles = []
        self.accelerator = accelerator

    def _hook_fn(self, module, input, output, name):
        act = output[0] if isinstance(output, tuple) else output
        self.activations[name] = act

    def register(self):
        self.activations = {}

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        for name, module in model.named_modules():
            if name in self.target_module_names:
                handle = module.register_forward_hook(partial(self._hook_fn, name=name))
                self._handles.append(handle)

    def clear(self):
        self.activations = {}

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class GradientRankCapture:
    def __init__(self, model, target_module_names):
        self.model = model
        self.target_module_names = set(target_module_names)
        self._handles = []
        self.gradient_ranks = {}

    def _hook_fn(self, module, input, output, name):
        grad = output[0] if isinstance(output, tuple) else output
        self.gradients[name] = grad

    def register(self):
        for name, module in self.model.named_modules():
            if name in self.target_module_names:
                handle = module.register_backward_hook(
                    partial(self._hook_fn, name=name)
                )
                self._handles.append(handle)

    def clear(self):
        self.gradients = {}

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _compute_gradient_rank(self, gradient):
        return stable_rank(gradient)
