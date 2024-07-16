# Standard Library
import contextlib
from typing import Any, Callable, Optional, TypeVar

# Third Party Library
import torch
from torch import nn

T = TypeVar("T", torch.Tensor, dict[Any, Any], list[Any], tuple[Any, ...])


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def recursive_tensor_copy(
    x: T,
    clone: Optional[bool] = None,
    detach: Optional[bool] = None,
    retain_grad: Optional[bool] = None,
) -> T:
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_tensor_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_tensor_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


class TraceLayer(contextlib.AbstractContextManager["TraceLayer"]):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """

    input: Optional[torch.Tensor] = None
    output: Optional[torch.Tensor] = None

    def __init__(
        self,
        module: nn.Module,
        layer: str,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Optional[Callable[[Any, str], Any]] = None,
        stop: bool = False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        layer_module = get_module(module, layer)

        def retain_hook(_m: Any, inputs: Any, output: Any) -> Any:
            if retain_input:
                retainer.input = recursive_tensor_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only.
            if edit_output:
                output = edit_output(output, self.layer)
            if retain_output:
                retainer.output = recursive_tensor_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad and retainer.output is not None:
                    output = recursive_tensor_copy(
                        retainer.output, clone=True, detach=False
                    )
            if stop:
                raise StopForward()
            return output

        self.registered_hook = layer_module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self) -> "TraceLayer":
        return self

    def __exit__(self, type: Any, _value: Any, _traceback: Any) -> bool | None:
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
        return None

    def close(self) -> None:
        self.registered_hook.remove()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass
