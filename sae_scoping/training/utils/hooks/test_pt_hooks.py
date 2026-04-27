from __future__ import annotations


import torch
import torch.nn as nn

from functools import partial

# TODO(Claude) PYTEST-FAILING BUG [IMPORT-47D8A2BA]: Both import paths are stale after the
# package reorg (trainers/ → training/, utils/hooks/ → training/sae_enhanced/hooks/).
# First tries 'utils.hooks.pt_hooks' (old location), then falls back to bare 'pt_hooks'
# (only works when run directly from this directory). Pytest collection fails with:
#   pytest: ModuleNotFoundError: No module named 'utils'
#   then: ModuleNotFoundError: No module named 'pt_hooks'
# The correct import is: from sae_scoping.training.sae_enhanced.hooks.pt_hooks import named_forward_hooks
# This blocks collection of test_pt_hooks_modify_inputs (1 test) and
# test_pt_hooks_backpropagates (1 test).
try:
    from utils.hooks.pt_hooks import named_forward_hooks
except ImportError:
    from pt_hooks import named_forward_hooks
"""Unit tester for pt_hooks.py"""


################ TEST CLASSES ################
class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        # No bias because later we want to zero it out and make sure it does actually
        # -> zero
        self.linear1 = nn.Linear(10, 10, bias=False)
        self.linear2 = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class HookModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def hook_fn(hook_module, hooks, name, module, input, output):
    return hook_module(output)  # NOTE: need to get the OUTPUT to not break the graph


##############################################


################ TESTS ################
def _test_pt_hooks_modify_inputs():
    """
    Integration test to see that you can modify activations.
    """

    def forward_hook(hooks, name, module, input, output):
        print(
            f"Module {name}: input shape {input[0].shape}, output shape {output.shape}"
        )

    def reshape_hook(hooks, name, module, input, output):
        return torch.ones(10)  # Unexpected shape

    # Example usage
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(5, 1)
            self.sequence = nn.Sequential(nn.Linear(1, 1))

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return self.sequence(x)

    model = SimpleModel()
    hook_dict = {
        "linear1": forward_hook,
        "relu": forward_hook,
        "linear2": forward_hook,
        "sequence.0": reshape_hook,
    }
    print("=" * 100)
    print("Expect 10 and printouts")
    # TODO(Adriano) add pre-support?
    with named_forward_hooks(model, hook_dict) as hooks:
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        print(output.shape)  # 10
    print("=" * 100)
    print("Expect 1")
    output = model(input_tensor)
    print(output.shape)  # 1
    print("=" * 100)
    print("Expect 1")
    hook_dict = {
        "linear1": forward_hook,
        "relu": forward_hook,
        "linear2": forward_hook,
    }
    # TODO(Adriano) add pre-support?
    with named_forward_hooks(model, hook_dict) as hooks:
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        print(output.shape)  # 1


def test_pt_hooks_modify_inputs():
    """
    Unit test to see that you can modify activations.
    """
    main_net = MainNet()
    hook_module = HookModule()

    with torch.no_grad():
        # Very unlikely to become zero (so we'll check this nulls it out OK)
        hook_module.linear.weight[:] = 0
        hook_module.linear.bias[:] = 0
        _ = hook_module(torch.ones(10))
        assert _.max() == 0 and _.min() == 0

        # TODO(Adriano) add pre-support?
        with named_forward_hooks(
            main_net,
            hook_dict={
                # Insert a hook right after lienar1
                "linear1": partial(hook_fn, hook_module)
            },
        ):
            input = torch.ones(1, 10, requires_grad=False)  # Just some dummy input
            output = main_net(input)
            assert (
                output.min() == 0 and output.max() == 0
            )  # Everything should be zero'd


def test_pt_hooks_backpropagates():
    """
    Unit test that gradients ARE Able to backpropagate.
    """
    main_net = MainNet()
    hook_module = HookModule()

    # Check that nothing is changing here
    assert main_net.linear1.weight.grad is None
    assert main_net.linear1.bias is None
    assert main_net.linear2.weight.grad is None
    assert main_net.linear2.bias is None
    assert hook_module.linear.weight.grad is None
    assert hook_module.linear.bias.grad is None

    # TODO(Adriano) add pre-support?
    with named_forward_hooks(
        main_net,
        hook_dict={
            # Insert a hook right after lienar1
            "linear1": partial(hook_fn, hook_module)
        },
    ):
        input = torch.ones(1, 10)
        # input -> linear1 -> hook_linear -> linear2 = output
        output = main_net(input)
        loss = output.sum()
        loss.backward()

    # Grad should include layers BEFORE the hook linear
    assert main_net.linear1.weight.grad is not None
    # Grad should include layers AFTER the hook linear
    assert main_net.linear2.weight.grad is not None
    # Grad should include the hook linear
    assert hook_module.linear.weight.grad is not None
    assert hook_module.linear.bias.grad is not None


if __name__ == "__main__":
    # TODO(Adriano) add some more flags or whatever to be able to pick which integration
    # test to use.
    _test_pt_hooks_modify_inputs()
#######################################
