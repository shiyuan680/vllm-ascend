# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# ruff: noqa: E402
"""A layerwise reload must not materialize a host-offloaded parameter on device.

The regression: vLLM's layerwise reload moves a layer to the meta device and
materializes it again on the default device.  For an Engram table that lives in
registered host memory this allocates a full per-rank device copy (22.89 GiB for
a 384M x 256 BF16 table over 8 ranks) on top of the weights a level-2 sleep
keeps resident, so checkpoint reload and the RL weight update die with
``torch.OutOfMemoryError`` inside ``materialize_meta_tensor``.

The tests drive the real upstream reload (the layerwise machinery itself is not
stubbed) on a CPU device, and assert on the tensors ``materialize_meta_tensor``
is asked to allocate: materializing the offloaded table's shape is exactly the
allocation that runs out of device memory.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm.model_executor.model_loader.reload import layerwise as layerwise_mod
from vllm.model_executor.model_loader.reload import meta as meta_mod
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

import vllm_ascend.patch.worker.patch_offload_reload  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.embedding import (
    HOST_OFFLOAD_STORAGE_ATTR,
    register_host_offload_storage,
)

# Distinct sizes so a materialization of the table cannot be confused with the
# rest of the layer.
ROWS = 8
WIDTH = 4
TABLE_SHAPE = (ROWS, WIDTH)
VECTOR_SHAPE = (WIDTH,)
MODEL_CONFIG = SimpleNamespace(dtype=torch.bfloat16)


@pytest.fixture
def materialized(monkeypatch):
    """Records the (size, dtype) of every tensor the reload materializes."""
    calls: list[tuple[tuple[int, ...], torch.dtype]] = []
    original = meta_mod.materialize_meta_tensor

    def record(tensor: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(tensor.size()), tensor.dtype))
        return original(tensor)

    monkeypatch.setattr(meta_mod, "materialize_meta_tensor", record)
    return calls


def _copy_into_parameter(param: nn.Parameter, source: torch.Tensor) -> bool:
    """Loader that only writes the transported tensor into its destination."""
    param.data.copy_(source)
    return True


class _HostOffloadedEngram(nn.Module):
    """Minimal stand-in for an Ascend Engram table offloaded to host memory."""

    def __init__(self) -> None:
        super().__init__()
        # Stands in for the registered host range (`HostUvaBuffer.tensor`).
        host_weight = torch.zeros(TABLE_SHAPE, dtype=torch.bfloat16)
        self._codes_uva = SimpleNamespace(tensor=host_weight)
        self.weight = nn.Parameter(host_weight, requires_grad=False)
        self.weight.weight_loader = _copy_into_parameter
        self.bias = nn.Parameter(torch.zeros(VECTOR_SHAPE), requires_grad=False)
        self.bias.weight_loader = _copy_into_parameter
        register_host_offload_storage(self)


class _EngramLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.engram = _HostOffloadedEngram()
        self.seen = nn.Parameter(torch.zeros(VECTOR_SHAPE), requires_grad=False)
        self.seen.weight_loader = _copy_into_parameter
        self.register_buffer("host", torch.zeros(VECTOR_SHAPE), persistent=False)
        self.host.weight_loader = _copy_into_parameter


def test_reload_keeps_offloaded_parameter_off_the_device(materialized):
    layer = _EngramLayer()
    host_storage = layer.engram._codes_uva.tensor
    storage_ptr = host_storage.untyped_storage().data_ptr()
    assert layer.engram.weight.untyped_storage().data_ptr() == storage_ptr

    record_metadata_for_reloading(layer)

    update = torch.arange(ROWS * WIDTH, dtype=torch.bfloat16).reshape(TABLE_SHAPE)
    initialize_layerwise_reload(layer)
    # The host range survives the move to meta through the parameter attribute.
    assert layer.engram.weight.is_meta
    assert layer.engram.weight.__dict__[HOST_OFFLOAD_STORAGE_ATTR] is host_storage

    # The reload's own loader wrapper drives the layer: it counts the load and
    # finalizes the layer, which is where the device materialization happens.
    assert layer.engram.weight.weight_loader(layer.engram.weight, update) is True
    finalize_layerwise_reload(layer, MODEL_CONFIG)

    reloaded = layer.engram.weight
    assert not reloaded.is_meta
    assert reloaded.device.type == "cpu"
    # The update landed in the registered host range, not in a device copy of it.
    assert reloaded.untyped_storage().data_ptr() == storage_ptr
    torch.testing.assert_close(host_storage, update)
    torch.testing.assert_close(reloaded.detach(), update)
    # The table's shape was never materialized: that allocation is the OOM.
    assert TABLE_SHAPE not in [shape for shape, _ in materialized]
    # The reload left no deferred state behind.
    assert not hasattr(layerwise_mod.get_layerwise_info(layer.engram), "host_offload_params")


def test_reload_still_materializes_the_rest_of_the_layer(materialized):
    layer = _EngramLayer()
    seen_ptr = layer.seen.untyped_storage().data_ptr()
    host_ptr = layer.host.untyped_storage().data_ptr()
    update = torch.ones(VECTOR_SHAPE, dtype=torch.bfloat16)

    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)
    assert layer.seen.is_meta and layer.host.is_meta

    # Drive a non-offloaded tensor through the same reload.
    layer.seen.weight_loader(layer.seen, update)
    finalize_layerwise_reload(layer, MODEL_CONFIG)

    assert not layer.seen.is_meta
    assert not layer.host.is_meta
    assert layer.seen.untyped_storage().data_ptr() == seen_ptr
    assert layer.host.untyped_storage().data_ptr() == host_ptr
    torch.testing.assert_close(layer.seen.detach(), update)
    # Non-offloaded tensors keep the upstream materialization...
    assert VECTOR_SHAPE in [shape for shape, _ in materialized]
    # ... while the offloaded table still does not.
    assert TABLE_SHAPE not in [shape for shape, _ in materialized]
