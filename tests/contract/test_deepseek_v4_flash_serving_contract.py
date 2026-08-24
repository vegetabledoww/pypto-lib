# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_KERNEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "deepseek_v4_flash_mtp"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def serving_contract_module():
    name = "_pypto_lib_test_deepseek_v4_serving_contract"
    module = _load_module(name, _KERNEL_DIR / "serving_contract.py")
    yield module
    sys.modules.pop(name, None)


def test_deepseek_v4_flash_serving_capabilities(serving_contract_module) -> None:
    contract = serving_contract_module.DEEPSEEK_V4_FLASH_SERVING_CONTRACT

    assert contract.schema_version == "1"
    assert contract.prefill_tile_tokens == 128
    assert contract.max_prefill_tokens_per_request == 8192
    assert contract.max_prefill_requests_per_partition == 1
    assert contract.requires_homogeneous_prefill_decode is True


@pytest.mark.parametrize(
    ("active_tokens", "expected"),
    [(1, 128), (128, 128), (129, 256), (8191, 8192), (8192, 8192)],
)
def test_deepseek_v4_flash_prefill_padding(
    serving_contract_module,
    active_tokens: int,
    expected: int,
) -> None:
    contract = serving_contract_module.DEEPSEEK_V4_FLASH_SERVING_CONTRACT

    assert contract.padded_prefill_tokens(active_tokens) == expected


@pytest.mark.parametrize("active_tokens", [0, -1, 8193])
def test_deepseek_v4_flash_prefill_padding_rejects_invalid_extents(
    serving_contract_module,
    active_tokens: int,
) -> None:
    contract = serving_contract_module.DEEPSEEK_V4_FLASH_SERVING_CONTRACT

    with pytest.raises(ValueError):
        contract.padded_prefill_tokens(active_tokens)


def test_deepseek_v4_flash_config_uses_serving_contract(
    serving_contract_module,
) -> None:
    previous = sys.modules.get("serving_contract")
    sys.modules["serving_contract"] = serving_contract_module
    config_name = "_pypto_lib_test_deepseek_v4_config"
    try:
        config = _load_module(config_name, _KERNEL_DIR / "config.py")
    finally:
        sys.modules.pop(config_name, None)
        if previous is None:
            sys.modules.pop("serving_contract", None)
        else:
            sys.modules["serving_contract"] = previous

    contract = serving_contract_module.DEEPSEEK_V4_FLASH_SERVING_CONTRACT
    assert config.PREFILL_BATCH == contract.max_prefill_requests_per_partition
    assert config.PREFILL_SEQ == contract.prefill_tile_tokens
