from __future__ import annotations

import unittest

from framework_engineer.kernel_interface_decomposer.capture_registry import (
    CAPTURE_BY_ARCHETYPE,
    CAPTURE_REGISTRY,
    CAPTURE_REGISTRY_VERSION,
    validate_capture_registry,
)


class TestCaptureRegistry(unittest.TestCase):
    def test_registry_contract_and_schema_values(self) -> None:
        validate_capture_registry()
        self.assertEqual(CAPTURE_REGISTRY_VERSION, "kid-execution-capture/v2")
        self.assertEqual(
            set(CAPTURE_BY_ARCHETYPE),
            {
                "pytorch_dispatch",
                "triton_launch",
                "cute_dsl_launch",
                "tilelang_launch",
                "tvm_ffi_call",
                "inductor_launch",
                "python_binding",
            },
        )
        self.assertEqual(len(CAPTURE_REGISTRY), len(CAPTURE_BY_ARCHETYPE))

    def test_verified_poc_common_interfaces_are_explicit(self) -> None:
        self.assertIn(
            "TorchDispatchMode.__torch_dispatch__",
            CAPTURE_BY_ARCHETYPE["pytorch_dispatch"].common_interfaces[0],
        )
        self.assertTrue(
            any(
                "JITFunction.__getitem__" in item
                for item in CAPTURE_BY_ARCHETYPE["triton_launch"].common_interfaces
            )
        )
        expected_common_interfaces = {
            "cute_dsl_launch": "cutlass.cute.compile",
            "tilelang_launch": "tilelang.JITKernel.__call__",
            "tvm_ffi_call": "tvm_ffi.module.Module",
            "inductor_launch": "CachingAutotuner.run",
            "python_binding": "Python-visible extension/binding export",
        }
        for archetype, fragment in expected_common_interfaces.items():
            with self.subTest(archetype=archetype):
                self.assertTrue(
                    any(
                        fragment in item
                        for item in CAPTURE_BY_ARCHETYPE[
                            archetype
                        ].common_interfaces
                    )
                )


if __name__ == "__main__":
    unittest.main()
