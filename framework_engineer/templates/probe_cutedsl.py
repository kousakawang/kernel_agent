from __future__ import annotations


def main() -> None:
    """Probe CuTe DSL availability.

    Framework Engineer should extend this probe to compile/run a minimal
    task-like CuTe DSL kernel in the target environment. This template avoids
    assuming a single import path because CuTe DSL packaging differs across
    installations.
    """
    errors = []
    imported = False

    for module_name in ("cutlass", "cutlass.cute", "cute"):
        try:
            module = __import__(module_name, fromlist=["*"])
            print(f"import ok: {module_name} -> {getattr(module, '__file__', '<unknown>')}")
            imported = True
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    if not imported:
        raise RuntimeError("CuTe DSL import failed:\n" + "\n".join(errors))

    print(
        "cutedsl available: import succeeded. "
        "Set usable_for_task=true only after adding/running a minimal compile+run probe."
    )


if __name__ == "__main__":
    main()
