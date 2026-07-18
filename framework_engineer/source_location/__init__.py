"""Source-location stages for Step 0.5 ``locate-kernel-source``.

Layer 1 enriches KID schemas with deterministic Python interface locations;
Layer 3 extracts resolved files into per-kernel workspaces. Layer 2 remains an
agent fallback for unresolved implementation, binding, and header layers.
"""
