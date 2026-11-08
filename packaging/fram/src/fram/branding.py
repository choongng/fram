"""Fram's branding, read by Tau through the ``tau_coding.branding`` entry point.

This module must not import Tau: Tau loads it while ``tau_coding`` is still
being imported. The keys are ``tau_coding.branding.Branding`` field names.
"""

from __future__ import annotations

BRANDING = {
    "product_name": "Fram",
    "command_name": "fram",
    # A four-pointed star, read as a compass rose. One terminal cell wide,
    # like Tau's τ.
    "mark": "✦",
    "logo": "✦ fram",
}
