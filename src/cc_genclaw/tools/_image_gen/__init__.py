"""All image-generation scripts behind the t2i / i2i atomic tools.

This directory holds everything needed to turn a prompt (+ optional
reference images) into a PNG: the protocol handlers, retry policy, and
the provider config. Self-contained: configured entirely by
``providers.yaml`` in this dir.
"""

from .core import generate_image

__all__ = ["generate_image"]
