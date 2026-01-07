#!/usr/bin/env python3
"""
Policy package - Pluggable policies for action inference.

Author: Jun Lv, Zheng Wang
"""

from .base import BasePolicy
from .tiny_mlp import TinyMLPPolicy
from .openpi_policy import OpenPiPolicy
try:
    from .diffusion_policy import DiffusionPolicy
    #from .openpi_policy import OpenPiPolicy
except ImportError:
    DiffusionPolicy = None
    OpenPiPolicy = None
try:
    from .diffusion_policy_iphone_umi import DiffusionPolicyIPhoneUMI
except Exception:
    DiffusionPolicyIPhoneUMI = None

# Policy class mapping - add new policy classes here
POLICY_CLASSES = {
    'TinyMLPPolicy': TinyMLPPolicy,
    'DiffusionPolicy': DiffusionPolicy,
    'DiffusionPolicyIPhoneUMI': DiffusionPolicyIPhoneUMI,
    "OpenPiPolicy": OpenPiPolicy,
}

__all__ = ['BasePolicy', 'TinyMLPPolicy', 'DiffusionPolicy', 'DiffusionPolicyIPhoneUMI','POLICY_CLASSES', 'OpenPiPolicy']