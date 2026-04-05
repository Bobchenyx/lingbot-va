# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_robotwin_cfg import va_robotwin_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
}