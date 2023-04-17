#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorrt as trt
from typing import List


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_trt_plugins() -> List[str]:
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    registry = trt.get_plugin_registry()
    plugins = [pl.name for pl in registry.plugin_creator_list]
    plugins = sorted(plugins)
    return plugins


def main():
    all_plugins = get_trt_plugins()
    print_str = "\n".join(all_plugins)
    print(print_str)


if __name__ == "__main__":
    main()
