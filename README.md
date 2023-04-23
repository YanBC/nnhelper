# Install
```bash
pip install --upgrade pip
pip install --upgrade pipx
pipx ensurepath
pipx install --verbose .
```

# Command line tools
After installation, the following tools should be available from command line:
- `info_onnx`: show onnx model inputs and outputs
- `info_trt`: show tensorrt model inputs and outputs
- `list_trt_plugins`: list all available tensorrt plugins
