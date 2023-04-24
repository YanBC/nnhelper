# Install
```bash
git clone https://github.com/YanBC/nnhelper.git
cd nnhelper
make
```

# Command line tools
After installation, the following tools should be available from command line:
- `info_onnx`: show onnx model inputs and outputs
- `info_trt`: show tensorrt model inputs and outputs
- `list_trt_plugins`: list all available tensorrt plugins
- `device_query`: list properties of CUDA enabled GPUs

By default, these tools are all installed under `~/.local/bin/`
