# TensorRTModule

A simple wrapper to TensorRT SDK that automatically compiles PyTorch module to TRT engine and reuse cache when possible.

Example Usage


```python
import tensorrt as trt
from Utility.TensorRTModule import TensorRTModule


TENSOR_RT_AOT_RESULT_PATH = "./cache"
DEVICE = "cuda:0"


def tensorrt_compile_with_cache(model: torch.nn.Module, inp_A: torch.Tensor, inp_B: torch.Tensor) -> TensorRTModule:
    def fill_optimization_profile(builder: trt.Builder) -> tuple[trt.IOptimizationProfile, ...]:
        static_profile = builder.create_optimization_profile()
        static_shape   = trt.Dims([d for d in inp_A.shape])
        
        static_profile.set_shape("inp_A", min=static_shape, opt=static_shape, max=static_shape)
        static_profile.set_shape("inp_B", min=static_shape, opt=static_shape, max=static_shape)
        return (static_profile,)

    def predict_output_shape(input_shapes: Mapping[str, torch.Size]) -> Mapping[str, torch.Size]:
        assert "inp_A" in set(input_shapes.keys())
        assert "inp_B" in set(input_shapes.keys())
        
        B, _, H, W = input_shapes["inp_A"]
        return {
            "flow"       : torch.Size((B, 2, H, W)),
            "flow_sm"    : torch.Size((B, 2, H//8, W//8)),
            "flow_cov"   : torch.Size((B, 2, H, W)),
            "flow_cov_sm": torch.Size((B, 2, H//8, W//8))
        }
    
    compiled_model = TensorRTModule.compile_torch_module(
        model.float(), TENSOR_RT_AOT_RESULT_PATH, 
        input_names=("inp_A", "inp_B"),
        output_names=("flow", "flow_sm", "flow_cov", "flow_cov_sm"),
        inputs=(inp_A.float().to(DEVICE), inp_B.float().to(DEVICE)),
        predict_output_shapes=predict_output_shape,
        create_optimization_profiles=fill_optimization_profile
    )
    return compiled_model

module = FlowFormer(...)
trt_module = tensorrt_compile_with_cache(module, torch.empty((1, 3, 640, 640)), torch.empty((1, 3, 640, 640)))
output = trt_module(inp_A, inp_B)

print(output["flow"].shape, output["flow_cov"].shape)
```
