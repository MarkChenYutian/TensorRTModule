"""
An easy-to-use encapsulation of pytorch module into TensorRT by exporting to ONNX
and then compile ONNX using tensorrt.
"""

from datetime import datetime
from typing import Mapping, Sequence, Callable
from pathlib import Path
from .Sandbox import Sandbox
from .Logger import Logger


import torch
import tensorrt as trt


AsyncInferenceContext = dict[str, torch.Tensor]


class TensorRTModule(torch.nn.Module):
    def __init__(self,
        serialized_module: bytes | trt.IStreamReader,
        input_names: tuple[str, ...],
        predict_output_shapes: Callable[[Mapping[str, torch.Size], ], Mapping[str, torch.Size]],
        
        # Unimportant inputs
        log_level: trt.ILogger.Severity = trt.Logger.INFO
    ):
        """
        Initialize tensorrt engine, runtime and I/O allocation.
        """
        super().__init__()
        self.runtime = trt.Runtime(logger=trt.Logger(log_level))
        self.engine  = self.runtime.deserialize_cuda_engine(serialized_module)
        self.context = self.engine.create_execution_context()
        self.stream  = torch.cuda.Stream()
        self.input_names = input_names
        
        self.predict_output_shapes = predict_output_shapes
        self.profile_cache: dict[tuple[torch.Size, ...], int] = dict()
        self.io_alloc_cache: dict[str, dict[torch.Size, torch.Tensor]] = dict()
        
        self.has_job_running = False

    @classmethod
    def compile_torch_module(
        cls,
        module: torch.nn.Module,
        cache_to: Path,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        inputs: tuple[torch.Tensor, ...],
        predict_output_shapes: Callable[[Mapping[str, torch.Size], ], Mapping[str, torch.Size]], 
        create_optimization_profiles: Callable[[trt.Builder, ], tuple[trt.IOptimizationProfile, ...]],
        dynamic_axis: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None = None,
        additional_config_setup: Callable[[trt.IBuilderConfig,], trt.IBuilderConfig] | None = None,
        as_fp16: bool = False,
    ) -> 'TensorRTModule':
        """
        Export a PyTorch module to ONNX, and then compile as TensorRT engine. Will automatically reuse previous 
        results when possible.
        ---
        Input
        * `module`       - The pytorch module to compile to TensorRT
        * `cache_to`     - The path to store intermediate results and final TensorRT engine executable to
        * `input_names`  - Human-readable names to each input (Tensor) of the module
        
                NOTE: this must be complete, incomplete naming of input tensor will cause failure in TRT
                      execution.
        
        * `output_names` - Human-readable names to each output (Tensor) of the module
        
                NOTE: this must be complete, incomplete naming of output tensor will cause failure in TRT 
                      execution.
        
        * `inputs`       - Example inputs passed to the model used to trace module computation graph.
        
        * `predict_output_shapes`     - A function, given the shape(s) of input(s), return the shape(s) of 
                output(s). Specifically, the function will have signature of
                
                `predict_output_shapes({keyof input_names : torch.Size}) -> {keyof output_names : torch.Size}`
                
                the output of function must match the `output_names` provided exactly, or TRT execution may 
                fail.
        
        * `create_optimization_profiles` - A function, given the builder, should register and create optimization
                profile(s) for the network. Typically optimization profile will include the shape spec of each 
                tensor (matched with the `input_names` argument), including `min_shape`, `max_shape` and optimum
                shape `opt_shape`. 
                
                Reference:
                - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/OptimizationProfile.html
        
        * `as_fp16` - Compile the model with half-precision (fp16) using TensorRT engine. Using this option does
                *not* require converting module as `fp16` prior to compiling. The TRT engine is designed to convert
                fp32 model to fp16 automatically with `as_fp16=True`. **Using FP16 may degrade the performance of model**
        """
        if cache_to.exists():
            cache_box = Sandbox.load(cache_to)
            cache_box.config = dict(build_time=cache_box.config.build_time, last_use=cls._get_curr_time())
        else:
            cache_box = Sandbox(cache_to)
            cache_box.config = dict(build_time=cls._get_curr_time(), last_use=cls._get_curr_time())
        
        network_hash = cls._get_module_onnx_md5(cache_box, module, input_names, output_names, inputs, dynamic_axis)
        
        # Get desired optimization profile and check if this is compiled before on the certain device
        test_logger  = trt.Logger(trt.Logger.INTERNAL_ERROR)
        test_builder = trt.Builder(test_logger)
        test_profiles = create_optimization_profiles(test_builder)
        cache_box = cls._select_compiled_engine_cache(cache_box, test_profiles, input_names, as_fp16, network_hash)
        # End
        
        # Export as ONNX
        if cache_box.path("export_network.onnx").exists():
            Logger.write("info", f"Using cache at {str(cache_box.path('export_network.onnx'))}")
        else:
            Logger.write("info", f"Can't find cache at {str(cache_box.path('export_network.onnx'))}, exporting network as ONNX...")
            inference_module = module.cuda().eval()
            torch.onnx.export(
                inference_module, inputs,
                str(cache_box.path("export_network.onnx")),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axis
            )
        
        # Convert to TensorRT engine
        trt_cache_path = cache_box.path("export_engine.trt")
        if trt_cache_path.exists():
            Logger.write("info", f"Using cache at {str(trt_cache_path)}")
        else:
            assert cache_box.path("export_network.onnx").exists()
            Logger.write("info", f"Can't find cache at {str(trt_cache_path)}, compiling a new TensorRTEngine")
            builder = trt.Builder(test_logger)
            network = builder.create_network()
            parser  = trt.OnnxParser(network, test_logger)
            
            onnx_buffer = cache_box.path("export_network.onnx").read_bytes()
            parser.parse(onnx_buffer)
            
            config  = builder.create_builder_config()
            if as_fp16:
                Logger.write("info", "Compiling as FP16 module.")
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            
            if additional_config_setup is not None:
                config = additional_config_setup(config)
            
            profiles = create_optimization_profiles(builder)
            Logger.write(
                "info", 
                f"Provided {len(profiles)} optimization profiles"
                "\n\t* ".join([cls._iprofile_repr(profile, input_names) for profile in profiles])
            )
            
            for profile in profiles: config.add_optimization_profile(profile)
            
            Logger.write("info", f"Start compiling TensorRT engine...")
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise SystemError("Failed to build TensorRT engine. Check previous stderr for potential problems.")
            Logger.write("info", f"Finished compiling : )")
            trt_cache_path.write_bytes(engine) #type: ignore
        
        bytes_engine = trt_cache_path.read_bytes()
        return cls(bytes_engine, input_names, predict_output_shapes)

    @staticmethod
    def _select_compiled_engine_cache(
        cachebox: Sandbox,
        profiles: tuple[trt.IOptimizationProfile, ...],
        tensor_names: tuple[str, ...],
        as_fp16: bool,
        onnx_md5_hash: str,
    ) -> Sandbox:
        profile_string = "\n".join(
            [TensorRTModule._iprofile_repr(profile, tensor_names) for profile in profiles]
        )
        device_name = torch.cuda.get_device_name()
        
        for child_box in cachebox.get_children():
            if not hasattr(child_box.config, "EngineProfile"): continue
            if not hasattr(child_box.config, "DeviceName"): continue
            if not hasattr(child_box.config, "FP16"): continue
            if not hasattr(child_box.config, "ONNXmd5"): continue
            if  child_box.config.EngineProfile == profile_string and\
                child_box.config.DeviceName == device_name and\
                child_box.config.FP16 == as_fp16 and\
                child_box.config.ONNXmd5 == onnx_md5_hash:
                return child_box
        
        child_box = cachebox.new_child(f"EngineProfile_{len(cachebox.get_children())}")
        child_box.config = {"EngineProfile": profile_string, "DeviceName": device_name, "FP16": as_fp16,
                            "ONNXmd5": onnx_md5_hash}
        return child_box

    @staticmethod
    def _get_curr_time() -> str: return datetime.now().strftime("%m_%d_%H%M%S")
        
    @staticmethod
    def _is_compatible_shape(tensor_shape: torch.Size, min_shape: trt.Dims, max_shape: trt.Dims) -> bool:
        assert len(tensor_shape) == len(min_shape) == len(max_shape), "Different number of dimensions in tensor shape!"
        
        min_shape_iter = (min_shape[idx] for idx in range(len(min_shape)))
        max_shape_iter = (max_shape[idx] for idx in range(len(max_shape)))
        
        for t_size, min_size, max_size in zip(tensor_shape, min_shape_iter, max_shape_iter):
            if not (min_size <= t_size <= max_size): return False
        else:
            return True

    @staticmethod
    def _get_module_onnx_md5(
        cache_box: Sandbox, module: torch.nn.Module, 
        input_names: tuple[str, ...], output_names: tuple[str, ...],
        inputs: tuple[torch.Tensor, ...],
        dynamic_axis: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None = None
    ) -> str:
        inference_module = module.cuda().eval()
        torch.onnx.export(
            inference_module, inputs,
            str(cache_box.path("temp.onnx")),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axis
        )
        onnx_hash = TensorRTModule._md5_hash_file(str(cache_box.path("temp.onnx")))
        cache_box.path("temp.onnx").unlink(missing_ok=False)
        return onnx_hash

    @staticmethod
    def _md5_hash_file(file_name: str) -> str:
        import hashlib
        md5 = hashlib.md5(usedforsecurity=False)
        CHUNK_SIZE = 65536
        with open(file_name, "rb") as f:
            while (data := f.read(CHUNK_SIZE)):
                md5.update(data)
        return md5.hexdigest()

    @staticmethod
    def _profile_repr(engine: trt.ICudaEngine, profile_id: int, tensor_name: tuple[str, ...]) -> str:
        tensors_repr: list[str] = []
        for name in tensor_name:
            min_shape, _, max_shape = engine.get_tensor_profile_shape(name, profile_id)
            tensors_repr.append(f"Tensor(name={name}, min_shape={min_shape}, max_shape={max_shape})")
        return f"Profile(id={profile_id}, cfg={tensors_repr})"
    
    @staticmethod
    def _iprofile_repr(profile: trt.IOptimizationProfile, tensor_name: tuple[str, ...]) -> str:
        tensors_repr: list[str] = []
        for name in tensor_name:
            min_shape, opt_shape, max_shape = profile.get_shape(name)
            tensors_repr.append(f"Tensor(name={name}, min_shape={min_shape}, opt_shape={opt_shape}, max_shape={max_shape})")
        return f"IOptimizationProfile({tensors_repr})"

    def select_profile(self, tensor_shape: tuple[torch.Size, ...], tensor_name: tuple[str, ...]) -> int:
        assert len(tensor_shape) == len(tensor_name)
        
        if tensor_shape in self.profile_cache:
            return self.profile_cache[tensor_shape] # Cache Hit!
        
        for pidx in range(self.engine.num_optimization_profiles):
            shape_compatible = True
            for shape, name in zip(tensor_shape, tensor_name):
                min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, pidx)
                shape_compatible &= self._is_compatible_shape(shape, min_shape, max_shape)
            
            if shape_compatible: return pidx
            
        raise ValueError(
            f"Unable to find optimization profile that fits the shape(s) {tensor_shape}."
            f"Provided shapes are \n"
            "\n".join([self._profile_repr(self.engine, pidx, tensor_name) for pidx in range(self.engine.num_optimization_profiles)])
        )

    def alloc_io_tensor(self, tensor_shape: torch.Size, tensor_name: str) -> torch.Tensor:
        if (tensor_name in self.io_alloc_cache) and (tensor_shape in self.io_alloc_cache[tensor_name]):
            return self.io_alloc_cache[tensor_name][tensor_shape]
        
        self.io_alloc_cache[tensor_name] = self.io_alloc_cache.get(tensor_name, dict()) | {
            tensor_shape : torch.empty(
                tensor_shape,
                memory_format=torch.contiguous_format,
                dtype=torch.float32,
                device='cuda'
            )
        }
        return self.io_alloc_cache[tensor_name][tensor_shape]

    def async_launch_forward(self, *inputs: torch.Tensor) -> AsyncInferenceContext:
        assert not self.has_job_running, "Async forwarding with TensorRT does not support multiple forwards simultaneously. This will cause data racing and corrupt results."
        
        self.has_job_running = True
        input_shapes = tuple(t.shape for t in inputs)
        profile_id = self.select_profile(input_shapes, self.input_names)
        self.context.set_optimization_profile_async(profile_id, self.stream.cuda_stream)
        
        # Setup execution context
        input_profile: dict[str, torch.Size] = dict()
        for shape, name in zip(input_shapes, self.input_names):
            input_profile[name] = shape
            self.context.set_input_shape(name, trt.Dims([*shape]))
        output_profile = dict(**self.predict_output_shapes(input_profile))

        io_profile = input_profile | output_profile        
        io_tensors = {
            name : self.alloc_io_tensor(shape, name)
            for name, shape in io_profile.items()
        }
        
        # Fill inputs
        for input, name in zip(inputs, self.input_names):
            io_tensors[name].copy_(input.contiguous())
        
        # Bind I/O to TensorRT execution context
        incomplete_setup = False
        for io_idx in range(self.engine.num_io_tensors):
            io_name = self.engine.get_tensor_name(io_idx)
            if io_name not in io_tensors:
                Logger.write("error", f"I don't see predicted shape for I/O tensor with name {io_name}. The predicted output shape list is not complete.")
                incomplete_setup = True
            else:
                self.context.set_tensor_address(io_name, io_tensors[io_name].data_ptr())
        
        if incomplete_setup:
            raise ValueError("Provided method `predict_output_shapes` does not generate shape for all outputs. Abort.")

        self.context.execute_async_v3(self.stream.cuda_stream)
        
        return {name: io_tensors[name] for name in output_profile}
    
    def async_receive_forward(self, ctx: AsyncInferenceContext) -> dict[str, torch.Tensor]:
        assert self.has_job_running, "Can't receive forward without any job actually running on TRT runtime."
        
        # Create output slots to avoid contamination of I/O buffer area.
        output_tensors = {
            name : torch.empty_like(tensor)
            for name, tensor in ctx.items()
        }
        self.stream.synchronize()
        
        for name, tensor in ctx.items():
            output_tensors[name].copy_(tensor)
        
        self.has_job_running = False
        return output_tensors

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.async_launch_forward(*inputs)
        return self.async_receive_forward(ctx)
