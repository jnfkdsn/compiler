import numpy as np

from ..tensor import Tensor
from ..tracer import TraceContext
from ..runtime import JITEngine
from ..nn.modules import Module
from ..nn.jit import _collect_parameter_slots, _TensorSlot

class TracedModule(Module):
    """
    Like torch.jit.ScriptModule (PyTorch), this wraps an nn.Module that has been traced
    and compiled to C++. It performs significantly faster than eager execution.
    """
    def __init__(self, original_module, example_inputs, use_hpc=True):
        super().__init__()
        self.original_module = original_module
        
        # Collect parameters to treat them as hidden inputs to the graph
        self._param_slots = _collect_parameter_slots(original_module)
        originals = [slot.get() for slot in self._param_slots]
        self._param_input_ids = []
        
        with TraceContext() as ctx:
            traced_inputs = []
            for i, x in enumerate(example_inputs):
                val = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
                t = Tensor.from_numpy(val, name=f"in_{i}")
                traced_inputs.append(t)
            
            # Temporarily replace module parameters with explicitly traced Inputs
            traced_params = []
            for i, (slot, p) in enumerate(zip(self._param_slots, originals)):
                tp = Tensor.from_numpy(p.data, name=f"param_{i}", requires_grad=False)
                slot.set(tp)
                traced_params.append(tp)
                
            try:
                # Trace the forward pass
                out = original_module(*traced_inputs)
                
                # Collect outputs
                if isinstance(out, Tensor):
                    out.mark_as_output()
                elif isinstance(out, (tuple, list)):
                    for o in out:
                        if isinstance(o, Tensor):
                            o.mark_as_output()
                else:
                    raise RuntimeError("Module trace failed: Output must be Tensor or Tuple[Tensor]")
                
                self.graph = ctx.graph
            finally:
                # Restore original eager parameters
                for slot, orig in zip(self._param_slots, originals):
                    slot.set(orig)
            
        self.engine = JITEngine(use_hpc_template=use_hpc)
        self.compiled_module = self.engine.compile_graph(self.graph)
        
    def forward(self, *args):
        # Gather user inputs
        np_args = [a.data if isinstance(a, Tensor) else np.asarray(a, dtype=np.float32) for a in args]
        
        # Append parameter inputs
        np_params = [slot.get().data for slot in self._param_slots]
        full_args = np_args + np_params
        
        out_data = self.compiled_module.run(*full_args)
        
        # Determine if we return a single tensor or tuple based on outputs
        output_nodes = [n for n in self.graph.nodes() if n.id in self.graph.output_ids]
        if len(output_nodes) == 1:
            return Tensor(data=out_data[0] if isinstance(out_data, list) else out_data, node=None, requires_grad=False)
        else:
            return tuple(Tensor(data=o, node=None, requires_grad=False) for o in out_data)

class TracedFunction:
    """
    Like PyTorch's traced python function, wrapping a pure functional computation graph.
    """
    def __init__(self, func, example_inputs, use_hpc=True):
        with TraceContext() as ctx:
            traced_inputs = []
            for i, x in enumerate(example_inputs):
                val = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
                t = Tensor.from_numpy(val, name=f"in_{i}")
                traced_inputs.append(t)
            
            out = func(*traced_inputs)
            
            if isinstance(out, Tensor):
                out.mark_as_output()
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, Tensor):
                        o.mark_as_output()
            else:
                raise RuntimeError("Function trace failed: Output must be Tensor or Tuple[Tensor]")
            
            self.graph = ctx.graph
            
        self.engine = JITEngine(use_hpc_template=use_hpc)
        self.compiled_module = self.engine.compile_graph(self.graph)
        
    def __call__(self, *args):
        np_args = [a.data if isinstance(a, Tensor) else np.asarray(a, dtype=np.float32) for a in args]
        out_data = self.compiled_module.run(*np_args)
        
        output_nodes = [n for n in self.graph.nodes() if n.id in self.graph.output_ids]
        if len(output_nodes) == 1:
            return Tensor(data=out_data[0] if isinstance(out_data, list) else out_data, node=None, requires_grad=False)
        else:
            return tuple(Tensor(data=o, node=None, requires_grad=False) for o in out_data)

def trace(func_or_module, example_inputs, use_hpc=True):
    """
    Trace a function or nn.Module and return an optimized JIT executable.
    This works exactly like `torch.jit.trace`.
    
    Args:
        func_or_module: The Python function or nn.Module to trace.
        example_inputs: A tuple of example inputs (Tensors or numpy arrays).
        use_hpc: Whether to use OpenMP+SIMD optimized C++ templates.
        
    Returns:
        A TracedModule or TracedFunction that will execute purely natively in C++.
    """
    if isinstance(func_or_module, Module):
        return TracedModule(func_or_module, example_inputs, use_hpc=use_hpc)
    elif callable(func_or_module):
        return TracedFunction(func_or_module, example_inputs, use_hpc=use_hpc)
    else:
        raise TypeError("trace() expects a Python function or nn.Module.")
