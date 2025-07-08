from enum import Enum

class DeviceTypes(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    GPU = "gpu" 

class DLFrameworks(Enum):
    TensorFlow = "tensorflow"
    PyTorch = "torch"

def getDeviceType(framework: DLFrameworks):

    def _torch_dev():
        import torch
        if torch.cuda.is_available():
            DEVICE = DeviceTypes.CUDA.value
        elif torch.backends.mps.is_available():
            DEVICE = DeviceTypes.MPS.value
        else:
            DEVICE = DeviceTypes.CPU.value
        
        return torch.device(DEVICE)


    def _tf_dev():
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            DEVICE = DeviceTypes.GPU.value.upper()
        else:
            DEVICE = DeviceTypes.CPU.value.upper()
        
        return tf.device(DEVICE)

    if framework is DLFrameworks.PyTorch:
        try:
            return _torch_dev()
        except ImportError:
            if framework.value == "torch":
                raise

    if framework is DLFrameworks.TensorFlow:
        try:
            return _tf_dev()
        except ImportError:
            if framework.value == "tensorflow":
                raise
    
