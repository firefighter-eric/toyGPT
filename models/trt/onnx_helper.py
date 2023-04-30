import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype=np.float32):
        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)

        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes,
                               dtype=self.target_dtype)  # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch):  # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output


def convert_onnx_to_engine(onnx_filename, engine_filename=None, max_batch_size=32, max_workspace_size=1 << 30,
                           fp16_mode=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    # builder.max_workspace_size = max_workspace_size
    # builder.fp16_mode = fp16_mode
    # builder.max_batch_size = max_batch_size

    print("Parsing ONNX file.")
    success = parser.parse_from_file(onnx_filename)
    print('success:', success)
    for error in range(parser.num_errors):
        print(parser.get_error(error))

    print("Building TensorRT engine. This may take a few minutes.")
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
    serialized_engine = builder.build_serialized_network(network, config)

    if engine_filename:
        with open(engine_filename, 'wb') as f:
            f.write(serialized_engine)

    return serialized_engine, logger


if __name__ == '__main__':
    convert_onnx_to_engine('data/trt/opt-125m.onnx',
                           engine_filename='data/trt/opt-125m.fp16.trt',
                           fp16_mode=True)
