import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from pycuda.autoinit import context

context = context


class TRTModel:
    def __init__(self, file):
        self.load(file)

        self.stream = None

    def load(self, file):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(open(file, "rb").read())
        self.engine = engine
        self.context = engine.create_execution_context()

        self.output_dtype = trt.nptype(engine.get_tensor_dtype('output'))
        self.n_vocab = engine.get_tensor_shape('output')[2]
        print(engine.get_tensor_shape('input_ids'))
        print(engine.get_tensor_shape('output'))
        print("n_vocab: ", self.n_vocab)

    def info(self):
        engine = self.engine
        for binding in engine:
            size = engine.get_tensor_shape(binding)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            print(f"[Info] binding: {binding}, size: {size}, dtype: {dtype}")

    def allocate_memory(self):
        # self.context.active_optimization_profile = 0
        BATCH_SIZE, SEQ_LENGTH = 1, 1024
        self.input = np.empty([BATCH_SIZE, SEQ_LENGTH], dtype=np.int32)
        self.output = np.empty([BATCH_SIZE, SEQ_LENGTH, self.n_vocab], dtype=self.output_dtype)

        # self.context.set_binding_shape(0, (BATCH_SIZE, SEQ_LENGTH))
        self.context.set_input_shape('input_ids', (BATCH_SIZE, SEQ_LENGTH))

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * self.input.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]
        print('bindings: ', self.bindings)
        self.stream = cuda.Stream()

    def __call__(self, batch):  # result gets copied into output
        if self.stream is None:
            self.allocate_memory()

        # # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # # Syncronize threads
        self.stream.synchronize()

        return self.output

    def sync_predict(self, batch):
        BATCH_SIZE, SEQ_LENGTH = batch.shape
        BATCH_SIZE, SEQ_LENGTH = 1, 64
        self.input = np.empty([BATCH_SIZE, SEQ_LENGTH], dtype=np.int32)
        self.output = np.empty([BATCH_SIZE, SEQ_LENGTH, self.n_vocab], dtype=np.float32)

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * self.input.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]

        # Transfer input data to device
        cuda.memcpy_htod(self.d_input, batch)
        # Execute model
        self.context.execute_v2(self.bindings)
        # Transfer predictions back
        cuda.memcpy_dtoh(self.output, self.d_output)

        return self.output


if __name__ == '__main__':
    trt_model = TRTModel('data/trt/opt-125m.trt')
    trt_model.info()
    o = trt_model(np.ones([1, 1024], dtype=np.int32))
    # o = trt_model.sync_predict(np.ones([1, 150], dtype=np.int32))
