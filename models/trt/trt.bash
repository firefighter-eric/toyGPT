trtexec --onnx=data/trt/opt-125m.onnx \
  --saveEngine=data/trt/opt-125m.trt \
  --explicitBatch \
  --minShapes=input_ids:1x1 \
  --optShapes=input_ids:1x128 \
  --maxShapes=input_ids:1x1024


trtexec --loadEngine=data/trt/opt-125m.trt \
  --shapes=input_ids:1x1024