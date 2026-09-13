# 모델 변환과 경량화 실습(ONNX/TensorRT)

학습이 끝난 PyTorch 모델을 그대로 서빙하는 것은 가장 간단하지만, 대개 가장 비싼 선택이다. Python 인터프리터와 학습 프레임워크의 오버헤드가 그대로 남고, 하드웨어별 최적화(연산자 융합, 정밀도 축소, 커널 튜닝)의 혜택을 받지 못한다. 책 7.2(모델 배포)와 12.2.4(LLM 최적화 전략)는 배포 환경 선택과 양자화·증류 같은 경량화 개념을 소개한다. 이 문서는 그 개념을 실제 코드로 옮겨, [모델 구조 이해하기](./understanding-model-architecture.md)에서 살펴본 ResNet-18을 ONNX로 내보내고 ONNX Runtime과 TensorRT로 실행하기까지의 과정을 단계별로 다룬다.

> 예제는 PyTorch 2.x, torchvision 0.13 이상, onnx 1.16 이상, onnxruntime 1.18 이상을 기준으로 한다. 출력 결과는 실행 환경(CPU 종류, GPU 유무)에 따라 달라진다.

<br>

## 1. 변환 파이프라인 개요

```
PyTorch 모델 (.pt / nn.Module)
   │  torch.onnx.export
   ▼
ONNX 그래프 (.onnx) ─ 프레임워크 중립 중간 표현, Netron으로 시각화
   │
   ├─▶ ONNX Runtime ─ 범용 실행 엔진 (CPU / CUDA / DirectML 등)
   ├─▶ TensorRT 엔진 (.engine / .plan) ─ NVIDIA GPU 전용 최적화
   ├─▶ OpenVINO ─ Intel CPU / iGPU
   └─▶ Core ML / TFLite / NCNN ─ 모바일·엣지
```

ONNX(Open Neural Network Exchange)는 연산자(operator)와 텐서로 이루어진 계산 그래프의 표준 형식이다. 학습 프레임워크와 실행 엔진을 분리해 주므로, 한 번 ONNX로 내보내면 하드웨어별 최적 엔진을 골라 실행할 수 있다. 서빙 프레임워크(Triton Inference Server, KServe 등)도 ONNX와 TensorRT 엔진을 백엔드로 직접 지원한다.

<br>

## 2. ONNX로 내보내기

**설치**
```
pip install onnx onnxruntime onnxscript
```

**예제 코드**
```Python
import torch
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # 배치 크기를 가변으로
    opset_version=17,
    dynamo=False,  # 기존 TorchScript 기반 exporter 사용
)
```

**핵심 인자**

- `model.eval()`: Dropout·BatchNorm이 추론 모드로 동작해야 한다. 학습 모드로 내보내면 결과가 달라진다.
- `dynamic_axes`: 지정하지 않으면 더미 입력의 형태(배치 1)가 고정된다. 서빙 시 배치 크기가 바뀌면 오류가 나므로 가변으로 둘 축을 명시한다. 이미지 크기가 가변이면 높이·너비 축도 추가한다.
- `opset_version`: ONNX 연산자 집합 버전. 실행 엔진이 지원하는 최신 버전을 쓰되, TensorRT 등 대상 엔진의 지원 범위를 먼저 확인한다.
- `dynamo`: PyTorch 2.x부터 `torch.export` 기반의 새 exporter가 추가되었다. 기존 exporter가 실패하는 동적 제어 흐름을 더 잘 다루지만, 성숙도는 모델에 따라 다르다. 한쪽에서 실패하면 다른 쪽을 시도한다.

**검증**
```Python
import onnx

onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)

from collections import Counter
print(Counter(node.op_type for node in onnx_model.graph.node).most_common(6))
print([(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
       for i in onnx_model.graph.input])
```

**출력 결과**
```
[('Conv', 20), ('Relu', 17), ('Identity', 16), ('Add', 8), ('MaxPool', 1), ('GlobalAveragePool', 1)]
[('input', ['batch', 3, 224, 224])]
```

PyTorch에서 BatchNorm이 20개였던 것이 ONNX 그래프에는 보이지 않는다. exporter가 추론 모드의 BatchNorm을 앞선 Conv의 가중치·편향에 접어 넣었기(folding) 때문이다. 이렇게 변환 단계에서 이미 첫 번째 최적화가 일어난다. 생성된 `resnet18.onnx`를 [Netron](https://netron.app/)에 올리면 이 그래프를 시각적으로 확인할 수 있다.

**자주 만나는 변환 실패**

| 증상 | 원인 | 대응 |
| --- | --- | --- |
| `Unsupported operator` | 사용한 연산에 대응하는 ONNX 연산자가 없음 | opset 상향, 연산을 지원되는 조합으로 재작성, 커스텀 연산자 등록 |
| 형태가 고정되어 배치 변경 시 오류 | `dynamic_axes` 누락 | 가변 축 명시 |
| Python 제어 흐름(if/for)이 사라짐 | 트레이싱은 더미 입력 기준 경로만 기록 | `torch.jit.script` 또는 `dynamo=True`로 내보내기, 제어 흐름을 텐서 연산으로 변경 |
| 결과 불일치 | 학습 모드로 내보냄, 전처리 불일치, 난수 연산 | `eval()` 확인, 전처리를 그래프 안에 포함하거나 동일하게 적용 |

<br>

## 3. ONNX Runtime으로 실행하고 검증하기

변환 후에는 반드시 원본과의 **수치 일치**와 **속도**를 함께 확인한다.

**예제 코드**
```Python
import time
import numpy as np
import torch
import onnxruntime as ort
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
dummy_input = torch.randn(1, 3, 224, 224)

session = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
x = dummy_input.numpy()

with torch.no_grad():
    ref = model(dummy_input).numpy()
out = session.run(["logits"], {"input": x})[0]
print("max abs diff:", np.abs(ref - out).max())

def bench(fn, n=20):
    fn()  # 워밍업
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000

with torch.no_grad():
    print(f"PyTorch      : {bench(lambda: model(dummy_input)):.1f} ms")
print(f"ONNX Runtime : {bench(lambda: session.run(['logits'], {'input': x})):.1f} ms")
```

**출력 결과** (CPU, 배치 1)
```
max abs diff: 2.6524067e-06
PyTorch      : 41.7 ms
ONNX Runtime : 15.9 ms
```

부동소수점 연산 순서가 달라지므로 `1e-5` 안팎의 차이는 정상이다. 그 이상 벌어지면 2절의 실패 표를 다시 점검한다. 같은 CPU에서 ONNX Runtime이 2~3배 빠른 것은 그래프 수준 최적화(연산자 융합, 상수 접기)와 스레드 풀·메모리 계획이 추론 전용으로 설계되었기 때문이다.

**실행 공급자(Execution Provider)**

`providers` 인자로 실행 하드웨어를 고른다. 앞에 둔 것이 우선하며, 지원하지 않는 연산자는 뒤의 공급자로 폴백된다.

| 공급자 | 대상 | 패키지 |
| --- | --- | --- |
| `CPUExecutionProvider` | 모든 CPU | `onnxruntime` |
| `CUDAExecutionProvider` | NVIDIA GPU | `onnxruntime-gpu` |
| `TensorrtExecutionProvider` | NVIDIA GPU, TensorRT로 부분 그래프 가속 | `onnxruntime-gpu` + TensorRT |
| `OpenVINOExecutionProvider` | Intel CPU/iGPU/NPU | `onnxruntime-openvino` |
| `CoreMLExecutionProvider` | Apple 기기 | `onnxruntime` (macOS) |

**세션 옵션**

```Python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4          # 연산 내부 병렬 스레드 수(코어 수에 맞춤)
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession("resnet18.onnx", opts, providers=["CPUExecutionProvider"])
```

서빙 컨테이너에서 CPU 코어 제한(예: 2 vCPU)을 두었다면 `intra_op_num_threads`를 그에 맞춰야 한다. 기본값은 물리 코어 수를 따르므로, 컨테이너 한도보다 많은 스레드가 생성되어 오히려 느려지는 경우가 흔하다.

<br>

## 4. 양자화

양자화는 FP32 가중치·활성화를 INT8 등 저정밀도로 바꿔 모델 크기와 메모리 대역폭을 줄이고, 하드웨어가 지원하면 연산도 가속한다. 책 12.2.4에서 소개한 개념을 ONNX Runtime의 두 방식으로 실습한다.

### 4.1 동적 양자화(dynamic quantization)

가중치는 미리 INT8로 저장하고, 활성화는 실행 시점에 범위를 계산해 양자화한다. 보정 데이터가 필요 없어 가장 간단하다.

```Python
from onnxruntime.quantization import quantize_dynamic, quant_pre_process, QuantType

quant_pre_process("resnet18.onnx", "resnet18_pre.onnx")   # 형태 추론·최적화 선처리
quantize_dynamic("resnet18_pre.onnx", "resnet18_int8.onnx", weight_type=QuantType.QInt8)
```

```Python
import os
for f in ["resnet18.onnx", "resnet18_int8.onnx"]:
    print(f, f"{os.path.getsize(f) / 1e6:.1f} MB")

s32 = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession("resnet18_int8.onnx", providers=["CPUExecutionProvider"])
a = s32.run(None, {"input": x})[0]
b = s8.run(None, {"input": x})[0]
print("top-1 일치:", a.argmax() == b.argmax(), "| max abs diff:", np.abs(a - b).max())
print(f"FP32 {bench(lambda: s32.run(None, {'input': x})):.1f} ms | INT8 {bench(lambda: s8.run(None, {'input': x})):.1f} ms")
```

**출력 결과** (AVX-512 VNNI 미지원 CPU)
```
resnet18.onnx 46.7 MB
resnet18_int8.onnx 11.7 MB
top-1 일치: True | max abs diff: 0.14093423
FP32 15.8 ms | INT8 95.7 ms
```

모델 크기는 1/4로 줄었지만 **속도는 6배 느려졌다**. 이 결과가 이 절에서 가장 중요한 교훈이다. 양자화는 하드웨어에 INT8 전용 명령(x86의 AVX-512 VNNI, ARM의 dot-product 명령, NVIDIA의 INT8 Tensor Core)이 있을 때만 빨라진다. 지원이 없으면 실행 시점의 양자화·역양자화 오버헤드만 추가된다. 또한 동적 양자화는 활성화 범위를 매번 계산하므로 Conv 위주의 비전 모델보다 MatMul 위주의 트랜스포머·RNN에 적합하다. **양자화 여부는 배포 대상 하드웨어에서 실측한 뒤 결정**해야 하며, 크기 절감만 목적이라면(모바일 배포, 다운로드 용량) 속도 손실을 감수할 수도 있다.

### 4.2 정적 양자화(static quantization)

대표 입력(보정 데이터, 보통 100~1,000개)을 미리 흘려 활성화 범위를 계산해 두므로 실행 시 오버헤드가 없고, VNNI·Tensor Core에서 Conv도 가속된다. 비전 모델의 INT8 배포는 대부분 이 방식을 쓴다.

```Python
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quantize_static,
)

class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, samples):          # samples: (N, 3, 224, 224) float32, 학습 전처리 적용 완료
        self._iter = iter(samples)
    def get_next(self):
        batch = next(self._iter, None)
        return None if batch is None else {"input": batch[None]}

calib = np.random.randn(200, 3, 224, 224).astype(np.float32)  # 실제로는 검증 세트 샘플 사용
quantize_static(
    "resnet18_pre.onnx", "resnet18_int8_static.onnx",
    calibration_data_reader=ImageCalibrationReader(calib),
    quant_format=QuantFormat.QDQ,          # Quantize-Dequantize 노드 삽입, TensorRT 호환
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,                      # 채널별 스케일, Conv 정확도 향상
)
```

보정 데이터는 실제 서비스 입력 분포를 대표해야 한다. 난수를 쓰면 활성화 범위가 잘못 잡혀 정확도가 크게 떨어진다. 양자화 후에는 반드시 검증 세트로 정확도(top-1, F1 등)를 재측정하고, 특정 계층에서 오차가 크면 그 계층만 FP32로 남기는(`nodes_to_exclude`) 혼합 정밀도를 적용한다.

<br>

## 5. TensorRT로 GPU 최적화

TensorRT는 NVIDIA GPU 전용 추론 최적화 라이브러리로, ONNX 그래프를 받아 대상 GPU에 맞춘 실행 엔진을 빌드한다. 연산자 융합, FP16/INT8 정밀도, 커널 자동 튜닝, 동적 배치를 지원하며, 같은 GPU에서 PyTorch 대비 수 배의 처리량을 얻는 경우가 많다.

**빌드 (CLI)**
```
trtexec --onnx=resnet18.onnx \
        --saveEngine=resnet18_fp16.engine \
        --fp16 \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:16x3x224x224 \
        --maxShapes=input:64x3x224x224
```

- `--fp16`: 반정밀도 사용. 대부분의 비전·NLP 모델에서 정확도 손실이 거의 없고 처리량이 2배 안팎 늘어난다. 첫 번째로 시도할 옵션이다.
- `--int8`: INT8 사용. 보정 캐시(`--calib`)가 필요하며, 4.2절의 QDQ 형식 ONNX를 입력하면 그 양자화 파라미터를 그대로 사용한다.
- `--minShapes/--optShapes/--maxShapes`: 동적 배치 범위. 엔진은 `optShapes` 기준으로 커널을 튜닝하므로 실제 서빙 배치 크기 분포의 중앙값을 넣는다.
- 빌드는 GPU 모델·드라이버·TensorRT 버전에 종속된다. **빌드한 환경과 다른 GPU에서는 엔진이 동작하지 않으므로** 서빙 이미지 빌드 단계에서 대상 GPU로 빌드하거나, 서빙 시작 시 빌드하고 캐시한다.

**실행 (Python)**
```Python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

logger = trt.Logger(trt.Logger.WARNING)
with open("resnet18_fp16.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

batch = np.random.randn(16, 3, 224, 224).astype(np.float32)
context.set_input_shape("input", batch.shape)

d_input = cuda.mem_alloc(batch.nbytes)
output = np.empty((16, 1000), dtype=np.float32)
d_output = cuda.mem_alloc(output.nbytes)
context.set_tensor_address("input", int(d_input))
context.set_tensor_address("logits", int(d_output))

stream = cuda.Stream()
cuda.memcpy_htod_async(d_input, batch, stream)
context.execute_async_v3(stream_handle=stream.handle)
cuda.memcpy_dtoh_async(output, d_output, stream)
stream.synchronize()
print(output.shape, output.argmax(axis=1)[:5])
```

실무에서는 이 저수준 코드를 직접 쓰기보다 **Triton Inference Server의 TensorRT 백엔드**에 엔진 파일을 올리고 동적 배치·다중 인스턴스 설정을 구성 파일로 관리하는 것이 일반적이다. PyTorch 코드를 크게 바꾸지 않고 TensorRT를 쓰고 싶다면 `torch_tensorrt.compile()`로 모듈 단위 변환도 가능하다.

**정밀도 선택 기준**

| 정밀도 | 처리량(상대) | 정확도 영향 | 요구 사항 |
| --- | --- | --- | --- |
| FP32 | 1× | 없음 | — |
| FP16 | 약 2× | 대부분 무시 가능, 수치 범위 민감 모델은 확인 | Pascal 이상 GPU |
| INT8 | 약 3~4× | 보정 품질에 따라 다름, 검증 필수 | Turing 이상, 보정 데이터 |
| FP8 | INT8 수준 | INT8보다 정확도 유리 | Hopper 이상 |

<br>

## 6. 변환 후 반드시 확인할 것

변환·경량화는 성능을 얻는 대신 **원본과 다른 모델**을 만든다. 배포 전 체크리스트는 다음과 같다.

1. **수치 일치**: 동일 입력 수백 개에 대해 원본과 출력 차이(최대 절대 오차, 분류라면 top-1 일치율)를 측정한다. 변환 직후(FP32)에는 `1e-4` 이내, FP16은 `1e-2` 이내가 일반적이다.
2. **과제 지표**: 검증 세트로 정확도·F1·NDCG 등 실제 과제 지표를 재측정한다. 수치 오차가 작아도 경계 근처 샘플의 예측이 뒤집힐 수 있다.
3. **엣지 케이스**: 최소·최대 배치, 극단적 입력값(0, 매우 큰 값), 실제 서비스 로그에서 뽑은 어려운 샘플로 재확인한다.
4. **속도 실측**: 배포 대상과 동일한 하드웨어·컨테이너 자원 한도에서 p50/p95 지연과 처리량을 측정한다. 개발 머신의 수치는 참고만 한다. 4.1절의 사례처럼 예상과 반대 결과가 나올 수 있다.
5. **메모리**: GPU 메모리 사용량과 CPU RSS를 확인한다. 엔진 빌드 시 작업 공간(workspace) 설정에 따라 크게 달라진다.
6. **재현 가능한 빌드**: 변환 스크립트, 사용한 라이브러리 버전, 보정 데이터, 검증 결과를 모델 레지스트리(MLflow 등)에 아티팩트로 함께 저장한다. 책 7.2.3의 버전 관리·롤백 전략이 변환된 모델에도 적용되어야 한다.
7. **전처리 일치**: 변환된 모델 앞의 전처리(리사이즈, 정규화, 토크나이저)가 학습 시와 동일한지 확인한다. 전처리를 ONNX 그래프 안에 포함시키면 이 불일치를 구조적으로 막을 수 있다.

<br>
<br>

모델 변환은 한 번의 명령이 아니라 "내보내기 → 검증 → 최적화 → 재검증"의 반복이다. 그 과정에서 얻는 것은 속도만이 아니다. 어떤 연산자가 몇 개인지, BatchNorm이 어디로 사라졌는지, 어느 계층이 양자화에 민감한지를 확인하는 일은 [모델 구조 이해하기](./understanding-model-architecture.md)의 연장이며, 변환된 모델을 어떤 하드웨어에 어떤 배치 크기로 올릴지 결정하는 일은 [모델 서빙 시 트레이드오프 고려사항](./model-serving-tradeoffs.md)의 출발점이다.

**참고 자료**

- PyTorch, `torch.onnx` 문서: https://pytorch.org/docs/stable/onnx.html
- ONNX Runtime, Quantize ONNX models: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- NVIDIA TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- Torch-TensorRT: https://pytorch.org/TensorRT/
- Netron: https://netron.app/
