import paddle
import paddle2onnx

# 1. GPU 인식 여부 확인
print(f"GPU 사용 가능 여부: {paddle.is_compiled_with_cuda()}")
print(f"현재 인식된 GPU: {paddle.device.get_device()}")

# 2. 라이브러리 버전 확인
print(f"Paddle 버전: {paddle.__version__}")
print(f"Paddle2ONNX 버전: {paddle2onnx.__version__}")

# 3. 간단한 연산 테스트 (DLL 충돌 체크)
x = paddle.ones([2, 2])
y = x + x
print("GPU 연산 테스트 완료!")