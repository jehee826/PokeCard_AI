import onnxruntime as ort
import numpy as np
import cv2

# 1. 환경 설정
model_path = r"C:\Users\USER\vscode-workspace\PokeCard_AI\PaddleOCR_pip\model.onnx"
image_path = r"C:\Users\USER\vscode-workspace\PokeCard_AI\PaddleOCR_pip\dataset\test\sprigatito.png"
sess = ort.InferenceSession(model_path)

# 2. 이미지 로드 및 전처리
img = cv2.imread(image_path)
if img is None:
    print("이미지를 찾을 수 없습니다!")
    exit()

h, w, _ = img.shape
img_input = cv2.resize(img, (640, 640))

# [수정 핵심] 정규화 계산 후 반드시 float32로 변환
img_input = img_input.astype('float32') / 255.0
img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img_input = img_input.transpose(2, 0, 1)
img_input = np.expand_dims(img_input, axis=0).astype('float32') # 여기서 한 번 더 확실하게!

# 3. 추론 실행
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: img_input})
heatmap = output[0][0][0]

# [디버그] 히트맵 저장
debug_heatmap = (heatmap * (255 / (np.max(heatmap) + 1e-6))).astype('uint8')
cv2.imwrite("heatmap_debug.jpg", debug_heatmap)
print(f"히트맵 저장 완료 (Max: {np.max(heatmap):.4f})")

# 4. 박스 그리기 및 저장 (이전과 동일)
_, binary_map = cv2.threshold(heatmap, 0.2, 255, cv2.THRESH_BINARY)
binary_map = binary_map.astype('uint8')
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) < 10: continue
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box * [w / 640, h / 640]
    box = box.astype(np.int32)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

cv2.imwrite("test_result_final.jpg", img)
print(f"최종 결과 저장 완료! 찾은 박스 개수: {len(contours)}")