import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# 1. 환경 설정
os.environ['FLAGS_use_mkldnn'] = '0'
FONT_PATH = r'C:\Windows\Fonts\malgun.ttf' # 한글 폰트 경로 (Malgun Gothic)

def get_ocr_result(img_path):
    """이미지 전처리 및 OCR 추론 실행"""
    # 전처리: 읽기 -> 2배 확대 -> 그레이스케일 -> 샤프닝
    img = cv2.imread(img_path)
    if img is None: return None
    
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    processed_img = cv2.filter2D(gray, -1, kernel)

    # 모델 로드 및 추론
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    result = ocr.ocr(processed_img, det=True, cls=True)
    
    return processed_img, result

def main():
    img_path = r'C:\Users\USER\vscode-workspace\PokeCard_AI\PaddleOCR\dataset\test\lucario.png'
    
    if not os.path.exists(img_path):
        print(f"❌ 파일을 찾을 수 없습니다: {img_path}")
        return

    print("--- 🔍이미지 분석 시작 ---")
    processed_img, result = get_ocr_result(img_path)

    if not result or not result[0]:
        print("텍스트를 인식하지 못했습니다.")
        return

    # 1. 터미널 결과 출력 및 시각화용 데이터 정리
    print(f"\n{'='*50}\n✅ 인식 성공! (검출된 라인: {len(result[0])})\n{'='*50}")
    
    boxes, txts, scores = [], [], []
    for line in result[0]:
        box, (text, score) = line[0], line[1]
        boxes.append(box)
        txts.append(text)
        scores.append(score)
        print(f"📝 텍스트: {text:15} | 신뢰도: {score:.4f}")

    # 2. 시각화 이미지 생성
    # draw_ocr은 RGB 이미지를 받으므로 변환 필요
    color_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    vis_img = draw_ocr(color_img, boxes, txts, scores, font_path=FONT_PATH)
    
    # 결과 저장 및 표시
    final_image = Image.fromarray(vis_img)
    final_image.save('ocr_result.jpg')
    print(f"\n🖼️ 시각화 결과가 'ocr_result.jpg'로 저장되었습니다.")
    final_image.show()

if __name__ == "__main__":
    main()