import os, cv2, json

# 1. 경로 설정 (사용자 환경에 맞게 수정)
IMAGE_DIR = r'C:\Users\USER\vscode-workspace\PokeCard_AI\PaddleOCR_pip\dataset\img_train'
LABEL_DIR = r'C:\Users\USER\vscode-workspace\PokeCard_AI\PaddleOCR_pip\dataset\labels_train'
SAVE_PATH = r'C:\Users\USER\vscode-workspace\PokeCard_AI\PaddleOCR_pip\dataset\labels_train\train_label.txt'

def convert_yolo_to_paddle():
    output_lines = []
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt') and f != 'classes.txt']
    
    for file_name in label_files:
        img_name = file_name.replace('.txt', '.png')
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        boxes = []
        with open(os.path.join(LABEL_DIR, file_name), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # YOLO: class, x_center, y_center, width, height (0~1 scale)
                _, xc, yc, bw, bh = map(float, parts)
                
                # 픽셀 좌표로 복원
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                
                # PaddleOCR 4점 좌표 형식 [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                boxes.append({
                    "transcription": "000",
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                })
        
        if boxes:
            # 이미지파일명 + 탭 + JSON (한 줄로 결합)
            line_data = f"{img_name}\t{json.dumps(boxes, ensure_ascii=False)}"
            output_lines.append(line_data)

    # 최종 파일 저장 (중간 줄바꿈 없이 깔끔하게 저장)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(output_lines))

if __name__ == "__main__":
    convert_yolo_to_paddle()
    print(f"변환 완료: {SAVE_PATH}")