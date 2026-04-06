import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 0. 기본 옵션 설정
# folder_name 변수 대신 origin 폴더 내의 모든 폴더를 대상으로 합니다.
num_per_original = 50   # 원본 1장당 생성할 개수

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
# 원본 데이터들이 모여있는 최상위 폴더
origin_base_path = os.path.abspath(os.path.join(current_dir, "dataset", "origin"))
# 저장될 학습 데이터 최상위 폴더
train_base_path = os.path.abspath(os.path.join(current_dir, "dataset", "train"))
# 배경화면 폴더
bg_path = os.path.abspath(os.path.join(current_dir, "dataset", "background"))

# 2. 배경이미지 리스트 로드 (모든 포켓몬 공통 사용)
def get_image_files(path):
    if not os.path.exists(path): return []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    return [f for f in os.listdir(path) if f.lower().endswith(valid_extensions)]

bg_files = get_image_files(bg_path)

# 3. 증강 파이프라인 설정
card_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=15, expand=True), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])
final_resize = transforms.Resize((224, 224))

# --- 4. 메인 루프: 모든 포켓몬 폴더 순회 ---
# origin_base_path 안에 있는 모든 폴더명을 가져옵니다.
pokemon_folders = [f for f in os.listdir(origin_base_path) 
                   if os.path.isdir(os.path.join(origin_base_path, f))]

print(f"📢 총 {len(pokemon_folders)}종의 포켓몬을 발견했습니다: {pokemon_folders}")

for folder_name in pokemon_folders:
    # 각 포켓몬별 경로 재설정
    origin_path = os.path.join(origin_base_path, folder_name)
    train_path = os.path.join(train_base_path, folder_name)
    
    original_files = get_image_files(origin_path)
    if not original_files:
        print(f"⏩ {folder_name}: 원본 이미지가 없어 건너뜁니다.")
        continue

    os.makedirs(train_path, exist_ok=True)
    print(f"🚀 {folder_name} 증강 시작 (원본: {len(original_files)}장)")

    # tqdm을 사용하여 진행 상황 시각화
    for i, file_name in enumerate(tqdm(original_files, desc=f"Processing {folder_name}")):
        img_path = os.path.join(origin_path, file_name)
        try:
            img = Image.open(img_path).convert('RGBA')
        except Exception as e:
            print(f"❌ 파일 로드 실패 ({file_name}): {e}")
            continue

        for j in range(num_per_original):
            # A. 카드 자체 변형
            aug_card = card_augmentation(img)
            
            # B. 배경 합성
            if bg_files:
                bg_file = random.choice(bg_files)
                background = Image.open(os.path.join(bg_path, bg_file)).convert('RGBA')
                
                bg_w, bg_h = background.size
                scale = random.uniform(0.4, 0.7)
                new_w = int(bg_w * scale)
                # 가로 세로 비율 유지하며 리사이즈
                aspect_ratio = aug_card.height / aug_card.width
                new_h = int(new_w * aspect_ratio)
                
                aug_card_resized = aug_card.resize((new_w, new_h), Image.LANCZOS)
                
                max_x = max(0, bg_w - new_w)
                max_y = max(0, bg_h - new_h)
                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)
                
                background.paste(aug_card_resized, (paste_x, paste_y), aug_card_resized)
                final_img = background.convert('RGB')
            else:
                final_img = aug_card.convert('RGB')

            # C. 최종 리사이즈 및 저장
            final_img = final_resize(final_img)
            save_name = f'{folder_name}_aug_{i}_{j:03d}.jpg'
            final_img.save(os.path.join(train_path, save_name), 'JPEG', quality=90)

print(f"\n✨ 모든 작업이 완료되었습니다! 데이터셋은 {train_base_path}에 저장되었습니다.")