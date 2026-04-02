import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

#!!!!!주의사항!!!!!#
#이거 좀 오래걸리니까 노트북으로는 돌리지말것 !!!!!!!!

#0.기본옵션 설정
folder_name = "emolga"  # 원하는 포켓몬 폴더명입력
num_per_original = 50   # 원본 1장당 생성할 개수

#1.이 코드파일 기준으로 경로생성
current_dir = os.path.dirname(os.path.abspath(__file__))
origin_path = os.path.join(current_dir, "..", "dataset", "origin", folder_name) #원본파일 폴더위치
train_path = os.path.join(current_dir, "..", "dataset", "train", folder_name)   #저장할파일 폴더위치
bg_path = os.path.join(current_dir, "..", "dataset", "background")              #배경화면파일 폴더위치
save_path_dir = train_path 

#2.배경이미지, 원본이미지 자동 리스트저장(파일명상관X)
def get_image_files(path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    return [f for f in os.listdir(path) if f.lower().endswith(valid_extensions)]

original_files = get_image_files(origin_path)
bg_files = get_image_files(bg_path)

if not original_files:
    print(f"❌ {train_path}에 이미지가 없습니다.")
    exit()
if not bg_files:
    print(f"⚠️ 배경 이미지가 없습니다. 배경 합성 없이 진행하거나 폴더를 확인하세요.")

#3.증강 파이프라인 (카드 자체 변형)
card_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=15, expand=True), # expand=True로 잘림 방지
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])

final_resize = transforms.Resize((224, 224)) # 모델 입력 최종 크기

#4.합성된파일 저장
os.makedirs(save_path_dir, exist_ok=True)
count = 0

print(f"🚀 {folder_name} 증강 시작 (원본: {len(original_files)}장)")

for i, file_name in enumerate(original_files):
    img_path = os.path.join(origin_path, file_name)
    img = Image.open(img_path).convert('RGBA') # 투명도 처리를 위해 RGBA 권장

    for j in range(num_per_original):
        # A. 카드 자체 변형
        aug_card = card_augmentation(img)
        
        # B. 배경 선택 및 로드
        if bg_files:
            bg_file = random.choice(bg_files)
            background = Image.open(os.path.join(bg_path, bg_file)).convert('RGBA')
            
            # 배경 크기에 맞춰 카드 크기 조절 (배경의 40~70% 사이)
            bg_w, bg_h = background.size
            scale = random.uniform(0.4, 0.7)
            new_w = int(bg_w * scale)
            new_h = int(aug_card.height * (new_w / aug_card.width))
            aug_card = aug_card.resize((new_w, new_h), Image.LANCZOS)
            
            # 랜덤 위치 선정
            max_x = max(0, bg_w - new_w)
            max_y = max(0, bg_h - new_h)
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)
            
            # 배경 위에 카드 합성 (mask를 사용해 투명도 유지)
            background.paste(aug_card, (paste_x, paste_y), aug_card)
            final_img = background.convert('RGB')
        else:
            final_img = aug_card.convert('RGB')

        # C. 최종 리사이즈 및 저장
        final_img = final_resize(final_img)
        save_name = f'{folder_name}_aug_{i}_{j:03d}.jpg'
        final_img.save(os.path.join(save_path_dir, save_name), 'JPEG', quality=90)
        count += 1
    print(f"현재까지 {count}장 완료....")

print(f"✅ 완료! 총 {count}장의 이미지가 {save_path_dir}에 생성되었습니다.")