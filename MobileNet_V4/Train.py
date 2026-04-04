import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
from tqdm import tqdm

def train_model():
    # 1. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 현재 사용 중인 장치: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    train_path = os.path.join(current_dir, "..", "dataset", "train")
    save_path = os.path.join(current_dir, "pokecard_model.pth")

    # 2. 데이터 로더 (학습 효율 극대화)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7, scale=(0.05, 0.4), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=train_path, transform=transform) 
    # pin_memory=True: CPU 메모리에서 GPU로 데이터를 더 빨리 보냄
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, 
                              num_workers=8, pin_memory=True)

    # 3 & 4. MobileNet V4 모델 설정
    num_classes = len(dataset.classes)
    print(f"📊 감지된 클래스: {num_classes}개 ({dataset.classes})")
    
    model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # 5. 손실함수 및 최적화 (과적합 방지 옵션 추가)
    criterion = nn.CrossEntropyLoss()
    # weight_decay: 모델이 너무 복잡하게 외우는 것을 방지 (L2 규제)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # 학습률 스케줄러: 손실이 줄어들수록 섬세하게 학습
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 6. 학습 시작
    epochs = 20 # 과적합 기미가 보이면 20정도가 적당합니다.
    
    print(f"🚀 {epochs} 에폭 학습 시작...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm으로 진행바 표시
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step() # 학습률 조정
        avg_loss = running_loss / len(train_loader)
        print(f"📢 Epoch {epoch+1} 평균 손실: {avg_loss:.4f}")

    # 7. 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"✅ 학습 완료! 모델 저장 위치: {save_path}")

# 윈도우 멀티프로세싱 에러 방지를 위한 메인 루프 가드
if __name__ == '__main__':
    train_model()