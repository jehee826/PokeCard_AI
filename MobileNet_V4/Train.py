import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# 1. 환경 설정 (5825U는 보통 CPU로 돌리지만, 외장 GPU가 있다면 자동으로 잡습니다)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")

# 2. 데이터 로더 (증강된 100장 불러오기)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 'augmented_pikachu' 폴더 안에 'pikachu' 폴더가 있는 구조여야 합니다
dataset = datasets.ImageFolder(root=r'C:\Users\jehee\vscode-workspace\Ai\dataset\train', transform=transform) 
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. MobileNet V4 모델 불러오기 (timm 라이브러리 사용)
# 'mobilenetv4_conv_small'은 웹용으로 가장 적합한 가벼운 모델입니다
model = timm.create_model('mobilenetv4_conv_small', pretrained=True)

# 4. 분류기 교체 (마지막 층을 우리 클래스 개수에 맞게 변경)
num_classes = len(dataset.classes) # 현재는 피카츄 1종류거나 배경 포함 2종류일 것임
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

# 5. 손실함수 및 최적화 도구
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 학습 시작 (100장이니 10번 정도만 반복해봅시다)
epochs = 10
model.train()

print("🚀 학습을 시작합니다...")
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

# 7. 모델 저장 (나중에 웹에서 쓰기 위해 .pth로 저장)
torch.save(model.state_dict(), "pikachu_model.pth")
print("✅ 학습 완료 및 모델 저장 성공!")