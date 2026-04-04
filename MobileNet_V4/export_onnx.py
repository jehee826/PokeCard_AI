import torch
import timm
import os
import onnx  # 가중치 병합을 위해 필요합니다

def export_onnx():
    # 1. 환경 및 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_NAME = "pokecard_model.pth"
    MODEL_NAME = 'mobilenetv4_conv_small'
    ONNX_NAME = "pokecard_model.onnx"

    # 카테고리 개수 자동 로드 (추론 코드와 동일한 정렬 방식)
    pokemonName_path = os.path.abspath(os.path.join(current_dir, "..", "dataset", "origin"))
    MY_CATEGORIES = sorted([f for f in os.listdir(pokemonName_path) 
                           if os.path.isdir(os.path.join(pokemonName_path, f))])
    NUM_CLASSES = len(MY_CATEGORIES)
    print(f"📊 클래스 개수: {NUM_CLASSES}")

    # 2. 모델 구조 생성 및 가중치 로드
    print(f"🔄 모델 구조 생성 중: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    weights_path = os.path.join(current_dir, WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        print(f"❌ 가중치 파일을 찾을 수 없습니다: {weights_path}")
        return

    # CPU로 로드하여 변환 안정성 확보
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print("✅ .pth 가중치 로드 완료")

    # 3. 가상 입력 데이터 생성 (Batch: 1, RGB: 3, Size: 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(current_dir, ONNX_NAME)

    # 4. ONNX 변환 시작
    print(f"🚀 ONNX 변환 시작...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,        # 가중치 포함
            opset_version=15,          # 최신 환경 호환성
            do_constant_folding=True,  # 최적화 적용
            input_names=['input'],     # 리액트에서 사용할 입력 노드명
            output_names=['output'],    # 리액트에서 사용할 출력 노드명
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✅ 기본 변환 성공")
    except Exception as e:
        print(f"❌ 변환 중 에러 발생: {e}")
        return

    # 5. [핵심] 가중치 강제 통합 (Single File Merge)
    # PyTorch 버전에 따라 .data 파일이 따로 생기는 현상을 방지합니다.
    try:
        model_proto = onnx.load(onnx_path)
        # 모든 외부 데이터를 파일 안으로 포함시켜 다시 저장
        onnx.save(model_proto, onnx_path)
        
        # 만약 .onnx.data 파일이 생성되었다면 삭제
        data_file_path = onnx_path + ".data"
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
            print("🗑️ 외부 데이터 파일(.data)을 삭제하고 단일 파일로 합쳤습니다.")
            
        print(f"✨ 최종 결과물 생성 완료: {onnx_path}")
    except Exception as e:
        print(f"⚠️ 병합 과정 중 알림 (이미 단일 파일일 수 있음): {e}")

if __name__ == "__main__":
    export_onnx()