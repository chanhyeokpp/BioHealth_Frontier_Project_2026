# BraTS2021 Brain Tumor Segmentation & Classification

BraTS2021 MRI 데이터를 활용해 뇌종양을 분할(Segmentation)하고,
종양 유형을 분류(Classification)하는 딥러닝 파이프라인입니다.

## 프로젝트 개요
- **Segmentation**: U-Net(2D)으로 종양 영역 분할
- **Classification**: ResNet(18/34/50)으로 종양 유형 다중 분류
- **XAI**: Grad-CAM으로 모델 판단 근거 시각화

## 결과
**Segmentation**: Mean Dice 0.855 / Mean IoU 0.762
**Classification**: Accuracy 93.94% / Macro ROC-AUC 0.984

## 사용 기술
Python, PyTorch, nibabel, matplotlib, U-Net(2D), ResNet, Grad-CAM

## 데이터 처리
- nibabel로 NIfTI(.nii.gz) 포맷의 BraTS2021 처리
- 용량·연산 제약을 고려해 3D 볼륨에서 종양 포함 2D 슬라이스 추출
- 밝기 정규화, Train/Val/Test 분할

## 개발 환경 및 해결한 이슈
- **Framework**: PyTorch / Apple Silicon MPS 가속
- **MPS 3D 연산 미지원**: CPU Fallback(`PYTORCH_ENABLE_MPS_FALLBACK=1`)으로 우회
- **Up-sampling 1px 오차**: `F.interpolate`로 보정
- **VRAM 관리**: MPS High Watermark 설정

## 실행
\`\`\`bash
python train.py       # 학습
python evaluate.py    # 평가 (Dice/IoU)
python inference.py   # 추론 및 시각화
python gradcam.py     # Grad-CAM
\`\`\`

## 파일 구조
- `data_loader.py` : 데이터셋·전처리
- `model.py` : U-Net / ResNet 모델 정의
- `train.py` / `evaluate.py` / `inference.py` / `gradcam.py`
