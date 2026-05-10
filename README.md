# BioHealth_Frontier_Project_2026
이 프로젝트는 BraTS2021 데이터를 활용하여 MRI 영상에서 뇌종양 부위를 자동으로 분할(Segmentation)하는 AI 모델을 구현한 프로젝트입니다.


# BioHealth_Frontier_Project
### 🧠 3D U-Net을 활용한 BraTS2021 뇌종양 분할 AI 모델

이 프로젝트는 BraTS2021 데이터를 활용하여 MRI 영상에서 뇌종양 부위를 자동으로 분할(Segmentation)하는 AI 모델을 구현한 프로젝트입니다.

#### 🛠️ 개발 환경 및 주요 설정
- **Framework**: PyTorch
- **Device**: Apple Silicon (M2/M3) GPU 가속 (MPS)
- **Special Config**: `PYTORCH_ENABLE_MPS_FALLBACK=1` 설정 (3D Pooling 연산 지원)
- **Data**: BraTS2021 Training Data (ASNR)

#### 🚀 해결한 이슈
- **MPS 3D Operator 이슈**: 맥북 GPU에서 지원하지 않는 3D 연산을 CPU Fallback으로 해결
- **Runtime Size Mismatch**: Up-sampling 과정에서의 1px 오차를 `F.interpolate`로 보정
- **Memory Optimization**: MPS High Watermark 설정을 통한 VRAM 관리
