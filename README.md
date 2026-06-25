# Brain Tumor Classification (ResNet18)

뇌종양 MRI를 4가지 유형으로 분류하는 딥러닝 프로젝트입니다.
ResNet18 기반 모델을 학습·평가하고, Grad-CAM으로 판단 근거를 시각화했습니다.

- **분류 클래스**: glioma / meningioma / pituitary / no tumor
- **모델**: ResNet18 (ImageNet pretrained)
- **해석(XAI)**: Grad-CAM

---

## 결과

| 지표 | 값 |
|---|---|
| Accuracy | 93.94% |
| Macro ROC-AUC | 0.984 |

ResNet18/34/50과 하이퍼파라미터(learning rate, optimizer, augmentation,
dropout, epoch)를 통제변인 방식으로 비교했으며,
epoch를 늘리기보다 validation 기준 best model을 저장(Early Stopping)하는 것이
더 중요하다는 점을 확인했습니다.

---

## 데이터셋

Kaggle Brain Tumor MRI Dataset (약 7,200장)
glioma / meningioma / pituitary / no tumor 4개 클래스

## 사용 기술

Python, PyTorch, torchvision, nibabel, matplotlib, ResNet18, Grad-CAM

---

## 개발 환경 / 해결한 이슈

- **Framework**: PyTorch / Apple Silicon MPS 가속
- **MPS 연산 미지원 이슈**: CPU Fallback(`PYTORCH_ENABLE_MPS_FALLBACK=1`)으로 우회
- **메모리 관리**: MPS High Watermark 설정으로 VRAM 관리

---

## 실행

\`\`\`bash
python check_data.py    # 데이터 무결성·shape 검사
python train.py         # 학습
python evaluate.py      # 평가 (Accuracy / ROC-AUC / Confusion Matrix)
python inference.py     # 단일 영상 추론·시각화
python gradcam.py       # Grad-CAM 시각화
\`\`\`

---

## 파일 구조

\`\`\`
config.py        # 설정 (클래스 수, pretrained 여부 등)
check_data.py    # 데이터 확인
data_loader.py   # 데이터셋·전처리
model.py         # ResNet18 모델 정의
train.py         # 학습
evaluate.py      # 평가
inference.py     # 추론
gradcam.py       # Grad-CAM
\`\`\`
