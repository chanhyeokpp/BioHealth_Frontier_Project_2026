from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Brain_Tumor_MRI_Dataset"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "checkpoints"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
RANDOM_SEED = 42

# Keep this False unless your environment can download torchvision weights.
USE_PRETRAINED = True
 
BEST_MODEL_PATH = MODEL_DIR / "best_resnet18_brain_tumor.pth"
