from sam_segment import extract_leaf
from fastapi import FastAPI, File, UploadFile
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

app = FastAPI()

# =========================
# PARAMETERS
# =========================

MODEL_PATH = "models/leaf_disease_rgb_only_resnet18.pth"
CLASSES = ["Yellow rust", "Healthy", "Brown rust"]
IMAGE_SIZE = (320,320)

DEVICE = "cpu"

# =========================
# LOAD MODEL
# =========================

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ CNN Model Loaded")

# =========================
# API ENDPOINT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    save_path = "received_image.jpg"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(save_path)
    img = extract_leaf(img)
    cv2.imwrite("segmented_leaf.jpg", img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMAGE_SIZE)

    img_rgb = img_rgb.astype(np.float32) / 255.0
    input_tensor = torch.tensor(img_rgb).permute(2,0,1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    pred_class = CLASSES[pred_idx.item()]
    confidence = float(conf.item()*100)

    return {
        "disease": pred_class,
        "confidence": confidence
    }