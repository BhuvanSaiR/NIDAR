from ultralytics import YOLO

# If you already have the pretrained weights (yolo11n.pt) or your own .pt:
model = YOLO("yolo11n.pt")   # or path/to/your/yolo11n.pt
model.export(format="ncnn")   # creates ./yolo11n_ncnn_model directory
