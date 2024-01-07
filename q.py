from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("./yolov8m-seg.pt")  # load a pretrained model (recommended for training)
results = model("datasets/data/images/train/LZW_0378.JPG",save=True)
print(results[0].boxes.data.cpu().numpy().tolist())
# Use the model
# model.train(data="/home/u2019110058/detect_looks/data.yaml", epochs=1000, workers=0)  # train the model
# model.train(data="./data.yaml", epochs=1000, workers=0)  # train the model
# success = model.export(format="onnx")  # export the model to ONNX format
