from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # builds a new model from scratch
model = YOLO('runs/detect/train5/weights/best.pt') # selects old models to train

# Train the model
#model.train(data="data.yaml", epochs=20)  # train the model


model.train(data="data.yaml", epochs=20, lr0=0.001, lrf=0.01) # Fine tuning av modellen med en mer smidig learning rate

metrics = model.val()  # evaluate model performance on the validation set