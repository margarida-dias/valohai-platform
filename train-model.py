import shutil
from ultralytics import YOLO
import argparse
import json


# Valohai: Use argparse to parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='SGD')
    return parser.parse_args()


args = parse_args()

# Define a method that prints out key metrics from the trainer
def print_valohai_metrics(trainer):
    metadata = {
        "epoch": trainer.epoch,
    }
    # Loop through the metrics
    for metric in trainer.metrics:
        metric_name = metric.split("metrics/")[-1]
        metric_value = trainer.metrics[metric]

        metadata[metric_name] = metric_value

    print(json.dumps(metadata))


# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# Valohai: Add a callback function, every time an epoch ends, print the metrics
model.add_callback("on_train_epoch_end", print_valohai_metrics)

# Use the model
# Valohai: Use the epoch value from argparse
model.train(data="coco128.yaml", epochs=args.epochs,
optimizer=args.optimizer, verbose=False)  # train the model

path = model.export(format="onnx")  # export the model to ONNX format

# Valohai: Copy the exported model to the Valohai outputs directory
shutil.copy(path, '/valohai/outputs/')

# Save the model with the alias production-model-a
file_metadata = {
    "valohai.alias": "production-model-a",
    "valohai.tags": ["project-a", "aerospace"]
}

with open("/valohai/outputs/best.onnx.metadata.json", "w") as outfile:
    json.dump(file_metadata, outfile)
