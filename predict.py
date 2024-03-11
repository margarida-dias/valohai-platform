from ultralytics import YOLO
import os

if __name__ == "__main__":

    path_to_model = "./valohai/outputs/best.onnx"
    path_to_images = "./valohai/inputs/images/"

    model = YOLO(path_to_model, task="detect")

    for image in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image)

        if os.path.isfile(image_path):
            # Run prediction on image
            # Save result as an image in /valohai/outputs/
            results = model.predict(image_path, save=True, project="./valohai/outputs", name="predictions")

            # Print the Boxes object containing the detection bounding boxes
            for r in results:
                print(r.boxes)