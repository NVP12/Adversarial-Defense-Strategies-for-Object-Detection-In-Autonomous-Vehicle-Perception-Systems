import os
import cv2
from ultralytics import YOLO
import math

def run_yolo_inference_from_list(model_path, image_list_path, output_txt_path, output_txt_path2):
    """
    Runs YOLOv8 inference on a list of images (from a .txt file).
    Saves 95% of results in output_txt_path and 5% in output_txt_path2.
    Each line: <image_path> x1,y1,x2,y2,class_id ...
    """
    model = YOLO(model_path)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path2), exist_ok=True)

    # Read image paths from the input .txt file
    with open(image_list_path, 'r') as f:
        image_files = [line.strip() for line in f if line.strip()]

    # Shuffle and split 5%
    total = len(image_files)
    split_index = math.ceil(total * 0.05)
    output_txt2_images = set(image_files[:split_index])
    
    with open(output_txt_path, "w") as out1, open(output_txt_path2, "w") as out2:
        for img_path in image_files:
            original_img = cv2.imread(img_path)
            if original_img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            orig_h, orig_w = original_img.shape[:2]
            resized_img = cv2.resize(original_img, (640, 640)) if (orig_w, orig_h) != (640, 640) else original_img

            results = model(resized_img)
            for r in results:
                line = [img_path]

                for box in r.boxes:
                    class_id = int(box.cls[0].item())
                    x_c, y_c, w, h = box.xywh[0].tolist()

                    x_c *= (orig_w / 640)
                    y_c *= (orig_h / 640)
                    w   *= (orig_w / 640)
                    h   *= (orig_h / 640)

                    x_min = int(x_c - w / 2)
                    y_min = int(y_c - h / 2)
                    x_max = int(x_c + w / 2)
                    y_max = int(y_c + h / 2)

                    line.append(f"{x_min},{y_min},{x_max},{y_max},{class_id}")

                output_line = ' '.join(line) + '\n'
                if img_path in output_txt2_images:
                    out2.write(output_line)
                else:
                    out1.write(output_line)

    print(f"95% saved to: {output_txt_path}")
    print(f"5% saved to: {output_txt_path2}")

