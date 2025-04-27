from ultralytics import YOLO
import cv2
import numpy as np
import logging
import os

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model = YOLO('./yolo11x-pose.pt')
log.info("YOLO model loaded successfully")

def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    results = model(image_path)
    return img, results

def annotate_and_analyze(img, results):
    left_wrist = None
    right_wrist = None
    mouth = None
    left_eye = None
    right_eye = None

    for result in results:
        keypoints = result.keypoints
        if keypoints and keypoints.xy is not None and len(keypoints.xy) > 0 and keypoints.conf is not None:
            keypoints_xy = keypoints.xy[0].cpu().numpy().tolist()
            keypoints_conf = keypoints.conf[0].cpu().numpy().tolist()

            for k, (x, y) in enumerate(keypoints_xy):
                if keypoints_conf[k] > 0.5:
                    if k == 0:  # mouth
                        mouth = (x, y)
                    elif k == 1:  # left eye
                        left_eye = (x, y)
                    elif k == 2:  # right eye
                        right_eye = (x, y)
                    elif k == 9:  # left wrist
                        left_wrist = (x, y)
                    elif k == 10:  # right wrist
                        right_wrist = (x, y)

    distances = {}
    if mouth and left_wrist:
        distances['left_wrist_to_mouth'] = euclidean(left_wrist, mouth)
    if mouth and right_wrist:
        distances['right_wrist_to_mouth'] = euclidean(right_wrist, mouth)
    if left_eye and right_eye:
        distances['eye_to_eye'] = euclidean(left_eye, right_eye)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 0)

    if mouth and left_wrist:
        cv2.circle(img, (int(left_wrist[0]), int(left_wrist[1])), 5, color, -1)
        cv2.circle(img, (int(mouth[0]), int(mouth[1])), 5, color, -1)
        cv2.line(img, (int(left_wrist[0]), int(left_wrist[1])), (int(mouth[0]), int(mouth[1])), color, 1)
        distance = distances['left_wrist_to_mouth']
        mid_point = ((int(left_wrist[0]) + int(mouth[0])) // 2, (int(left_wrist[1]) + int(mouth[1])) // 2)
        cv2.putText(img, f"{int(distance)}px", mid_point, font, font_scale, (255, 0, 0), thickness)

    if mouth and right_wrist:
        cv2.circle(img, (int(right_wrist[0]), int(right_wrist[1])), 5, color, -1)
        cv2.circle(img, (int(mouth[0]), int(mouth[1])), 5, color, -1)
        cv2.line(img, (int(right_wrist[0]), int(right_wrist[1])), (int(mouth[0]), int(mouth[1])), color, 1)
        distance = distances['right_wrist_to_mouth']
        mid_point = ((int(right_wrist[0]) + int(mouth[0])) // 2, (int(right_wrist[1]) + int(mouth[1])) // 2)
        cv2.putText(img, f"{int(distance)}px", mid_point, font, font_scale, (255, 0, 0), thickness)

    if left_eye and right_eye:
        cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 5, (0, 0, 255), -1)
        cv2.line(img, (int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1])), (255, 255, 0), 1)
        distance = distances['eye_to_eye']
        print("distance between eyes %f", distance)
        mid_point = ((int(left_eye[0]) + int(right_eye[0])) // 2, (int(left_eye[1]) + int(right_eye[1])) // 2)
        cv2.putText(img, f"{int(distance)}px", mid_point, font, font_scale, (0, 0, 255), thickness)

    return img, distances

def main():
    input_folder = "input_images"
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not os.path.isfile(input_path):
            continue
        try:
            img, results = process_image(input_path)
            annotated_img, distances = annotate_and_analyze(img, results)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, annotated_img)

            brushing = False
            if 'eye_to_eye' in distances:
                eye_distance = distances['eye_to_eye']
                threshold_distance = eye_distance * 4  # threshold is 4x the eye-to-eye distance
                if ('left_wrist_to_mouth' in distances and distances['left_wrist_to_mouth'] < threshold_distance) or \
                   ('right_wrist_to_mouth' in distances and distances['right_wrist_to_mouth'] < threshold_distance):
                    brushing = True

            print(f"{filename}: {'likely brushing teeth' if brushing else 'not brushing teeth'}")

        except Exception as e:
            log.exception(f"Failed to process image {filename}: {e}")

if __name__ == "__main__":
    main()