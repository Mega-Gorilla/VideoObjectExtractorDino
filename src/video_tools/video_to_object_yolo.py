import json
import os
from ultralytics import YOLO
import cv2
import numpy as np

def detect_objects(image_path, model, conf_threshold=0.25):
    # 画像の読み込み
    image = cv2.imread(image_path)
    
    # オブジェクト検出の実行
    results = model(image)[0]
    
    # 結果の処理
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        class_id = int(class_id)
        if score > conf_threshold:
            detections.append({
                "label": results.names[class_id],
                "confidence": float(score),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })
    
    # 結果の注釈付き画像の生成
    annotated_frame = results.plot()
    
    return annotated_frame, detections

def extract_frames(video_path, output_dir, frame_rate):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    frame_interval = int(fps / frame_rate)
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        frame_count += 1
    
    video.release()
    print(f"{len(extracted_frames)}フレームを抽出しました。")
    return extracted_frames

def process_video(video_path, output_dir, model_path, frame_rate=1, conf_threshold=0.25):
    frames_dir = os.path.join(output_dir, "frames")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # YOLOv8モデルの読み込み
    model = YOLO(model_path)
    
    # フレームの抽出
    frame_paths = extract_frames(video_path, frames_dir, frame_rate)
    
    # 各フレームに対してオブジェクト検出を実行
    for i, frame_path in enumerate(frame_paths):
        print(f"フレーム {i+1}/{len(frame_paths)} を処理中...")
        annotated_frame, results = detect_objects(frame_path, model, conf_threshold)
        
        # 結果の保存
        cv2.imwrite(os.path.join(results_dir, f"annotated_frame_{i:06d}.jpg"), annotated_frame)
        with open(os.path.join(results_dir, f"detection_results_{i:06d}.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"処理が完了しました。結果は {results_dir} に保存されました。")

def main():
    VIDEO_PATH = "videos/sample_video.mp4"
    OUTPUT_DIR = "outputs/video_detection_yolo"
    MODEL_PATH = "weights/yolov8n.pt"  # または他のYOLOv8モデルのパス
    FRAME_RATE = 1  # 1フレーム/秒で処理
    CONF_THRESHOLD = 0.25
    
    process_video(VIDEO_PATH, OUTPUT_DIR, MODEL_PATH, FRAME_RATE, CONF_THRESHOLD)

if __name__ == "__main__":
    main()