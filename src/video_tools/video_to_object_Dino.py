import json
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import numpy as np
import torch
from torchvision.ops import box_convert
from datetime import timedelta
from tqdm import tqdm

class SubtitleDetector:
    def __init__(self, model_config="groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 model_weights="weights/groundingdino_swint_ogc.pth"):
        """
        字幕検出器の初期化
        
        Args:
            model_config (str): モデル設定ファイルのパス
            model_weights (str): モデルの重みファイルのパス
        """
        self.model = load_model(model_config, model_weights)

    def detect_objects(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        画像内のオブジェクトを検出する
        
        Args:
            image_path (str): 入力画像のパス
            text_prompt (str): 検出するテキストのプロンプト
            box_threshold (float): バウンディングボックスの閾値
            text_threshold (float): テキスト検出の閾値
            
        Returns:
            tuple: (注釈付き画像, 検出結果のリスト)
        """
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        results = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            x1, y1, x2, y2 = box
            results.append({
                "label": phrase,
                "confidence": float(logit),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })

        return annotated_frame, results

    def detect_subtitles(self, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        画像内の字幕を検出し、画面下部の字幕領域のみを抽出する
        
        Args:
            image_path (str): 入力画像のパス
            text_prompt (str): 検出するテキストのプロンプト
            box_threshold (float): バウンディングボックスの閾値
            text_threshold (float): テキスト検出の閾値
            
        Returns:
            tuple: (字幕領域の画像, 検出結果のリスト)
        """
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        lower_third_threshold = 2 * h / 3
        filtered_boxes = []
        filtered_logits = []
        filtered_phrases = []
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            if box[1] >= lower_third_threshold and box[3] >= lower_third_threshold:
                filtered_boxes.append(box)
                filtered_logits.append(logit)
                filtered_phrases.append(phrase)
        
        black_background = np.zeros_like(image_source)
        
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box)
            black_background[y1:y2, x1:x2] = image_source[y1:y2, x1:x2]
        
        results = []
        for box, logit, phrase in zip(filtered_boxes, filtered_logits, filtered_phrases):
            x1, y1, x2, y2 = box
            results.append({
                "label": phrase,
                "confidence": float(logit),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })
        
        return black_background, results

    @staticmethod
    def extract_frames(video_path, output_dir, frame_rate):
        """
        動画からフレームを抽出する
        
        Args:
            video_path (str): 入力動画のパス
            output_dir (str): 出力ディレクトリのパス
            frame_rate (float): 抽出するフレームレート
            
        Returns:
            list: 抽出されたフレームのパスとタイムスタンプのタプルのリスト
        """
        os.makedirs(output_dir, exist_ok=True)

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
                timestamp_ms = int((frame_count / fps) * 1000)
                timestamp = str(timedelta(milliseconds=timestamp_ms))
                formatted_timestamp = timestamp.replace(':', '_').replace('.', '_')
                
                frame_path = os.path.join(output_dir, f"frame_{formatted_timestamp}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append((frame_path, timestamp_ms))

            frame_count += 1

        video.release()
        return extracted_frames

    def process_video(self, video_path, output_dir, text_prompt, frame_rate=1, 
                 box_threshold=0.35, text_threshold=0.25,
                 output_json=True):
        """
        動画を処理し、フレームごとに字幕を検出する
        """
        
        frames_dir = os.path.join(output_dir, "frames")
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        frame_paths = self.extract_frames(video_path, frames_dir, frame_rate)

        for frame_path, timestamp_ms in tqdm(frame_paths, desc="フレーム処理中"):
            annotated_frame, results = self.detect_subtitles(
                frame_path, text_prompt, box_threshold, text_threshold)

            hours = timestamp_ms // (3600 * 1000)
            minutes = (timestamp_ms % (3600 * 1000)) // (60 * 1000)
            seconds = (timestamp_ms % (60 * 1000)) // 1000
            milliseconds = timestamp_ms % 1000
            formatted_timestamp = f"{hours:01d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"
            
            cv2.imwrite(os.path.join(results_dir, f"annotated_frame_{formatted_timestamp}.jpg"), 
                    annotated_frame)
            
            if output_json:
                timestamp = str(timedelta(milliseconds=timestamp_ms))
                results_with_timestamp = {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": timestamp,
                    "detections": results
                }
                
                with open(os.path.join(results_dir, f"detection_results_{formatted_timestamp}.json"), "w") as f:
                    json.dump(results_with_timestamp, f, indent=2)

        print(f"処理が完了しました。結果は {results_dir} に保存されました。")


# 使用例
def main():
    detector = SubtitleDetector()
    
    VIDEO_PATH = "videos/sample_video.mp4"
    OUTPUT_DIR = "outputs/video_detection_dino"
    TEXT_PROMPT = "text"
    FRAME_RATE = 0.5
    BOX_THRESHOLD = 0.30
    TEXT_THRESHOLD = 0.20

    detector.process_video(
        VIDEO_PATH, 
        OUTPUT_DIR, 
        TEXT_PROMPT, 
        FRAME_RATE, 
        BOX_THRESHOLD, 
        TEXT_THRESHOLD
    )

if __name__ == "__main__":
    main()