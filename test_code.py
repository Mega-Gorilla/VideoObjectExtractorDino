import json
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2,os
import numpy as np

def detect_objects(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
    # モデルの読み込み
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

    # 画像の読み込み
    image_source, image = load_image(image_path)

    # オブジェクト検出の実行
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # 結果の注釈付き画像の生成
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # 結果をJSONに変換
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

def extract_frames(video_path, output_dir, frame_rate):
    """
    MP4ファイルから指定されたフレームレートでJPG画像を抽出する
    
    :param video_path: 入力MP4ファイルのパス
    :param output_dir: 出力JPG画像を保存するディレクトリ
    :param frame_rate: 抽出するフレームレート（fps）
    :return: 生成されたJPG画像のパスリスト
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
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

def process_video(video_path, output_dir, text_prompt, frame_rate=1, box_threshold=0.35, text_threshold=0.25):
    """
    ビデオを処理し、各フレームでオブジェクト検出を実行する
    
    :param video_path: 入力MP4ファイルのパス
    :param output_dir: 出力ディレクトリ
    :param text_prompt: 検出するオブジェクトを指定するテキストプロンプト
    :param frame_rate: 処理するフレームレート（デフォルト: 1 fps）
    :param box_threshold: バウンディングボックスの閾値
    :param text_threshold: テキスト閾値
    """
    frames_dir = os.path.join(output_dir, "frames")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # フレームの抽出
    frame_paths = extract_frames(video_path, frames_dir, frame_rate)

    # 各フレームに対してオブジェクト検出を実行
    for i, frame_path in enumerate(frame_paths):
        print(f"フレーム {i+1}/{len(frame_paths)} を処理中...")
        annotated_frame, results = detect_objects(frame_path, text_prompt, box_threshold, text_threshold)

        # 結果の保存
        cv2.imwrite(os.path.join(results_dir, f"annotated_frame_{i:06d}.jpg"), annotated_frame)
        with open(os.path.join(results_dir, f"detection_results_{i:06d}.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"処理が完了しました。結果は {results_dir} に保存されました。")

def main():
    VIDEO_PATH = "videos/test_video.mp4"
    OUTPUT_DIR = "outputs/video_detection"
    TEXT_PROMPT = "texts"
    FRAME_RATE = 1  # 1フレーム/秒で処理
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    process_video(VIDEO_PATH, OUTPUT_DIR, TEXT_PROMPT, FRAME_RATE, BOX_THRESHOLD, TEXT_THRESHOLD)

if __name__ == "__main__":
    main()