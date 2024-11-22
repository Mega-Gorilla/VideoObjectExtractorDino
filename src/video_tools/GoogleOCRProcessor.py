from google.cloud import vision
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import re
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

@dataclass
class SubtitleEntry:
    index: int
    start_time: str
    end_time: str
    text: str

@dataclass
class TextBlock:
    text: str
    confidence: float
    height: int
    width: int
    vertices: List[Dict[str, int]]

class GoogleOCRProcessor:
    def __init__(self, auth_credentials: str, use_api_key: bool = False,
                 min_height_percent: float = 3.0):
        if use_api_key:
            self.client = vision.ImageAnnotatorClient(
                client_options={"api_key": auth_credentials}
            )
        else:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_credentials
            self.client = vision.ImageAnnotatorClient()
        
        self.min_height_percent = min_height_percent

    def _get_text_block_info(self, text_annotation) -> TextBlock:
        vertices = text_annotation.bounding_poly.vertices
        height = max(v.y for v in vertices) - min(v.y for v in vertices)
        width = max(v.x for v in vertices) - min(v.x for v in vertices)
        
        vertices_list = [{"x": v.x, "y": v.y} for v in vertices]
        
        return TextBlock(
            text=text_annotation.description,
            confidence=0.0,
            height=height,
            width=width,
            vertices=vertices_list
        )

    def _is_valid_text_block(self, block_info: TextBlock) -> bool:
        if block_info.confidence < self.min_confidence:
            return False

        if block_info.height < self.min_box_height:
            return False

        if self.target_region:
            top, bottom, left, right = self.target_region
            if not (top <= block_info.center_y <= bottom and 
                   left <= block_info.center_x <= right):
                return False

        return True

    def process_single_image(self, image_path: str) -> Dict:
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)

            if response.error.message:
                print(f"Error processing {image_path}: {response.error.message}")
                return None

            import cv2
            img = cv2.imread(image_path)
            image_height, image_width = img.shape[:2]

            all_blocks = []
            
            for text_annotation in response.text_annotations[1:]:
                block_info = self._get_text_block_info(text_annotation)
                
                block_data = {
                    "text": block_info.text,
                    "debug_info": {
                        "height": block_info.height,
                        "width": block_info.width,
                        "height_percent": (block_info.height / image_height) * 100,
                        "width_percent": (block_info.width / image_width) * 100,
                        "vertices": block_info.vertices,
                    }
                }
                
                height_percent = (block_info.height / image_height) * 100
                if height_percent >= self.min_height_percent:
                    all_blocks.append(block_data)

            language_info = []
            if hasattr(response, 'full_text_annotation') and response.full_text_annotation.pages:
                for lang in response.full_text_annotation.pages[0].property.detected_languages:
                    language_info.append({
                        "language_code": lang.language_code,
                        "confidence": lang.confidence
                    })

            return {
                "image_info": {
                    "height": image_height,
                    "width": image_width,
                },
                "filter_settings": {
                    "min_height_percent": self.min_height_percent,
                },
                "language_info": language_info,
                "text_blocks": all_blocks
            }

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict[str, Dict]:
        results = {}
        path = Path(directory_path)
        
        # 対象ファイルの総数を取得
        total_files = sum(len(list(path.glob(f"*{ext}"))) for ext in extensions)
        
        # tqdmでプログレスバーを表示
        with tqdm(total=total_files, desc="Processing images") as pbar:
            for ext in extensions:
                for image_path in path.glob(f"*{ext}"):
                    ocr_result = self.process_single_image(str(image_path))
                    if ocr_result:
                        results[image_path.name] = ocr_result
                    pbar.update(1)
                        
        return results

    def save_results(self, results: Dict[str, Dict], output_path: str) -> None:
        # 出力ディレクトリのパスを取得
        output_dir = os.path.dirname(output_path)
        
        # ディレクトリが存在しない場合は作成
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 出力パスがディレクトリの場合はエラーを発生
        if os.path.isdir(output_path):
            raise ValueError(f"Output path '{output_path}' is a directory. Please specify a file path.")
        
        # 結果を保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _parse_timestamp_from_filename(self, filename: str) -> tuple:
        pattern = r"(\d{2}-\d{2}-\d{2},\d{3})\s*-->\s*(\d{2}-\d{2}-\d{2},\d{3})"
        match = re.search(pattern, filename)
        if match:
            start_time = match.group(1).replace('-', ':')
            end_time = match.group(2).replace('-', ':')
            return start_time, end_time
        return None, None

    def generate_srt(self, ocr_results: Dict[str, Dict], output_path: str) -> None:
        subtitle_entries = []
        index = 1

        for filename, result in tqdm(ocr_results.items(), desc="Generating SRT"):
            start_time, end_time = self._parse_timestamp_from_filename(filename)
            if start_time and end_time and result["text_blocks"]:
                # スペースを除去して結合
                texts = [block["text"].strip() for block in result["text_blocks"]]
                combined_text = "".join(texts)
                
                entry = SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=combined_text
                )
                subtitle_entries.append(entry)
                index += 1

        # タイムスタンプでソート
        subtitle_entries.sort(key=lambda x: datetime.strptime(x.start_time, '%H:%M:%S,%f'))
        
        # インデックスを振り直し
        for i, entry in enumerate(subtitle_entries, 1):
            entry.index = i

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in tqdm(subtitle_entries, desc="Writing SRT file"):
                f.write(f"{entry.index}\n")
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{entry.text}\n\n")

if __name__ == '__main__':
    processor = GoogleOCRProcessor('Your API KEY', use_api_key=True)
    
    results = processor.process_directory('./outputs/video_detection_dino/results')
    processor.save_results(results, 'ocr_results.json')
    processor.generate_srt(results, 'subtitles.srt')