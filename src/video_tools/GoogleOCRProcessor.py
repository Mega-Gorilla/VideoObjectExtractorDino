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

class GoogleOCRProcessor:
    def __init__(self, auth_credentials: str, use_api_key: bool = False):
        if use_api_key:
            self.client = vision.ImageAnnotatorClient(
                client_options={"api_key": auth_credentials}
            )
        else:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_credentials
            self.client = vision.ImageAnnotatorClient()

    def process_single_image(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)

            if response.error.message:
                print(f"Error processing {image_path}: {response.error.message}")
                return None

            texts = response.text_annotations
            return texts[0].description if texts else None

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_directory(self, directory_path: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict[str, str]:
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

    def save_results(self, results: Dict[str, str], output_path: str) -> None:
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

    def generate_srt(self, ocr_results: Dict[str, str], output_path: str) -> None:
        subtitle_entries = []
        index = 1

        # tqdmでプログレスバーを表示
        for filename, text in tqdm(ocr_results.items(), desc="Generating SRT"):
            start_time, end_time = self._parse_timestamp_from_filename(filename)
            if start_time and end_time and text:
                entry = SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text.replace('\n', ' ').strip()
                )
                subtitle_entries.append(entry)
                index += 1

        subtitle_entries.sort(key=lambda x: datetime.strptime(x.start_time, '%H:%M:%S,%f'))

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