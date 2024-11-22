import os
import cv2
import numpy as np
from datetime import timedelta
import re
import shutil

class ImageProcessor:
    def __init__(self, folder_path, output_path, similarity_threshold=0.95):
        self.folder_path = folder_path
        self.output_path = output_path
        self.similarity_threshold = similarity_threshold
        os.makedirs(output_path, exist_ok=True)
        
    def get_timestamp_from_filename(self, filename):
        try:
            match = re.search(r'(?:annotated_frame_|frame_)(\d+)_(\d+)_(\d+)_(\d+)', filename)
            if match:
                h, m, s, ms = map(int, match.groups())
                time_ms = h * 3600000 + m * 60000 + s * 1000 + ms  # ミリ秒を直接使用
                return time_ms
            return 0
        except Exception as e:
            print(f"Error parsing timestamp from {filename}: {e}")
            return 0

    def compute_image_similarity(self, img1, img2):
        try:
            if img1.shape != img2.shape:
                return 0
            diff = cv2.absdiff(img1, img2)
            diff_sum = np.sum(diff)
            max_diff = img1.shape[0] * img1.shape[1] * 255 * 3
            similarity = 1 - (diff_sum / max_diff)
            return similarity
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0
        
    def format_timestamp(self, ms):
        try:
            time = timedelta(milliseconds=ms)
            hours, remainder = divmod(time.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms%1000:03d}"
        except Exception as e:
            print(f"Error formatting timestamp: {e}")
            return "00:00:00,000"
        
    def process_images(self):
        filenames = [f for f in os.listdir(self.folder_path) if f.endswith(('.jpg', '.png'))]
        if not filenames:
            return []
                
        filenames.sort(key=lambda x: self.get_timestamp_from_filename(x))
        
        timestamp_groups = []
        base_image = cv2.imread(os.path.join(self.folder_path, filenames[0]))
        if base_image is None:
            return []
                
        current_group = [filenames[0]]
        base_timestamp = self.get_timestamp_from_filename(filenames[0])
        
        for i, filename in enumerate(filenames[1:]):
            filepath = os.path.join(self.folder_path, filename)
            current_image = cv2.imread(filepath)
            
            if current_image is None:
                continue
                    
            similarity = self.compute_image_similarity(base_image, current_image)
            
            # 黒画像チェックを追加
            is_black = self.is_solid_color(current_image, (0,0,0))
            
            if similarity > self.similarity_threshold and not is_black:
                current_group.append(filename)
            else:
                if current_group:
                    start_time = self.format_timestamp(base_timestamp)
                    end_time = self.format_timestamp(self.get_timestamp_from_filename(current_group[-1]))
                    timestamp_groups.append((start_time, end_time))
                    
                    new_filename = f"{start_time} --> {end_time}.jpg"
                    new_filename = new_filename.replace(":", "-")
                    shutil.copy2(
                        os.path.join(self.folder_path, current_group[0]),
                        os.path.join(self.output_path, new_filename)
                    )
                
                if not is_black:
                    base_image = current_image
                    current_group = [filename]
                    base_timestamp = self.get_timestamp_from_filename(filename)
                else:
                    base_image = None
                    current_group = []
                    
        # 最後のグループの処理
        if current_group:
            start_time = self.format_timestamp(base_timestamp)
            end_time = self.format_timestamp(self.get_timestamp_from_filename(current_group[-1]))
            timestamp_groups.append((start_time, end_time))
            
            new_filename = f"{start_time} --> {end_time}.jpg"
            new_filename = new_filename.replace(":", "-")
            shutil.copy2(
                os.path.join(self.folder_path, current_group[0]),
                os.path.join(self.output_path, new_filename)
            )
        
        return timestamp_groups
    
    def save_timestamps(self, output_file):
        timestamp_groups = self.process_images()
        if not timestamp_groups:
            print("No timestamp groups to save")
            return []
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for start, end in timestamp_groups:
                    f.write(f"{start} --> {end}\n")
            return timestamp_groups
        except Exception as e:
            print(f"Error saving timestamps: {e}")
            return []
        
    def is_solid_color(self, image, target_color, tolerance=30):
        """
        画像が特定の色で塗りつぶされているかチェックする関数
        
        Parameters:
        -----------
        image : np.ndarray
            チェックする画像（OpenCV形式）
        target_color : tuple
            検出する色（BGR形式）。例: (0, 0, 0) for 黒
        tolerance : int
            色の許容誤差（0-255）
            
        Returns:
        --------
        bool
            画像が指定色で塗りつぶされている場合True
        """
        try:
            if image is None:
                return False
                
            # 画像の平均色を計算
            mean_color = cv2.mean(image)[:3]
            
            # 指定色との差分を計算
            color_diff = sum(abs(m - t) for m, t in zip(mean_color, target_color))
            
            # 画像のピクセル値の標準偏差を計算（色のばらつきを確認）
            std_dev = np.std(image, axis=(0, 1)).mean()
            
            # 平均色が目標色に近く、かつ色のばらつきが小さい場合に真と判定
            return color_diff < tolerance and std_dev < tolerance/2
            
        except Exception as e:
            print(f"Error in is_solid_color: {e}")
            return False

    def remove_solid_color_images(self, target_colors,folder_path):
        """
        指定された色で塗りつぶされた画像を削除する関数
        
        Parameters:
        -----------
        target_colors : list of tuple
            削除対象の色のリスト（BGR形式）
            例: [(0, 0, 0), (255, 255, 255)] for 黒と白
        
        Returns:
        --------
        list
            削除された画像のファイル名リスト
        """
        try:
            removed_files = []
            
            # フォルダ内の画像ファイルを取得
            filenames = [f for f in os.listdir(folder_path) 
                        if f.endswith(('.jpg', '.png'))]
                        
            for filename in filenames:
                filepath = os.path.join(folder_path, filename)
                image = cv2.imread(filepath)
                
                if image is None:
                    continue
                    
                # 各対象色についてチェック
                for color in target_colors:
                    if self.is_solid_color(image, color):
                        try:
                            os.remove(filepath)
                            removed_files.append(filename)
                            print(f"Removed {filename} (solid {color} color)")
                            break
                        except Exception as e:
                            print(f"Error removing {filename}: {e}")
            
            return removed_files
            
        except Exception as e:
            print(f"Error in remove_solid_color_images: {e}")
            return []

# 使用例
if __name__ == "__main__":
    processor = ImageProcessor(
        "input_folder_path",
        "output_folder_path",
        similarity_threshold=0.95
    )
    timestamps = processor.save_timestamps("output_timestamp.txt")