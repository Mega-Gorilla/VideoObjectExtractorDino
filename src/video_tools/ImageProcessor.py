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
            numbers = re.findall(r'\d+', filename)
            if len(numbers) >= 4:
                h, m, s, ms = map(int, numbers[:4])
                return h * 3600000 + m * 60000 + s * 1000 + ms
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
        try:
            filenames = [f for f in os.listdir(self.folder_path) if f.endswith(('.jpg', '.png'))]
            if not filenames:
                print("No image files found in directory")
                return []
                
            filenames.sort(key=lambda x: self.get_timestamp_from_filename(x))
            
            timestamp_groups = []
            base_image = cv2.imread(os.path.join(self.folder_path, filenames[0]))
            if base_image is None:
                print(f"Could not read first image: {filenames[0]}")
                return []
                
            current_group = [filenames[0]]
            
            for filename in filenames[1:]:
                filepath = os.path.join(self.folder_path, filename)
                current_image = cv2.imread(filepath)
                
                if current_image is None:
                    continue
                    
                similarity = self.compute_image_similarity(base_image, current_image)
                print(f"Similarity between {current_group[0]} and {filename}: {similarity}")
                
                if similarity > self.similarity_threshold:
                    current_group.append(filename)
                else:
                    if current_group:
                        start_time = self.format_timestamp(self.get_timestamp_from_filename(current_group[0]))
                        end_time = self.format_timestamp(self.get_timestamp_from_filename(current_group[-1]))
                        timestamp_groups.append((start_time, end_time))
                        
                        # Save image with timestamp range as filename
                        new_filename = f"{start_time} --> {end_time}.jpg"
                        new_filename = new_filename.replace(":", "-")  # Replace colons for Windows compatibility
                        shutil.copy2(
                            os.path.join(self.folder_path, current_group[0]),
                            os.path.join(self.output_path, new_filename)
                        )
                        print(f"Saved image: {new_filename}")
                    
                    base_image = current_image
                    current_group = [filename]
            
            if current_group:
                start_time = self.format_timestamp(self.get_timestamp_from_filename(current_group[0]))
                end_time = self.format_timestamp(self.get_timestamp_from_filename(current_group[-1]))
                timestamp_groups.append((start_time, end_time))
                
                new_filename = f"{start_time} --> {end_time}.jpg"
                new_filename = new_filename.replace(":", "-")
                shutil.copy2(
                    os.path.join(self.folder_path, current_group[0]),
                    os.path.join(self.output_path, new_filename)
                )
            
            return timestamp_groups
        except Exception as e:
            print(f"Error in process_images: {e}")
            return []
    
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

# 使用例
if __name__ == "__main__":
    processor = ImageProcessor(
        "input_folder_path",
        "output_folder_path",
        similarity_threshold=0.95
    )
    timestamps = processor.save_timestamps("output_timestamp.txt")