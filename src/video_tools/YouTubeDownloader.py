from pytubefix import YouTube, Playlist
from pytubefix.cli import on_progress
import os
from typing import Optional, Dict, Any, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import time
import random

@dataclass
class DownloadResult:
    """ダウンロード結果を格納するデータクラス"""
    url: str
    success: bool
    file_path: Optional[str] = None
    error_message: Optional[str] = None

class YouTubeDownloader:
    """PytubeFixを使用したYouTubeダウンローダークラス"""
    
    def __init__(self, 
                 output_path: str = "./downloads",
                 max_workers: int = 4,
                 use_oauth: bool = False):
        """
        初期化メソッド
        
        Args:
            output_path (str): ダウンロードしたファイルの保存先ディレクトリ
            max_workers (int): 同時ダウンロードの最大数
            use_oauth (bool): OAuth認証を使用するかどうか
        """
        self.output_path = output_path
        self.max_workers = max_workers
        self.use_oauth = use_oauth
        self._setup_logging()
        self._create_output_directory()
    
    def _setup_logging(self) -> None:
        """ロギングの設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_output_directory(self) -> None:
        """出力ディレクトリが存在しない場合は作成"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.logger.info(f"Created output directory: {self.output_path}")

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        動画の情報を取得
        
        Args:
            url (str): YouTube動画のURL
            
        Returns:
            Dict[str, Any]: 動画の情報を含む辞書
        """
        try:
            yt = YouTube(
                url,
                use_oauth=self.use_oauth,
                allow_oauth_cache=True
            )
            return {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "views": yt.views,
                "description": yt.description,
                "publish_date": yt.publish_date,
                "captions": list(yt.captions.keys()) if yt.captions else []
            }
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            raise

    def download_video(self, 
                      url: str, 
                      filename: Optional[str] = None,
                      resolution: Optional[str] = None,
                      output_path: Optional[str] = None) -> str:
        """
        動画をダウンロード
        
        Args:
            url (str): YouTube動画のURL
            filename (Optional[str]): 保存するファイル名
            resolution (Optional[str]): 動画の解像度（指定しない場合は最高画質）
            output_path (Optional[str]): 出力ディレクトリ
            
        Returns:
            str: ダウンロードしたファイルのパス
        """
        # 出力パスの設定
        target_path = output_path or self.output_path
        try:
            yt = YouTube(
                url,
                use_oauth=self.use_oauth,
                allow_oauth_cache=True,
                on_progress_callback=on_progress
            )
            
            self.logger.info(f"Downloading: {yt.title}")
            
            # ストリームの取得
            if resolution:
                stream = yt.streams.filter(progressive=True, res=resolution).first()
                if not stream:
                    self.logger.warning(f"Resolution {resolution} not available. Using highest resolution.")
                    stream = yt.streams.get_highest_resolution()
            else:
                stream = yt.streams.get_highest_resolution()
            
            # ファイル名の設定
            if filename:
                output_filename = filename
            else:
                output_filename = f"{yt.title}_{stream.resolution}"
            output_filename = "".join(c for c in output_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            #すでに同一データがある場合はスキップ
            if os.path.exists(os.path.join(target_path,output_filename)):
                return os.path.join(target_path,output_filename)
            
            # ダウンロードの実行
            output_path = stream.download(
                output_path=target_path,
                filename=output_filename
            )
            
            self.logger.info(f"Download completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            raise

    def download_audio(self, 
                      url: str, 
                      filename: Optional[str] = None,
                      output_path: Optional[str] = None) -> str:
        """
        音声のみをダウンロード
        
        Args:
            url (str): YouTube動画のURL
            filename (Optional[str]): 保存するファイル名
            output_path (Optional[str]): 出力ディレクトリ
            
        Returns:
            str: ダウンロードしたファイルのパス
        """
        try:
            yt = YouTube(
                url,
                use_oauth=self.use_oauth,
                allow_oauth_cache=True,
                on_progress_callback=on_progress
            )
            
            self.logger.info(f"Downloading audio from: {yt.title}")
            
            # 音声ストリームを取得
            stream = yt.streams.get_audio_only()
            if not stream:
                raise ValueError("No audio stream available")
            
            # 出力パスの設定
            target_path = output_path or self.output_path
            
            # ファイル名の設定
            if filename:
                output_filename = filename
            else:
                output_filename = f"{yt.title}_audio"
            output_filename = "".join(c for c in output_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # ダウンロードの実行
            output_path = stream.download(
                output_path=target_path,
                filename=output_filename
            )
            
            self.logger.info(f"Audio download completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading audio: {str(e)}")
            raise

    def download_playlist(self, 
                         playlist_url: str,
                         download_type: str = 'video',
                         resolution: Optional[str] = None,
                         output_path: Optional[str] = None) -> List[DownloadResult]:
        """
        プレイリストをダウンロード
        
        Args:
            playlist_url (str): YouTubeプレイリストのURL
            download_type (str): 'video' または 'audio'
            resolution (Optional[str]): 動画の解像度（videoの場合のみ）
            output_path (Optional[str]): 出力ディレクトリ
            
        Returns:
            List[DownloadResult]: ダウンロード結果のリスト
        """
        try:
            playlist = Playlist(playlist_url)
            self.logger.info(f"Downloading playlist: {playlist.title}")
            
            # プレイリスト用のサブディレクトリを作成
            target_path = output_path or self.output_path
            playlist_dir = os.path.join(target_path, f"playlist_{playlist.title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(playlist_dir, exist_ok=True)
            
            results = []
            
            # プログレスバーの設定
            with tqdm(total=len(playlist.video_urls), desc="Downloading playlist") as pbar:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_url = {}
                    
                    for url in playlist.video_urls:
                        if download_type == 'video':
                            future = executor.submit(
                                self.download_video,
                                url,
                                resolution=resolution,
                                output_path=playlist_dir
                            )
                        else:  # audio
                            future = executor.submit(
                                self.download_audio,
                                url,
                                output_path=playlist_dir
                            )
                        future_to_url[future] = url
                    
                    for future in as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            file_path = future.result()
                            results.append(DownloadResult(
                                url=url,
                                success=True,
                                file_path=file_path
                            ))
                        except Exception as e:
                            results.append(DownloadResult(
                                url=url,
                                success=False,
                                error_message=str(e)
                            ))
                        pbar.update(1)
            
            # 結果のサマリーを表示
            successful = len([r for r in results if r.success])
            self.logger.info(f"Playlist download completed. Success: {successful}/{len(playlist.video_urls)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error downloading playlist: {str(e)}")
            raise

    def download_with_captions(self, 
                             url: str, 
                             caption_code: str = 'a.ja',
                             filename: Optional[str] = None,
                             resolution: Optional[str] = None) -> Tuple[str, str]:
        """
        字幕付きで動画をダウンロード
        
        Args:
            url (str): YouTube動画のURL
            caption_code (str): 字幕の言語コード
            filename (Optional[str]): 保存するファイル名
            resolution (Optional[str]): 動画の解像度
            
        Returns:
            Tuple[str, str]: (動画のパス, 字幕のパス)
        """
        try:
            yt = YouTube(
                url,
                use_oauth=self.use_oauth,
                allow_oauth_cache=True,
                on_progress_callback=on_progress
            )
            
            # 動画のダウンロード
            video_path = self.download_video(url, filename, resolution)
            
            # 字幕のダウンロード
            if caption_code in yt.captions:
                caption = yt.captions[caption_code]
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                caption_path = os.path.join(self.output_path, f"{base_filename}.srt")
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption.generate_srt_captions())
                self.logger.info(f"Caption downloaded: {caption_path}")
            else:
                raise ValueError(f"Caption code '{caption_code}' not available")
            
            return video_path, caption_path
            
        except Exception as e:
            self.logger.error(f"Error downloading video with captions: {str(e)}")
            raise
        
# 使用例
if __name__ == "__main__":

    # インスタンスの作成（OAuth認証付き）
    downloader = YouTubeDownloader(
        output_path="./downloads",
        max_workers=4,
        use_oauth=True  # OAuth認証を使用
    )

    # 単一の動画をダウンロード
    video_path = downloader.download_video(
        "https://www.youtube.com/watch?v=ppKPaPKlQ3Q",
        resolution="720p"
    )

    """    # 音声のみをダウンロード
    audio_path = downloader.download_audio(
        "https://www.youtube.com/watch?v=VIDEO_ID"
    )

    # プレイリストをダウンロード
    results = downloader.download_playlist(
        "https://www.youtube.com/playlist?list=PLAYLIST_ID",
        download_type="video",
        resolution="720p"
    )

    # 字幕付きで動画をダウンロード
    video_path, caption_path = downloader.download_with_captions(
        "https://www.youtube.com/watch?v=VIDEO_ID",
        caption_code="a.ja",  # 日本語字幕
        resolution="1080p"
    )

    # 動画情報の取得
    info = downloader.get_video_info("https://www.youtube.com/watch?v=VIDEO_ID")
    print(f"Title: {info['title']}")
    print(f"Author: {info['author']}")
    print(f"Available captions: {info['captions']}")
    """
