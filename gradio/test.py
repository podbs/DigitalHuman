from video_processor import VideoProcessor
import sys
import os

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ref_video_path = "./gradio/reference_video/saved_video.mp4"
output_iamge_path = "./gradio/test_image.jpg"

image_processor = VideoProcessor(ref_video_path, output_iamge_path)
image_processor.process_video()