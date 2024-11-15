import cv2
# import ffmpeg
from moviepy.editor import VideoFileClip

class VideoProcessor:
    def __init__(self, video_path, output_image_path):
        self.video_path = video_path
        self.output_image_path = output_image_path
    
    def process_video(self):
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频文件：{self.video_path}")
            print(f"错误信息：{cap.get(cv2.CAP_PROP_FOURCC)}")  # 获取视频编码信息
            return False

        # 读取第一帧
        ret, frame = cap.read()
        
        if ret:
            # 保存第一帧为图片
            cv2.imwrite(self.output_image_path, frame)
            print(f"第一帧已保存到：{self.output_image_path}")
        else:
            print("无法读取视频帧")
            return False

        # 释放视频对象
        cap.release()
        return True
    def video_to_audio(self, input_file, output_file):

        try:
            # 加载MP4视频
            video = VideoFileClip(input_file)
            
            # 提取音频
            audio = video.audio
            
            # 保存音频为WAV格式
            audio.write_audiofile(output_file)
            
            print("音频提取并保存为WAV格式成功！")
            return True
        
        except Exception as e:
            print(f"发生错误：{e}")
            return False


    

