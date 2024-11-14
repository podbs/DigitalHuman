import cv2
import ffmpeg

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
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream)
            print(f"音频已成功保存到：{output_file}")
            return True
        except ffmpeg.Error as e:
            print(f"转换视频到音频时出错：{e}")
            return False

    

