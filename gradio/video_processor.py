import cv2
# import ffmpeg
from moviepy.editor import VideoFileClip
from PIL import Image
import face_recognition

class VideoProcessor:
    def __init__(self, video_path, output_image_path):
        self.video_path = video_path
        self.output_image_path = output_image_path
        self.first_frame_saved = False
        
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
            self.first_frame_saved = True
            print(f"第一帧已保存到：{self.output_image_path}")
            
        else:
            print("无法读取视频帧")
            return False

        # 释放视频对象
        cap.release()
        if self.first_frame_saved:
            image = face_recognition.load_image_file(self.output_image_path)
            face_location,  = face_recognition.face_locations(image)
            # print(face_location)
            top, right, bottom, left = face_location
            
            # 寬度（水平方向）的像素數從左到右開始計數
            # 高度（垂直方向）的像素數從上到下開始計數     
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            
            # 手動調整，確保人臉被包括到圖片内
            top -= 150
            bottom += 50
            left -= 50
            right += 50
            length = right - left
            width = bottom - top
            print("length = {}, width = {}".format(length, width))
            print("After cut out Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
            x_shorter = 0 if length > width else 1
            
            if x_shorter:
                gap = (width - length) // 2 
                if (width - length) % 2 == 0:
                    left -= gap
                    right += gap
                else:
                    left -= gap
                    right += (gap + 1)
            else:
                gap = (length - width) // 2
                if(length - width) % 2 == 0:
                    top -= gap
                    bottom += gap
                else:
                    top -= (gap + 1)
                    bottom += gap
                    
            # 查看原圖像的分辨率
            # pil_image_original = Image.fromarray(image)
            # print(pil_image_original.size)
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            # pil_image.show()
            print("输出图像分辨率为：{}".format( pil_image.size))
            pil_image.save(self.output_image_path)
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


    

