import shutil
import gradio as gr
import json
import os
import random
import os
import sys
import time
from video_processor import VideoProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hallo'))
from TTS.tts_wrapper import TTSWrapper

sys.path.append('.')
# print(sys.path)
from ADFA.adfa_wrapper import ADFAWrapper
# print("正在工作的目录是："+os.getcwd())
# print("我想找的是："+os.path.join(os.getcwd(), "recordings/output_aduio/output_audio.wav"))

ref_audio_path = './recordings/ref_aduio/ref_audio.wav'                #用户输入的音频存储的地址
ref_video_path = './gradio/reference_video/saved_video.mp4'                    #用户输入的视频存储的地址
output_image_path = 'recordings/reference_images/user_face.jpg'    #用户正面照存储的地址
output_aduio_path = "recordings/output_audio/output_audio.wav"   #输出音频存储的地址
output_video_path = "recordings/output_video/output_video.mp4"       #输出视频存储的地址

# file_path = os.path.join(os.getcwd(), 'reference_text.json')
# print(os.getcwd())
# debug的话是在根目录下运行的，而运行时就可以进入我的工作目录 此问题怎么解决？
polling_interval = 5

class UserData:
    def __init__(self, ref_text='', output_text='' ,pose_weight=1, face_weight=1, lip_weight=1):
        self.ref_text = ref_text
        self.output_text = output_text
        self.pose_weight = pose_weight
        self.face_weight = face_weight
        self.lip_weight = lip_weight
        
user_data = UserData()

with open('./gradio/reference_text.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# print(data)
# print(data['sentences'][0])

sentences_len = len(data['sentences'])
random_number = random.randint(0, sentences_len)


def clear_video(video):
    # 清空视频输入框
    return None  # 返回 None 会清空视频输入框

def submit_video(input_video):
    # 提交视频时的处理逻辑
    if input_video is not None:
        # return f"视频已提交，长度为：{len(input_video)}秒", input_video  # 假设返回的是视频的长度信息和视频
        reference_video_path = ref_video_path
        # with open(reference_video_path, "wb") as f:
        #     shutil.copyfileobj(input_video, f)
        source_path = input_video
        
        try:
        # 复制视频文件到目标路径
            shutil.copy2(source_path, reference_video_path)  # copy2 保留文件的元数据（如修改时间等）
            
            # 删除原始视频文件
            # os.remove(source_path)
            print(f"视频文件已从 {source_path} 移动到 {reference_video_path} 并删除原文件。")
            os.remove(source_path)
            return None
        except Exception as e:
            print(f"发生错误: {e}")
        
        # print(input_video)
    else:
        return None

def submit_text(output_text):
    user_data.output_text = output_text
    print(output_text)
    return ''
    
def update_slider1(value):
    user_data.face_weight = value
    # return f"face_wight最终值是 {value}"
    print(f"face_wight最终值是 {value} \n")

def update_slider2(value):
    user_data.pose_weight = value
    # return f"pose_weight最终值是 {value}"
    print(f"pose_weight最终值是 {value} \n")
    
def update_slider3(value):
    user_data.lip_weight = value
    # return f"lip_weight最终值是 {value}"
    print(f"lip_weight最终值是 {value} \n")
    
    
def generate_video(ref_text, output_text, pose_weight, face_weight, lip_weight):
    #生成视频(output_video)的函数
    
    #提取人脸图片(output_image)
    processor = VideoProcessor(ref_video_path, output_image_path)
    if processor.process_video():
        print("人脸提取完成")
    else:
        print("人脸提取失败")
    
    #提取音频(ref_audio)
    if processor.video_to_audio(ref_video_path, ref_audio_path):
        print("音频提取成功")
    else:
        print("音频提取失败")
        
    #生成音频(output_audio)
    tts = TTSWrapper(
        model_type="F5-TTS", 
        ckpt_file="", 
        vocab_file="", 
        output_dir="recordings/output_audio/", 
        remove_silence=False, 
        speed=1.0, 
        load_vocoder_from_local=True
    )
    print("ref_text:"+ref_text)
    print("output_text:"+output_text)
    
    generated_wave = tts(ref_audio_path, ref_text, output_text)
    
    #然后生成视频(output_video)
    ADFA_wrapper = ADFAWrapper(config_path='ADFA/config/custom.yaml')
    # time.sleep(polling_interval * 4)  # 等待指定时间后再检查
    while True:
        if os.path.exists(output_aduio_path):
            print("文件存在")
            Output_video_path = ADFA_wrapper(
                source_image_path=output_image_path,
                driving_audio_path=output_aduio_path,
                output_path=output_video_path,
                pose_weight=pose_weight,
                face_weight=face_weight,
                lip_weight=lip_weight
            )
            break  # 如果找到文件，退出轮询
        else:
            print("文件不存在，继续检查...")
        time.sleep(polling_interval)  # 等待指定时间后再次检查
    
    return Output_video_path

def on_generate_button_click(ref_text, output_text, pose_weight, face_weight, lip_weight):
    video_path = generate_video(ref_text, output_text, pose_weight, face_weight, lip_weight)
    # gr.Video.update(value=video_path)  # 更新输出视频

    return gr.Video(value=video_path)

if __name__ == '__main__':
    with gr.Blocks(css=".center-text { text-align: center; }") as demo:
        gr.Markdown("<h1 class='center-text'>DigitalHuman</h1>", elem_id="title")
        gr.Markdown("<p class='center-text'>此界面接受一个视频输入、一个文本输入，并提供三个滑块用于参数调整。输出包括处理后的视频和参考文本。</p>")
        
        with gr.Row():
            with gr.Column(scale=1):  # 左边的输入组件
                input_video = gr.Video(label="输入视频", elem_id="input_video", format='mp4')
                # 添加清空按钮和提交按钮
                with gr.Row():
                    clear_button = gr.Button("清空")
                    submit_button = gr.Button("提交")
                output_text = gr.Textbox(label="输出文本", placeholder="请输入您想让虚拟人所说的一段话", lines=5, submit_btn=True)
                output_text.submit(submit_text, output_text)   #此处把outputtext给设为0了
                
                with gr.Accordion("位姿表情参数", open=False):
                    slider1 = gr.Slider(value=1, minimum=0, maximum=1, label="pose_weight", step=0.05)
                    slider2 = gr.Slider(value=1, minimum=0, maximum=1, label="face_weight", step=0.05)
                    slider3 = gr.Slider(value=1, minimum=0, maximum=1, label="lip_weight", step=0.05)
                
                slider1.change(update_slider1, slider1, None)
                slider2.change(update_slider2, slider2, None)
                slider3.change(update_slider3, slider3, None)
                
                
                # 为按钮绑定回调函数
                clear_button.click(clear_video, inputs=input_video, outputs=input_video)  # 清空视频输入
                submit_button.click(submit_video, inputs=input_video, outputs=input_video)  # 提交视频
            with gr.Column(scale=1):  # 参考文本列
                reference_text = gr.Textbox(label="参考文本",value=data['sentences'][-1], lines=2, interactive=False)
                user_data.ref_text = data['sentences'][random_number]
                output_video = gr.Video(label="输出视频", interactive=False, show_download_button=True)
                with gr.Row():
                    generate_button = gr.Button("生成视频",interactive=True)
                    clear1_button = gr.Button("清空")
            
            
            generate_button.click(on_generate_button_click, inputs=[reference_text, output_text, slider1, slider2, slider3], outputs=output_video)        
            clear1_button.click(clear_video, inputs=output_video, outputs=output_video)  # 清空输出视频
        
        demo.queue()
        demo.launch(share=True, inline=True)
