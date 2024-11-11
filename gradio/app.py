import gradio as gr
import json
import os
import random
# file_path = os.path.join(os.getcwd(), 'reference_text.json')
# print(os.getcwd())
# debug的话是在根目录下运行的，而运行时就可以进入我的工作目录 此问题怎么解决？

with open('/home/lzh/DigitalHuman/gradio/reference_text.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# print(data)
# print(data['sentences'][0])

random_number = random.randint(0, 19)

def toggle_text_input(selected_option):
    # 如果选择了某个特定选项，显示输入框，否则隐藏
    if selected_option == "有误":
        return gr.update(visible=True)  # 显示输入框
    else:
        return gr.update(visible=False)  # 隐藏输入框

def clear_input_video(input_video):
    # 清空视频输入框
    return None  # 返回 None 会清空视频输入框

def submit_video(input_video, output_text):
    # 提交视频时的处理逻辑
    if input_video is not None:
        return f"视频已提交，长度为：{len(input_video)}秒", input_video  # 假设返回的是视频的长度信息和视频
    else:
        return "没有输入视频", None

with gr.Blocks(css=".center-text { text-align: center; }") as demo:
    gr.Markdown("<h1 class='center-text'>DigitalHuman</h1>", elem_id="title")
    gr.Markdown("<p class='center-text'>此界面接受一个视频输入、一个文本输入，并提供三个滑块用于参数调整。输出包括处理后的视频和参考文本。</p>")
    
    with gr.Row():
        with gr.Column(scale=1):  # 左边的输入组件
            input_video = gr.Video(label="输入视频", elem_id="input_video")
            # 添加清空按钮和提交按钮
            with gr.Row():
                clear_button = gr.Button("清空视频")
                submit_button = gr.Button("提交视频")
            output_text = gr.Textbox(label="输出文本", placeholder="请输入您想让虚拟人所说的一段话", lines=5, submit_btn=True)
            
            with gr.Accordion("位姿表情参数", open=False):
                slider1 = gr.Slider(value=1, minimum=1, maximum=100, label="pose_weight")
                slider2 = gr.Slider(value=1, minimum=1, maximum=100, label="face_weight")
                slider3 = gr.Slider(value=1, minimum=1, maximum=100, label="lip_weight")
            
            # # 隐藏的文本框，默认不可见
            # correct_translation = gr.Textbox(
            #     label="更正栏", 
            #     placeholder="若输入视频的字幕（参考文本）有错误，请在此提交正确的翻译", 
            #     submit_btn=True, 
            #     visible=False,  # 默认不可见
            #     lines=5, 
            #     interactive=True
            # )

            
            # 为按钮绑定回调函数
            clear_button.click(clear_input_video, inputs=input_video, outputs=input_video)  # 清空视频输入
            submit_button.click(submit_video, inputs=[input_video, output_text], outputs=[output_text, input_video])  # 提交视频

        with gr.Column(scale=1):  # 右边的输出组件
            output_video = gr.Video(label="输出视频", interactive=False)
            
            with gr.Column(scale=2):  # 参考文本列，占 2 份空间
                reference_text = gr.Textbox(label="参考文本",value=data['sentences'][random_number], lines=2, interactive=False)
            
            # with gr.Column(scale=1):  # 选择框列，占 1 份空间
            #     dropdown = gr.Dropdown(
            #         label="参考字幕是否正确？", 
            #         choices=["正确", "有误"], 
            #         value="正确"  # 设置默认选项
            #     )
                # dropdown.change(toggle_text_input, dropdown, correct_translation)  # 根据选择显示或隐藏参考文本框
                
            
    demo.queue()
    demo.launch(share=True, inline=True)
