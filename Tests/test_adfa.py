import sys
sys.path.append('.')
from ADFA.adfa_wrapper import ADFAWrapper

if __name__ == '__main__':
    wrapper = ADFAWrapper(config_path='ADFA/config/custom.yaml')
    # output_video_path = wrapper(
    #     source_image_path="recordings/adfa/ref_image/jake.png",
    #     driving_audio_path="recordings/adfa/tgt_audio/test_audio.wav",
    #     output_path="recordings/adfa/gen_animate/output.mp4",
    #     pose_weight=1.0,
    #     face_weight=1.0,
    #     lip_weight=1.0
    # )
    output_video_path = wrapper(
        source_image_path="recordings/reference_images/user_face.jpg",
        driving_audio_path="recordings/output_audio/output_audio.wav",
        output_path="recordings/output_video/output.mp4",
        pose_weight=1.0,
        face_weight=1.0,
        lip_weight=1.0
    )