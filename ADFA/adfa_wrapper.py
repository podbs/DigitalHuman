import os
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel

import numpy as np
from moviepy.editor import VideoClip, AudioFileClip, VideoFileClip

def tensor_to_video(tensor, output_video_file, audio_source, fps=25):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_file (str): The file path where the output video will be saved.
        audio_source (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 25 fps.
    """
    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()  # convert to [f, h, w, c]
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # to [0, 255]

    def make_frame(t):
        # get index
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]
        
    new_video_clip = VideoClip(make_frame, duration=tensor.shape[0] / fps)
    audio_clip = AudioFileClip(audio_source).subclip(0, tensor.shape[0] / fps)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_file, fps=fps, codec='libx264')
    
class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    """
    def __init__(self, reference_unet, denoising_unet, face_locator, imageproj, audioproj):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self):
        pass

    def get_modules(self):
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }

def process_audio_emb(audio_emb):
    concatenated_tensors = []
    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)] for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
    audio_emb = torch.stack(concatenated_tensors, dim=0)
    return audio_emb

class ADFAWrapper:
    def __init__(self, config_path='./hallo/self.configs/inference/custom.yaml'):
        # 1. Load configuration
        self.config = OmegaConf.load(config_path)

        # 2. Initialize device and dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = self._get_weight_dtype(self.config.weight_dtype)

        # 3. Initialize models and components
        self.vae = AutoencoderKL.from_pretrained(self.config.vae.model_path)
        self.reference_unet = UNet2DConditionModel.from_pretrained(self.config.base_model_path, subfolder="unet")
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.config.base_model_path,
            self.config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(self.config.unet_additional_kwargs),
            use_landmark=False,
        )
        self.face_locator = FaceLocator(conditioning_embedding_channels=320)
        self.image_proj = ImageProjModel(
            cross_attention_dim=self.denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        )
        self.audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=768,
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device=self.device, dtype=self.weight_dtype)

        # Freeze the weights of the models
        self._freeze_modules()

        # Create the network
        self.net = Net(
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            face_locator=self.face_locator,
            imageproj=self.image_proj,
            audioproj=self.audio_proj
        )

        # Load weights for net
        m, u = self.net.load_state_dict(
            torch.load(
                os.path.join(self.config.audio_ckpt_dir, "net.pth"),
                map_location="cpu",
            ),
        )
        assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
        print("loaded weight from ", os.path.join(self.config.audio_ckpt_dir, "net.pth"))

        # Initialize pipeline
        self.pipeline = FaceAnimatePipeline(
            vae=self.vae,
            reference_unet=self.net.reference_unet,
            denoising_unet=self.net.denoising_unet,
            face_locator=self.net.face_locator,
            scheduler=self._initialize_scheduler(),
            image_proj=self.net.imageproj
        ).to(device=self.device, dtype=self.weight_dtype)

    def _get_weight_dtype(self, dtype_str):
        if dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bf16":
            return torch.bfloat16
        elif dtype_str == "fp32":
            return torch.float32
        else:
            return torch.float32

    def _freeze_modules(self):
        self.vae.requires_grad_(False)
        self.reference_unet.requires_grad_(False)
        self.denoising_unet.requires_grad_(False)
        self.face_locator.requires_grad_(False)
        self.image_proj.requires_grad_(False)
        self.audio_proj.requires_grad_(False)

        self.reference_unet.enable_gradient_checkpointing()
        self.denoising_unet.enable_gradient_checkpointing()

    def _initialize_scheduler(self):
        sched_kwargs = OmegaConf.to_container(self.config.noise_scheduler_kwargs)
        if self.config.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        return DDIMScheduler(**sched_kwargs)

    def __call__(self, source_image_path, driving_audio_path, output_path, pose_weight, face_weight, lip_weight, cache_dir=None):
        """
        Run the ADFA inference process to generate a video from a source image and driving audio.

        Args:
            source_image_path (str): Path to the source image (e.g., .jpg, .png).
            driving_audio_path (str): Path to the driving audio file (e.g., .wav).
            output_path (str): Path where the output video will be saved (e.g., .mp4).
            pose_weight (float): The weight for controlling the head pose motion during animation.
            face_weight (float): The weight for controlling the face motion (e.g., expressions).
            lip_weight (float): The weight for controlling the lip motion (e.g., lip-syncing).

        Returns:
            str: The path to the generated output video.

        Example usage:
            wrapper = ADFAWrapper(self.config_path='self.configs/inference/default.yaml')
            output_video_path = wrapper(
                source_image_path="image.jpg",
                driving_audio_path="audio.wav",
                output_path="output.mp4",
                pose_weight=1.0,
                face_weight=1.0,
                lip_weight=1.0
            )

        Detailed description of arguments:
        - `source_image`: The reference image to animate. This image typically contains a face that will be animated based on the audio input.
        - `driving_audio`: The audio file that drives the animation. The audio should be in .wav format, and its content will be used to animate the lips and expressions of the face in the `source_image`.
        - `output_path`: The file path where the generated video will be saved. It should include the desired file name and extension (e.g., "output.mp4").
        - `pose_weight`: This controls the weight of the head motion in the animation. A higher value will result in more exaggerated head movements.
        - `face_weight`: This controls the weight of the facial expressions in the animation. A higher value will result in more pronounced facial expressions.
        - `lip_weight`: This controls the weight of the lip synchronization in the animation. A higher value will result in more precise lip movements to match the audio.

        """
        motion_scale = [pose_weight, face_weight, lip_weight]

        if cache_dir is None:
            cache_dir = os.path.dirname(output_path)
        img_size = (self.config.data.source_image.width,
                    self.config.data.source_image.height)
        clip_length = self.config.data.n_sample_frames
        face_analysis_model_path = self.config.face_analysis.model_path
        with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
            source_image_pixels, \
            source_image_face_region, \
            source_image_face_emb, \
            source_image_full_mask, \
            source_image_face_mask, \
            source_image_lip_mask = image_processor.preprocess(
                source_image_path, cache_dir, self.config.face_expand_ratio)

        # 3.2 prepare audio embeddings
        sample_rate = self.config.data.driving_audio.sample_rate
        assert sample_rate == 16000, "audio sample rate must be 16000"
        fps = self.config.data.export_video.fps
        wav2vec_model_path = self.config.wav2vec.model_path
        wav2vec_only_last_features = self.config.wav2vec.features == "last"
        audio_separator_model_file = self.config.audio_separator.model_path
        with AudioProcessor(
            sample_rate,
            fps,
            wav2vec_model_path,
            wav2vec_only_last_features,
            os.path.dirname(audio_separator_model_file),
            os.path.basename(audio_separator_model_file),
            os.path.join(cache_dir, "audio_preprocess")
        ) as audio_processor:
            audio_emb, audio_length = audio_processor.preprocess(driving_audio_path, clip_length)

        self.pipeline.to(device=self.device, dtype=self.weight_dtype)

        audio_emb = process_audio_emb(audio_emb)

        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)

        source_image_full_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_lip_mask
        ]


        times = audio_emb.shape[0] // clip_length

        tensor_result = []

        generator = torch.manual_seed(42)

        for t in range(times):
            print(f"[{t+1}/{times}]")

            if len(tensor_result) == 0:
                # The first iteration
                motion_zeros = source_image_pixels.repeat(
                    self.config.data.n_motion_frames, 1, 1, 1)
                motion_zeros = motion_zeros.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
            else:
                motion_frames = tensor_result[-1][0]
                motion_frames = motion_frames.permute(1, 0, 2, 3)
                motion_frames = motion_frames[0-self.config.data.n_motion_frames:]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames = motion_frames.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

            audio_tensor = audio_emb[
                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
            ]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=self.net.audioproj.device, dtype=self.net.audioproj.dtype)
            audio_tensor = self.net.audioproj(audio_tensor)

            pipeline_output = self.pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=source_image_face_emb,
                face_mask=source_image_face_region,
                pixel_values_full_mask=source_image_full_mask,
                pixel_values_face_mask=source_image_face_mask,
                pixel_values_lip_mask=source_image_lip_mask,
                width=img_size[0],
                height=img_size[1],
                video_length=clip_length,
                num_inference_steps=self.config.inference_steps,
                guidance_scale=self.config.cfg_scale,
                generator=generator,
                motion_scale=motion_scale,
            )

            tensor_result.append(pipeline_output.videos)

        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :audio_length]

        # save the result after all iteration
        tensor_to_video(tensor_result, output_path, driving_audio_path)
        return output_path