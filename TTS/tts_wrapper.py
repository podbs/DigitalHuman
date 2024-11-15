import os
import re
import numpy as np
import soundfile as sf
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'F5_TTS/src'))
from pathlib import Path
from cached_path import cached_path
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

class TTSWrapper:
    def __init__(self, model_type="F5-TTS", ckpt_file="", vocab_file="", output_dir="output", remove_silence=False, speed=1.0, load_vocoder_from_local=False):
        """
        Initializes the TTSWrapper with the necessary configurations and models.

        Args:
            model_type (str): Type of model to use ("F5-TTS" or "E2-TTS").
            ckpt_file (str): Path to the checkpoint file.
            vocab_file (str): Path to the vocab file.
            output_dir (str): Directory to save the output audio file.
            remove_silence (bool): Whether to remove silence from the generated audio.
            speed (float): Speed of the generated audio (default is 1.0).
            load_vocoder_from_local (bool): Whether to load the vocoder from a local path.
        """
        # Set up model, checkpoint, and vocab file
        self.model_type = model_type
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.output_dir = output_dir
        self.remove_silence = remove_silence
        self.speed = speed
        self.vocos_local_path = "./checkpoints/tts/charactr/vocos-mel-24khz"

        # Load vocoder
        self.vocoder = load_vocoder(is_local=load_vocoder_from_local, local_path=self.vocos_local_path)

        # Load models based on model type
        if self.model_type == "F5-TTS":
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            if self.ckpt_file == "":
                repo_name = "tts"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                # self.ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
                # self.ckpt_file = str(cached_path(f"checkpoints/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
                self.ckpt_file = str(cached_path(f"checkpoints/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))
        elif self.model_type == "E2-TTS":
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            if self.ckpt_file == "":
                repo_name = "tts"
                exp_name = "E2TTS_Base"
                ckpt_step = 1200000
                # self.ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
                # self.ckpt_file = str(cached_path(f"checkpoints/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
                self.ckpt_file = str(cached_path(f"checkpoints/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Load model
        self.model = load_model(model_cls, model_cfg, self.ckpt_file, self.vocab_file)

    def __call__(self, ref_audio, ref_text, gen_text):
        """
        Generates speech from the input reference audio and text.

        Args:
            ref_audio (str): Path to the reference audio file.
            gen_text (str): Text to generate the speech from.

        Returns:
            final_wave (np.array): The generated audio waveform.
        """
        voices = {"main": {"ref_audio": ref_audio, "ref_text": ref_text}}

        # Preprocess reference audio and text
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )

        # Process text and generate audio
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, gen_text)
        reg2 = r"\[(\w+)\]"
        
        for text in chunks:
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                voice = "main"

            if voice not in voices:
                voice = "main"

            text = re.sub(reg2, "", text).strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]

            # Perform inference
            audio, final_sample_rate, _ = infer_process(
                ref_audio, ref_text, text, self.model, self.vocoder, speed=self.speed
            )
            generated_audio_segments.append(audio)

        # Concatenate generated audio segments
        final_wave = np.concatenate(generated_audio_segments)

        # Save output if needed
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # wave_path = Path(self.output_dir) / "generated_output.wav"
        wave_path = Path(self.output_dir) / "output_audio.wav"
        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)

            # Remove silence if specified
            if self.remove_silence:
                remove_silence_for_generated_wav(f.name)

        print(f"Generated audio saved to: {wave_path}")
        return final_wave