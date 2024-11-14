import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TTS.tts_wrapper import TTSWrapper


if __name__ == '__main__':
    ref_audio = "recordings/tts/ref_audio/test_ref.wav"
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "No one ever called me father nature."

    tts = TTSWrapper(
        model_type="F5-TTS", 
        ckpt_file="", 
        vocab_file="", 
        output_dir="recordings/tts/gen_audio/", 
        remove_silence=False, 
        speed=1.0, 
        load_vocoder_from_local=True
    )

    generated_wave = tts(ref_audio, ref_text, gen_text)
