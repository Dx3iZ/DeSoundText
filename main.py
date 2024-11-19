import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice
import numpy as np
import tempfile
import soundfile
from colorama import Fore, Style
import os
import warnings
from datetime import datetime

def log_with_time(message):
    current_time = datetime.now().strftime("[%H:%M:%S]")
    print(f"{Fore.LIGHTBLUE_EX}{current_time} {message}{Style.RESET_ALL}")

log_with_time(f'{Fore.YELLOW}Proqram işə düşür... Hazırlanır.')

# Xəbərdarlıq mesajlarını görməməzliyə vurmaq
warnings.filterwarnings("ignore", category=FutureWarning)


# Cihaz və model ayarları
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
log_with_time(f'{Fore.YELLOW}Model yüklənir...')
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
log_with_time(f'{Fore.YELLOW}Model uğurla yükləndi.')

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)

log_with_time(f'{Fore.YELLOW}Konfiqurasiyası tamamlandı.')

generate_kwargs = {
    "return_timestamps": True,
    "language": "azerbaijani",
    "task": "transcribe",
    "is_multilingual": True,
}

# Ses kaydetme və işlem fonksiyonu
def record_and_process_audio(sr=16000):
    log_with_time(f'{Fore.CYAN}Səs dinlənilir...')
    recording = []
    silence_duration = 0
    max_silence_duration = 4.0  # Sessizlik süresi (artırıldı)
    silence_threshold = 0.004  # Daha düşük bir eşik (daha toleranslı)

    stream = sounddevice.InputStream(samplerate=sr, channels=1, dtype="float32")

    with stream:
        while True:
            audio_chunk, _ = stream.read(int(sr * 0.1))  # 0.1 saniyelik örnekler
            recording.append(audio_chunk)
            rms = np.sqrt(np.mean(np.square(audio_chunk)))

            if rms < silence_threshold:
                silence_duration += 0.1
            else:
                silence_duration = 0

            if silence_duration > max_silence_duration:
                log_with_time(f'{Fore.GREEN}Sessizlik algılandı, səs dayandı.')
                break

    audio = np.concatenate(recording)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    soundfile.write(temp_file.name, audio, sr)
    return temp_file.name

# Ses işleme ve yazıya çevirme
def transcribe_audio(audio_path):
    log_with_time(f'{Fore.YELLOW}Səs işlənir...')

    # forced_decoder_ids parametrlərini sıfırlamaq
    model.config.forced_decoder_ids = None

    # Audio faylını yükləyin və siqnalı alın
    audio, sr = soundfile.read(audio_path)

    # Processor vasitəsilə modeli girişlərə hazırlayın
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding="longest", truncation=False, return_attention_mask=True,)
    inputs = {key: value.to(device).to(torch_dtype) for key, value in inputs.items()}  # Verilən növü uyğunlaşdırın

    # attention_mask əlavə olunur
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, 0, :]).to(device)

    # Modelin generate metodu ilə transkripsiya
    with torch.no_grad():
        result = model.generate(**inputs, **generate_kwargs)

    # Mətnin dekodlanması
    text = processor.batch_decode(result, skip_special_tokens=True)[0]
    log_with_time(f'{Fore.MAGENTA}Mətn Alındı: {Fore.WHITE}{text}')
    return text


# Ana işlem
if __name__ == "__main__":
    try:
        audio_file = record_and_process_audio()
        transcribed_text = transcribe_audio(audio_file)
    finally:
        if audio_file:
            os.remove(audio_file)