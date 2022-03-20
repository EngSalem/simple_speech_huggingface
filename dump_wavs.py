from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

dataset_name = "librispeech_asr"

print('Loading librispeech validation set ...')
ds = load_dataset(dataset_name, "clean", split="validation")

ds = ds.map(map_to_array)

for ix, audio_file in enumerate(ds["speech"]):
    #print(type(audio_file))
    write(f"./dev_wavs/wav_{ix}.wav",16000, np.array(audio_file))

