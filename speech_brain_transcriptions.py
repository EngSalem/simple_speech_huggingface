from speechbrain.pretrained import EncoderDecoderASR
from datasets import load_dataset

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

dataset = load_dataset("librispeech_asr", "clean", split="validation")
dataset = dataset.map(map_to_array)


for audio_file in dataset["speech"]:
    asr_model.transcribe_file(audio_file)