import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf
from jiwer import wer

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model ', action='store', dest='model_type',
                    help='', default='facebook/s2t-small-librispeech-asr')

args = parser.parse_args()


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

model_name = args.model_type
dataset_name = "librispeech_asr"

print('Loading model ...')
model = Speech2TextForConditionalGeneration.from_pretrained(model_name)
processor = Speech2TextProcessor.from_pretrained(model_name)

print('Loading librispeech validation set ...')
ds = load_dataset(dataset_name, "clean", split="validation")
ds = ds.map(map_to_array)

print('Start decoding ...')
transcriptions = []
for audio_file in ds["speech"]:
    inputs = processor(audio_file, sampling_rate=16_000, return_tensors="pt")
    generated_ids = model.generate(inputs=inputs["input_features"], attention_mask=inputs["attention_mask"])
    transcriptions.append(processor.batch_decode(generated_ids))

print('Load ground truth ...')
ground_truth = [t.lower() for t in ds['text']]

with open(f"wer_{model_name}_validation",'w') as writer:
     writer.write(f"WER : {str(wer(ground_truth,transcriptions))}")

print('Dumping transcriptions of the model')
with open(f"transcripts_{model_name}_validation",'w') as writer:
    for transcript in transcriptions:
        writer.write(transcript+'\n')



