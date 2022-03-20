from jiwer import wer
from datasets import load_dataset


dataset_name = "librispeech_asr"
model_name = "facebook/s2t-small-librispeech-asr"
ds = load_dataset(dataset_name, "clean", split="validation")

ground_truth = [t.lower() for t in ds['text']]
transcripts = [t.strip() for t in open(f"wer_{model_name.split('/')[-1]}_validation",'r').readlines()]

print("wer ", wer(ground_truth,transcripts))
