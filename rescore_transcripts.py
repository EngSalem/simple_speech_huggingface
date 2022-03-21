from jiwer import wer
from datasets import load_dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model ', action='store', dest='model_type',
                    help='', default='facebook/s2t-small-librispeech-asr')

args = parser.parse_args()

dataset_name = "librispeech_asr"
model_name = args.model_type
ds = load_dataset(dataset_name, "clean", split="validation")

ground_truth = [t.lower() for t in ds['text']]
transcripts = [t.strip() for t in open(f"wer_{model_name.split('/')[-1]}_validation",'r').readlines()]

print("wer ", wer(ground_truth,transcripts))
