import pandas as pd
import os

_trans_model1 = '../transcripts_s2t-medium-librispeech-asr_validation'
_trans_model2 = '../transcripts_s2t-medium-librispeech-asr_validation'
_trans_model3 = '../transcripts_s2t-small-librispeech-asr_validation'


model_1 = [t.strip() for t in open(_trans_model1,'r').readlines()]
model_2 = [t.strip().upper() for t in open(_trans_model2,'r').readlines()]
model_3 = [t.strip() for t in open(_trans_model3,'r').readlines()]

ids =[]
trans_1 = []
trans_2 = []
trans_3 =[]

for f in os.listdir('../dev_sample/'):
    trans_1.append(model_1[int(f.strip('wav_').strip('.wav'))])
    trans_2.append(model_2[int(f.strip('wav_').strip('.wav'))])
    trans_3.append(model_3[int(f.strip('wav_').strip('.wav'))])
    ids.append(f)


pd.DataFrame.from_dict({'file id': ids,'system1': trans_1, 'system2': trans_2, 'system3': trans_3}).to_csv('system_transcripts.csv', index=False)

