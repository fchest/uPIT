import scipy
import numpy as np
import math,os

wav_path = '/data1/NLPRMNT/fancunhang/wsj/wav/wsj0/data/2speakers/wav8k/min/tt/mix/'

for wav_path,_,files_wav in (os.walk(wav_path)):
    print('wav_path:%s'%wav_path)
total_files=len(files_wav)
np.savetxt('tr.lst',files_wav,fmt='%s')
print('successed!')


