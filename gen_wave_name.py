import scipy
import numpy as np
import math,os

wav_path = '/data1/fancunhang/speech_enhancement/speech_separation/WSJ0/WSJ0_data_wav/wav8k/min/cv/mix/'

for wav_path,_,files_wav in (os.walk(wav_path)):
    print('wav_path:%s'%wav_path)
total_files=len(files_wav)
np.savetxt('lists/cv_tf.txt',files_wav,fmt='%s')
print('successed!')


