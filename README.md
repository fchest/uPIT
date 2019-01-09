# uPIT
Utterance-level Permutation Invariant Training

This project is used for uPIT training of two speakers.

We use Tensorflow(1.0) LSTM(BLSTM) to do PIT.

Reference:

Kolbæk, M., Yu, D., Tan, Z.-H., & Jensen, J. (2017). Multi-talker Speech Separation and Tracing with Permutation Invariant Training of Deep Recurrent Neural Networks, 1–10. Retrieved from http://arxiv.org/abs/1703.06284

Adapted from https://github.com/snsun/pit-speech-separation
Several Improvements：
1.using the cmvn of STFT as the input features
2.the learning rate can be larger, for example 0.0005
