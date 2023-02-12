# -*- coding: utf-8 -*-
"""
# @Time    : 2023/1/8 11:45 上午
# @Author  : HOY
# @Email   : hoytra0@163.com
# @File    : demo.py
"""
import torch
import librosa
import webrtcvad
import wenetruntime as wenet

def vad_excutor(audio):
    vads = []
    pcm_wave = (audio * 2.0 ** 15).astype('int16')
    windows = librosa.util.frame(pcm_wave, frame_length=320, hop_length=320, axis=0)
    for window in windows:
        window = window.tobytes()
        vads.append(vad.is_speech(window, 16000))
    return vads, windows


demo_wav = 'demo.wav'
vad = webrtcvad.Vad(2)
decoder = wenet.Decoder(lang='chs')
audio, sr = librosa.load(demo_wav, sr=16000, mono=False)
vad_0, window_0 = vad_excutor(audio[0])
vad_1, window_1 = vad_excutor(audio[1])
begin_0 = -1
begin_1 = -1
result = []
for i in range(len(vad_0)):
    is_speech_0 = vad_0[i]
    is_speech_1 = vad_1[i]
    if is_speech_0:
        begin_0 = i
    if begin_0 != -1 and is_speech_0 == 0:
        end_0 = i
        frame_0 = window_0[begin_0*640:end_0*640]
        result_0 = decoder.decode(frame_0)
        """
        result_0为json格式数据，需要提取文本
        result_0 = json.loads(......)
        """
        begin_0 = -1
        result.append({'channel':0, 'text':result_0})

    if is_speech_1:
        begin_1 = i
    if begin_1 != -1 and is_speech_1 == 0:
        end_1 = i
        frame_1 = window_0[begin_0*640:end_1*640]
        result_1 = decoder.decode(frame_1)
        ### 同上
        begin_1 = -1
        result.append({'channel':1, 'text':result_1})
print(result)



