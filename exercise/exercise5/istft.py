# ハミング窓の短時間逆フーリエ変換
# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

from ..exercise3 import ft

def ifft(f):
    n = len(f)
    f = f.conjugate()
    x = ft.fft(f)
    return (1/n) * x.conjugate()

# TODO:余裕があれば短時間逆フーリエ変換を実装し、復元した音声を保存する。